"""Deterministic train-only retrieval, including a bounded contrastive policy."""

import random
import re

from .storage import digest
from .taxonomy import messages_for

# Semantic method priors, not measured model confusion or co-occurrence edges.
CONTRAST_PAIRS = (
    ("anger", "annoyance"),
    ("annoyance", "disapproval"),
    ("admiration", "approval"),
    ("joy", "excitement"),
    ("joy", "relief"),
    ("sadness", "disappointment"),
    ("sadness", "grief"),
    ("fear", "nervousness"),
    ("curiosity", "confusion"),
    ("realization", "surprise"),
    ("remorse", "embarrassment"),
)


def example_policy_snapshot(version):
    if version == "ranked":
        return {"version": version, "rule": "original similarity or random order"}
    if version != "contrastive-v1":
        raise ValueError("unsupported example policy")
    return {
        "version": version,
        "pairs": CONTRAST_PAIRS,
        "pair_source": "semantic priors; not empirical confusion",
        "rule": "first admissible anchor, highest-ranked positive-overlap counterpart within candidate pool, remaining lexical order",
        "counterpart": "contains adjacent label absent from anchor; excludes corresponding anchor label",
        "fallback": "original order if no admissible contrast; preserve all native labels",
    }


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


class Retriever:
    def __init__(self, store, corpus_id):
        self.store, self.corpus_id = store, corpus_id
        self.semantic_resources = None
        rows = store.rows(
            "SELECT s.* FROM sample_set_members m JOIN samples s USING(sample_id) WHERE sample_set_id=? ORDER BY ordinal",
            (corpus_id,),
        )
        label_rows = store.rows(
            "SELECT l.sample_id,d.label_name FROM sample_set_members m JOIN sample_labels l USING(sample_id) JOIN label_definitions d ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE m.sample_set_id=? ORDER BY l.source_label_position",
            (corpus_id,),
        )
        labels = {r["sample_id"]: [] for r in rows}
        for row in label_rows:
            labels[row["sample_id"]].append(row["label_name"])
        self.rows = [{**r, "labels": labels[r["sample_id"]]} for r in rows]
        self.tokens = [words(r["raw_text"]) for r in rows]

    def _contrast_order(self, pool, scores, query, config, render, counter, budget):
        def is_query_copy(row):
            return (
                config["deduplicate"]
                and row["normalized_text_sha256"] == query["normalized_text_sha256"]
            )

        anchor = next(
            (
                i
                for i in pool
                if not is_query_copy(self.rows[i])
                and counter.count(render([self.rows[i]])) <= budget
            ),
            None,
        )
        if anchor is None or scores[anchor] <= 0:
            return pool, None
        row = self.rows[anchor]
        labels = set(row["labels"])
        boundaries = [
            (a, b)
            for pair in CONTRAST_PAIRS
            for a, b in (pair, pair[::-1])
            if a in labels and b not in labels
        ]
        for i in pool:
            other = self.rows[i]
            if i == anchor or scores[i] <= 0 or is_query_copy(other):
                continue
            if (
                config["deduplicate"]
                and other["normalized_text_sha256"] == row["normalized_text_sha256"]
            ):
                continue
            boundary = next(
                (
                    (a, b)
                    for a, b in boundaries
                    if b in other["labels"] and a not in other["labels"]
                ),
                None,
            )
            if boundary and counter.count(render([row, other])) <= budget:
                return [anchor, i] + [j for j in pool if j not in (anchor, i)], {
                    "anchor_sample_id": row["sample_id"],
                    "contrast_sample_id": other["sample_id"],
                    "anchor_label": boundary[0],
                    "contrast_label": boundary[1],
                }
        return pool, None

    def select(self, query, config, taxonomy, counter):
        method = config["method"]
        budget = (
            config["model"]["context_tokens"]
            - config["model"]["max_tokens"]
            - config["model"]["safety_tokens"]
        )
        evidence = config["require_evidence"]
        version = config.get("prompt_version", "native-labels-v1")
        policy = config.get("example_policy", "ranked")

        def render(examples):
            return messages_for(
                query["raw_text"], examples, taxonomy, evidence, version
            )

        base = render([])
        base_tokens = counter.count(base)
        if base_tokens > budget:
            raise ValueError("input exceeds configured token budget")
        if method == "zero-shot":
            return [], [], {"method": method, "full_scores": []}, base
        scores = None
        semantic_trace = {}
        if method == "lexical":
            q = words(query["raw_text"])
            scores = [len(q & t) / len(q | t) if q | t else 0.0 for t in self.tokens]
            order = sorted(
                range(len(self.rows)),
                key=lambda i: (-scores[i], self.rows[i]["source_id"]),
            )
        elif method == 'semantic':
            from .embeddings import load_index, load_query_bundle
            identity = (config['embedding']['index_id'], config['embedding']['queries_artifact_id'])
            if self.semantic_resources is None or self.semantic_resources[0] != identity:
                index = load_index(self.store, identity[0], self.corpus_id)
                if [r['sample_id'] for r in index['records']] != [r['sample_id'] for r in self.rows]:
                    raise ValueError('semantic corpus order mismatch')
                queries = load_query_bundle(self.store, identity[1], index)
                self.semantic_resources = (identity, index, queries)
            _, index, queries = self.semantic_resources
            entry = queries.get(query.get('sample_id'))
            if entry is None or entry['text_sha256'] != query['text_sha256'] or digest(query['raw_text'].encode()) != entry['text_sha256']:
                raise ValueError('query text not present in frozen query bundle')
            scores = (index['vectors'] @ entry['vector']).tolist()
            order = sorted(range(len(self.rows)), key=lambda i: (-scores[i], self.rows[i]['source_id']))
            semantic_trace = {'index_id': identity[0], 'queries_artifact_id': identity[1],
                              'query_vector_sha256': entry['vector_sha256'], 'index_fingerprint': index['row']['fingerprint']}
        else:
            order = list(range(len(self.rows)))
            random.Random(digest([config["seed"], query["text_sha256"]])).shuffle(order)
        pool = order[: config["candidate_count"]]
        original_ranks = {i: rank for rank, i in enumerate(pool, 1)}
        contrast_pair = None
        if policy == "contrastive-v1":
            pool, contrast_pair = self._contrast_order(
                pool, scores, query, config, render, counter, budget
            )
        selected, candidates, seen = [], [], set()
        for rank, i in enumerate(pool, 1):
            row = self.rows[i]
            reason = "selected"
            count = counter.count(render([row])) - base_tokens
            if config["deduplicate"] and (
                row["normalized_text_sha256"] == query["normalized_text_sha256"]
                or row["normalized_text_sha256"] in seen
            ):
                reason = "duplicate_text"
            elif len(selected) >= config["k"]:
                reason = "beyond_k"
            elif counter.count(render(selected + [row])) > budget:
                reason = "token_budget"
            if reason == "selected":
                selected.append(row)
                seen.add(row["normalized_text_sha256"])
            components = (
                {"metric": "cosine" if method == 'semantic' else "jaccard"}
                if scores is not None
                else {"seed": config["seed"]}
            )
            components.update(original_rank=original_ranks[i], selection_role="ranked")
            if contrast_pair:
                if row["sample_id"] == contrast_pair["anchor_sample_id"]:
                    components["selection_role"] = "anchor"
                elif row["sample_id"] == contrast_pair["contrast_sample_id"]:
                    components["selection_role"] = "contrast"
                components["contrast_pair"] = contrast_pair
            candidates.append(
                {
                    "sample_id": row["sample_id"],
                    "stage": "selection",
                    "candidate_rank": rank,
                    "similarity_score": None if scores is None else scores[i],
                    "rerank_score": None,
                    "score_components_json": components,
                    "decision": "selected" if reason == "selected" else "excluded",
                    "reason_code": reason,
                    "selected_rank": len(selected) if reason == "selected" else None,
                    "example_tokens": max(0, count),
                }
            )
        trace = {
            "method": method,
            "example_policy": policy,
            "contrast_pair": contrast_pair,
            "selection_order": [self.rows[i]["sample_id"] for i in pool],
            "score_dtype": "float32" if method == 'semantic' else ("python-float64" if scores else None),
            "corpus_sample_ids": [r["sample_id"] for r in self.rows],
            "full_scores": scores,
            "candidate_order": [self.rows[i]["sample_id"] for i in order],
            "selected_sample_ids": [r["sample_id"] for r in selected],
            "input_tokens": counter.count(render(selected)),
            "token_budget": budget,
            **semantic_trace,
        }
        return (
            selected,
            candidates,
            trace,
            render(selected),
        )
