"""Deterministic lexical Jaccard and random baselines, with full score traces."""

import random
import re
from .storage import digest
from .taxonomy import messages_for


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


class Retriever:
    def __init__(self, store, corpus_id):
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

    def select(self, query, config, taxonomy, counter):
        method = config["method"]
        budget = (
            config["model"]["context_tokens"]
            - config["model"]["max_tokens"]
            - config["model"]["safety_tokens"]
        )
        evidence = config["require_evidence"]
        base = messages_for(query["raw_text"], [], taxonomy, evidence)
        base_tokens = counter.count(base)
        if base_tokens > budget:
            raise ValueError("input exceeds configured token budget")
        if method == "zero-shot":
            return [], [], {"method": method, "full_scores": []}, base
        scores = None
        if method == "lexical":
            q = words(query["raw_text"])
            scores = [len(q & t) / len(q | t) if q | t else 0.0 for t in self.tokens]
            order = sorted(
                range(len(self.rows)),
                key=lambda i: (-scores[i], self.rows[i]["source_id"]),
            )
        else:
            order = list(range(len(self.rows)))
            random.Random(digest([config["seed"], query["text_sha256"]])).shuffle(order)
        selected, candidates, seen = [], [], set()
        for rank, i in enumerate(order[: config["candidate_count"]], 1):
            row = self.rows[i]
            reason = "selected"
            count = (
                counter.count(
                    messages_for(query["raw_text"], [row], taxonomy, evidence)
                )
                - base_tokens
            )
            if config["deduplicate"] and (
                row["normalized_text_sha256"] == query["normalized_text_sha256"]
                or row["normalized_text_sha256"] in seen
            ):
                reason = "duplicate_text"
            elif len(selected) >= config["k"]:
                reason = "beyond_k"
            elif (
                counter.count(
                    messages_for(
                        query["raw_text"], selected + [row], taxonomy, evidence
                    )
                )
                > budget
            ):
                reason = "token_budget"
            if reason == "selected":
                selected.append(row)
                seen.add(row["normalized_text_sha256"])
            candidates.append(
                {
                    "sample_id": row["sample_id"],
                    "stage": "selection",
                    "candidate_rank": rank,
                    "similarity_score": None if scores is None else scores[i],
                    "rerank_score": None,
                    "score_components_json": {"metric": "jaccard"}
                    if scores is not None
                    else {"seed": config["seed"]},
                    "decision": "selected" if reason == "selected" else "excluded",
                    "reason_code": reason,
                    "selected_rank": len(selected) if reason == "selected" else None,
                    "example_tokens": max(0, count),
                }
            )
        trace = {
            "method": method,
            "score_dtype": "python-float64" if scores else None,
            "corpus_sample_ids": [r["sample_id"] for r in self.rows],
            "full_scores": scores,
            "candidate_order": [self.rows[i]["sample_id"] for i in order],
            "selected_sample_ids": [r["sample_id"] for r in selected],
            "input_tokens": counter.count(
                messages_for(query["raw_text"], selected, taxonomy, evidence)
            ),
            "token_budget": budget,
        }
        return (
            selected,
            candidates,
            trace,
            messages_for(query["raw_text"], selected, taxonomy, evidence),
        )
