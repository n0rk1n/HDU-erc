"""Catch label-leaking, ineffective reranking, and unrecorded prompt changes."""

import hashlib
import json

import pytest
from test_runner import Counter, response
from test_storage import source_fixture

from emotion_lab.config import validate_config
from emotion_lab.datasets import import_goemotions, make_set, normalize
from emotion_lab.preparation import prepare_run
from emotion_lab.retrieval import Retriever
from emotion_lab.runner import execute_run
from emotion_lab.storage import Store, digest
from emotion_lab.taxonomy import load_taxonomy


@pytest.fixture
def contrast_corpus(tmp_path):
    source = source_fixture(tmp_path / "source")
    train = (
        "late train again\t2\ta1\n"
        "late train today again\t2\ta2\n"
        "late train again really\t2\ta3\n"
        "late train again seriously\t2\ta4\n"
        "late train\t3,10\tb1\n"
        "late bus\t3\tb2\n"
        "wonderful present\t17\tc1\n"
    )
    (source / "train.tsv").write_text(train)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["expected_rows"]["train"] = 7
    manifest["files"]["train.tsv"].update(
        sha256=hashlib.sha256(train.encode()).hexdigest(), bytes=len(train.encode())
    )
    (source / "manifest.json").write_text(json.dumps(manifest))
    with Store(tmp_path / "research", initialize=True) as store:
        ds = import_goemotions(store, source)
        corpus = make_set(
            store,
            {
                "dataset_version_id": ds,
                "name": "train",
                "purpose": "corpus",
                "split": "train",
            },
        )
        yield store, corpus


def query(text):
    return {
        "raw_text": text,
        "text_sha256": digest(text.encode()),
        "normalized_text_sha256": digest(normalize(text).encode()),
    }


def config(**overrides):
    return validate_config(
        dict(
            method="lexical",
            k=4,
            candidate_count=7,
            prompt_version="native-labels-v2",
            example_policy="contrastive-v1",
            model={"context_tokens": 30000, "max_tokens": 100},
            **overrides,
        )
    )


def test_contrast_replaces_redundant_neighbor_preserving_all_native_labels(
    contrast_corpus,
):
    store, corpus = contrast_corpus
    retriever = Retriever(store, corpus)
    cfg = config()
    selected, candidates, trace, messages = retriever.select(
        query("late train again please"), cfg, load_taxonomy(), Counter()
    )
    assert [r["source_id"] for r in selected] == ["a1", "b1", "a2", "a3"]
    assert selected[1]["labels"] == ["annoyance", "disapproval"]
    candidate = next(
        r for r in candidates if r["sample_id"] == selected[1]["sample_id"]
    )
    assert candidate["score_components_json"]["original_rank"] == 5
    assert candidate["score_components_json"]["selection_role"] == "contrast"
    assert trace["contrast_pair"]["anchor_label"] == "anger"
    assert trace["contrast_pair"]["contrast_label"] == "annoyance"
    assert json.loads(messages[0]["content"].split("Training examples:\n")[1])[1][
        "labels"
    ] == ["annoyance", "disapproval"]
    cfg["example_policy"] = "ranked"
    baseline = retriever.select(
        query("late train again please"), cfg, load_taxonomy(), Counter()
    )[0]
    assert [r["source_id"] for r in baseline] == ["a1", "a2", "a3", "a4"]


def test_zero_overlap_does_not_force_arbitrary_contrast_and_query_copy_is_excluded(
    contrast_corpus,
):
    store, corpus = contrast_corpus
    retriever = Retriever(store, corpus)
    cfg = config()
    zero = retriever.select(
        query("unrelated vocabulary"), cfg, load_taxonomy(), Counter()
    )
    assert zero[2]["contrast_pair"] is None
    assert [r["source_id"] for r in zero[0]] == ["a1", "a2", "a3", "a4"]
    identical = store.one("SELECT * FROM samples WHERE source_id='a1'")
    selected, _, trace, _ = retriever.select(identical, cfg, load_taxonomy(), Counter())
    assert "a1" not in [r["source_id"] for r in selected]
    assert trace["contrast_pair"]["anchor_sample_id"] == selected[0]["sample_id"]


def test_contrast_budget_never_discards_anchor_to_force_a_pair(contrast_corpus):
    store, corpus = contrast_corpus
    retriever = Retriever(store, corpus)
    cfg = config()
    _, _, _, base = retriever.select(
        query("late train again please"),
        {**cfg, "method": "zero-shot", "k": 0},
        load_taxonomy(),
        Counter(),
    )
    cfg["model"]["context_tokens"] = Counter().count(base) + 110 + 100 + 256
    selected, _, trace, messages = retriever.select(
        query("late train again please"), cfg, load_taxonomy(), Counter()
    )
    assert selected[0]["source_id"] == "a1"
    assert Counter().count(messages) <= cfg["model"]["context_tokens"] - 356
    assert len(selected) == 1
    assert trace["contrast_pair"] is None


@pytest.mark.parametrize(
    "override",
    [
        {"prompt_version": "typo"},
        {"example_policy": "typo"},
        {"example_policy": "contrastive-v1", "method": "random", "k": 4},
        {"example_policy": "contrastive-v1", "method": "lexical", "k": 1},
    ],
)
def test_invalid_prompt_or_policy_cannot_silently_fall_back(override):
    with pytest.raises(ValueError):
        validate_config({"method": "zero-shot", **override})


@pytest.mark.parametrize("version", ["native-labels-v2", "native-labels-v3"])
def test_actual_versioned_request_has_complete_frozen_prompt_snapshot(configured, version):
    store, cfg = configured
    cfg["prompt_version"] = version
    cfg["model"]["context_tokens"] = 30000
    run = prepare_run(store, cfg, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    row = store.one("SELECT * FROM experiment_runs WHERE run_id=?", (run,))
    snapshot = store.json(row["prompt_config_artifact_id"])
    attempt = store.one("SELECT * FROM v_call_audit WHERE run_id=? LIMIT 1", (run,))
    request = store.json(attempt["request_artifact_id"])
    assert (
        request["messages"][0]["content"].split("\nLabel definitions:")[0]
        == snapshot["instruction"]
    )
    assert snapshot["version"] == version
    assert snapshot["example_policy"]["version"] == "ranked"


def test_v3_changes_instruction_without_changing_query_or_training_evidence(contrast_corpus):
    store, corpus = contrast_corpus
    retriever = Retriever(store, corpus)
    cfg = config()
    old = retriever.select(query("late train again please"), cfg, load_taxonomy(), Counter())
    cfg["prompt_version"] = "native-labels-v3"
    cfg = validate_config(cfg)
    new = retriever.select(query("late train again please"), cfg, load_taxonomy(), Counter())
    assert [r["sample_id"] for r in old[0]] == [r["sample_id"] for r in new[0]]
    assert old[2]["contrast_pair"] == new[2]["contrast_pair"]
    assert old[3][1] == new[3][1]
    assert old[3][0]["content"] != new[3][0]["content"]
    assert old[3][0]["content"].split("\nLabel definitions:")[1] == new[3][0]["content"].split("\nLabel definitions:")[1]
