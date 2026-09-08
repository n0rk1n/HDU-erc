"""Freeze the exact protocol, environment, and planned sample IDs before execution."""

import importlib.metadata
import os
import platform
import subprocess

from .config import PROJECT_ROOT, validate_config
from .llm.counter import TokenCounter
from .llm.parameters import thinking_extra_body
from .retrieval import Retriever, example_policy_snapshot
from .storage import digest, now
from .taxonomy import load_taxonomy, prompt_instruction


def resolve_set(store, value, purpose=None):
    rows = store.rows(
        "SELECT * FROM sample_sets WHERE sample_set_id=? OR name=?", (value, value)
    )
    if len(rows) != 1 or rows[0]["status"] != "frozen":
        raise ValueError("set must uniquely identify a frozen sample set")
    if purpose and rows[0]["purpose"] != purpose:
        raise ValueError("expected " + purpose + " set")
    return rows[0]


def prepare_run(store, config, dry_run=False, counter=None):
    config = validate_config(config)
    corpus = resolve_set(
        store, config.get("corpus_set_id") or config.get("corpus_set"), "corpus"
    )
    evaluation = resolve_set(
        store, config.get("evaluation_set_id") or config.get("evaluation_set")
    )
    if (
        corpus["dataset_version_id"] != evaluation["dataset_version_id"]
        or evaluation["purpose"] == "corpus"
    ):
        raise ValueError("incompatible experiment sample sets")
    config.update(
        corpus_set_id=corpus["sample_set_id"],
        evaluation_set_id=evaluation["sample_set_id"],
    )
    taxonomy = load_taxonomy()
    native = store.rows(
        "SELECT label_name FROM label_definitions WHERE dataset_version_id=? ORDER BY label_id",
        (corpus["dataset_version_id"],),
    )
    if [r["label_name"] for r in native] != list(taxonomy["labels"]):
        raise ValueError("taxonomy/native label order mismatch")
    counter = counter or TokenCounter(config["model"]["tokenizer_model"])
    config["counter"] = {"identity": counter.identity, "version": counter.version}
    model = config["model"]
    thinking_extra_body(
        model["name"],
        model["base_url"],
        model["thinking"],
        model.get("reasoning_effort"),
    )
    retriever = Retriever(store, corpus["sample_set_id"])
    samples = store.rows(
        "SELECT s.sample_id,s.raw_text,s.text_sha256,s.normalized_text_sha256,m.ordinal FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=? ORDER BY m.ordinal",
        (evaluation["sample_set_id"],),
    )
    estimates = []
    for sample in samples:
        *_, messages = retriever.select(sample, config, taxonomy, counter)
        estimates.append(counter.count(messages))
    preview = {
        "sample_count": len(samples),
        "method": config["method"],
        "input_tokens_min": min(estimates),
        "input_tokens_max": max(estimates),
        "max_planned_attempts": min(
            len(samples) * config["max_attempts"], config["max_total_attempts"]
        ),
        "config_sha256": digest(config),
        "model_calls": 0,
        "sample_set_id": evaluation["sample_set_id"],
        "counter": config["counter"],
    }
    if dry_run:
        return preview

    def git(*args):
        return (
            subprocess.check_output(["git", *args], cwd=PROJECT_ROOT).decode().strip()
        )

    commit = git("rev-parse", "HEAD")
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=PROJECT_ROOT
    )
    sources = application_sources()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names
        else None,
        "device": "cpu",
        "accelerator": None,
        "dependencies": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "sqlite_version": __import__("sqlite3").sqlite_version,
        "counter": config["counter"],
        "source_hashes": {k: digest(v.encode()) for k, v in sources.items()},
    }
    with store.transaction():
        config_artifact = store.put(config, "run_config")
        protocol = store.put(
            {
                "question": config.get(
                    "research_question",
                    "Compare frozen-model emotion classification methods",
                ),
                "config": config,
                "final_prediction_rule": "first-valid-by-attempt-number",
                "labels": "native-28",
                "failed_predictions": "empty-set",
                "zero_division": 0,
            },
            "experiment_protocol",
        )
        exp = store.insert(
            "experiments",
            name=config["name"],
            research_question=config.get(
                "research_question", "Fixed model retrieval comparison"
            ),
            dataset_version_id=corpus["dataset_version_id"],
            corpus_set_id=corpus["sample_set_id"],
            evaluation_set_id=evaluation["sample_set_id"],
            protocol_artifact_id=protocol,
            comparison_group=config.get("comparison_group", "default"),
            status="frozen",
            frozen_at=now(),
        )
        run = store.insert(
            "experiment_runs",
            experiment_id=exp,
            method_name=config["method"],
            method_config_artifact_id=config_artifact,
            model_config_artifact_id=store.put(model, "model_config"),
            prompt_config_artifact_id=store.put(
                {
                    "version": config["prompt_version"],
                    "instruction": prompt_instruction(
                        config["prompt_version"], config["require_evidence"]
                    ),
                    "require_evidence": config["require_evidence"],
                    "example_policy": example_policy_snapshot(config["example_policy"]),
                    "label_definitions": taxonomy["labels"],
                    "example_format": {
                        "text": "training raw_text",
                        "labels": "complete native label set",
                    },
                },
                "prompt_config",
            ),
            taxonomy_config_artifact_id=store.put(taxonomy, "taxonomy"),
            environment_artifact_id=store.put(environment, "environment"),
            price_snapshot_artifact_id=store.put(config["pricing"], "pricing")
            if config.get("pricing")
            else None,
            code_commit=commit,
            index_id=config.get('embedding', {}).get('index_id'),
            working_diff_artifact_id=store.put(
                {"git_diff": diff.decode(), "application_sources": sources},
                "working_sources",
                inline=False,
            ),
            seed=config["seed"],
            repetition_no=config["repetition_no"],
            config_sha256=digest(config),
            status="ready",
        )
        for sample in samples:
            store.insert(
                "run_items",
                run_id=run,
                sample_id=sample["sample_id"],
                ordinal=sample["ordinal"],
                input_sha256=sample["text_sha256"],
                status="pending",
            )
        store.event("run_id", run, "prepared", None, "ready")
    return run


def application_sources():
    return {
        str(p.relative_to(PROJECT_ROOT)): p.read_text()
        for p in (PROJECT_ROOT / "emotion_lab").rglob("*.py")
        if "vendor" not in p.parts
    }
