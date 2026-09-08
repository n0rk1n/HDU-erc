"""Recompute application fingerprints rather than trusting stored summaries."""

import sqlite3


def check_invariants(store):
    from .database import SCHEMA, digest
    from emotion_lab.datasets import normalize

    errors = []
    reference = sqlite3.connect(":memory:")
    try:
        reference.executescript(SCHEMA.read_text())
        expected = {
            r[0]: r[1]
            for r in reference.execute(
                "SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            )
        }
        actual = {
            r["name"]: r["sql"]
            for r in store.rows(
                "SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            )
        }
        if expected != actual:
            errors.append({"schema": "objects differ from versioned migration"})
    finally:
        reference.close()
    for row in store.rows("SELECT * FROM sample_sets WHERE status='frozen'"):
        members = [
            m["sample_id"]
            for m in store.rows(
                "SELECT sample_id FROM sample_set_members WHERE sample_set_id=? ORDER BY ordinal",
                (row["sample_set_id"],),
            )
        ]
        if (
            digest(members) != row["members_sha256"]
            or len(members) != row["sample_count"]
        ):
            errors.append(
                {
                    "sample_set_id": row["sample_set_id"],
                    "error": "member count/hash mismatch",
                }
            )
    for row in store.rows(
        "SELECT sample_id,raw_text,text_sha256,normalized_text_sha256 FROM samples"
    ):
        if (
            digest(row["raw_text"].encode()) != row["text_sha256"]
            or digest(normalize(row["raw_text"]).encode())
            != row["normalized_text_sha256"]
        ):
            errors.append(
                {"sample_id": row["sample_id"], "error": "text hash mismatch"}
            )
    for row in store.rows("SELECT * FROM experiment_runs"):
        try:
            config = store.json(row["method_config_artifact_id"])
            if digest(config) != row["config_sha256"]:
                errors.append(
                    {
                        "run_id": row["run_id"],
                        "error": "configuration fingerprint mismatch",
                    }
                )
        except (ValueError, OSError):
            pass  # Artifact validator reports the precise corruption.
    for row in store.rows("SELECT * FROM evaluations WHERE status='completed'"):
        try:
            snapshot = store.json(row["prediction_snapshot_artifact_id"])
            actual = store.rows(
                "SELECT run_item_id,prediction_id FROM evaluation_items WHERE evaluation_id=?",
                (row["evaluation_id"],),
            )
            if {r["run_item_id"]: r["prediction_id"] for r in actual} != {
                r["run_item_id"]: r["prediction_id"] for r in snapshot
            } or len(actual) != row["expected_items"]:
                errors.append(
                    {
                        "evaluation_id": row["evaluation_id"],
                        "error": "prediction snapshot mismatch",
                    }
                )
        except (ValueError, OSError):
            pass
    return errors
