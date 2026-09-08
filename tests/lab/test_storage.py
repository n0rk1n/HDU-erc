import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from emotion_lab.storage import Store
from emotion_lab.datasets import import_goemotions, make_set


def source_fixture(path):
    path.mkdir()
    files = {
        "train.tsv": "I love it\t17,18\tt1\nangry!\t2,3\tt2\n",
        "dev.tsv": "neutral but happy\t17,27\td1\nsad\t25\td2\n",
        "test.tsv": "fine\t27\tx1\n",
        "LICENSE": "test fixture only",
        "upstream_README.md": "synthetic offline fixture",
    }
    official = Path(__file__).resolve().parents[2] / "config/emotion_labels.json"
    files["emotions.txt"] = "\n".join(json.loads(official.read_text())) + "\n"
    for name, value in files.items():
        (path / name).write_text(value)
    manifest = {
        "dataset": "GoEmotions",
        "source_repository": "fixture://offline",
        "source_commit": "fixture-v1",
        "expected_rows": {"train": 2, "dev": 2, "test": 1},
        "files": {
            n: {
                "sha256": hashlib.sha256(v.encode()).hexdigest(),
                "bytes": len(v.encode()),
                "url": "fixture://" + n,
            }
            for n, v in files.items()
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture
def lab(tmp_path):
    store = Store(tmp_path / "research", initialize=True)
    ds = import_goemotions(store, source_fixture(tmp_path / "source"))
    yield store, ds
    store.close()


def test_refuse_unknown_database_without_modifying_it(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    db = root / "experiments.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages(id INTEGER)")
    before = db.read_bytes()
    with pytest.raises(ValueError, match="unknown|未知"):
        Store(root, initialize=True)
    assert db.read_bytes() == before


def test_schema_artifacts_and_backup_detect_byte_corruption(lab, tmp_path):
    store, _ = lab
    assert (
        len(
            store.rows(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        == 23
    )
    aid = store.put(b"original bytes", "test", inline=False)
    assert store.read(aid) == b"original bytes"
    assert store.verify()["ok"]
    backup = store.backup(tmp_path / "backup")
    with Store(backup) as restored:
        assert restored.verify()["ok"]
        assert restored.read(aid) == b"original bytes"
    record = store.one("SELECT * FROM artifacts WHERE artifact_id=?", (aid,))
    (store.artifact_root / record["relative_path"]).write_bytes(b"corrupt! bytes")
    assert not store.verify()["ok"]
    with pytest.raises(ValueError, match="hash|哈希"):
        store.read(aid)


def test_import_idempotent_and_multilabel_immutable(lab, tmp_path):
    store, ds = lab
    assert import_goemotions(store, tmp_path / "source") == ds
    assert store.one("SELECT COUNT(*) n FROM samples")["n"] == 5
    assert store.one("SELECT COUNT(*) n FROM sample_labels")["n"] == 8
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("UPDATE samples SET raw_text='changed'")
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("DELETE FROM sample_labels")
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute(
            "INSERT INTO sample_labels(sample_id,dataset_version_id,label_id,annotation_source,source_label_position) SELECT sample_id,'wrong',0,'test',0 FROM samples LIMIT 1"
        )


def test_train_only_frozen_sets(lab):
    store, ds = lab
    with pytest.raises(ValueError, match="train"):
        make_set(
            store,
            {
                "dataset_version_id": ds,
                "name": "bad",
                "purpose": "corpus",
                "split": "dev",
            },
        )
    sid = make_set(
        store,
        {
            "dataset_version_id": ds,
            "name": "train",
            "purpose": "corpus",
            "split": "train",
        },
    )
    assert (
        store.one("SELECT sample_count FROM sample_sets WHERE sample_set_id=?", (sid,))[
            "sample_count"
        ]
        == 2
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("DELETE FROM sample_set_members WHERE sample_set_id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute(
            "UPDATE sample_sets SET source_split='test' WHERE sample_set_id=?", (sid,)
        )


def test_manifest_corruption_prevents_partial_import(tmp_path):
    source = source_fixture(tmp_path / "source")
    (source / "train.tsv").write_text("tampered")
    with Store(tmp_path / "data", initialize=True) as store:
        with pytest.raises(ValueError, match="hash|哈希"):
            import_goemotions(store, source)
        assert not store.rows("SELECT * FROM samples")


def test_backup_verification_detects_manifest_corruption(lab, tmp_path):
    store, _ = lab
    output = store.backup(tmp_path / "archive")
    manifest_path = output / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["experiments.sqlite3"] = "incorrect"
    manifest_path.write_text(json.dumps(manifest))
    with Store(output) as restored:
        assert not restored.verify()["ok"]


def test_portable_backup_never_depends_on_sqlite_shared_memory(lab, tmp_path):
    store, _ = lab
    output = store.backup(tmp_path / "portable")
    manifest = json.loads((output / "backup_manifest.json").read_text())
    assert not any(name.endswith(("-shm", "-wal")) for name in manifest["files"])
    with Store(output) as restored:
        assert restored.read_only
        assert restored.verify()["ok"]
        assert restored.one("SELECT COUNT(*) n FROM samples")["n"] == 5
