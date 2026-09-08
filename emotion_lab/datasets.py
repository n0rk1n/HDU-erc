"""Import immutable source bytes; materialize ordered, frozen sample sets."""

import csv
import io
import json
from pathlib import Path
import random
import unicodedata
import zipfile

from .storage import digest, now

IMPORTER_VERSION = "goemotions-tsv-v1"
NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def import_goemotions(store, source):
    source = Path(source).resolve()
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    required = {
        "train.tsv",
        "dev.tsv",
        "test.tsv",
        "emotions.txt",
        "LICENSE",
        "upstream_README.md",
    }
    if not required <= set(manifest["files"]):
        raise ValueError("source manifest is incomplete")
    contents = {}
    for filename, meta in manifest["files"].items():
        path = (source / filename).resolve()
        if not path.is_relative_to(source):
            raise ValueError("unsafe source path")
        data = path.read_bytes()
        if digest(data) != meta["sha256"] or len(data) != meta["bytes"]:
            raise ValueError(f"source hash mismatch: {filename}")
        contents[filename] = data
    labels = contents["emotions.txt"].decode().splitlines()
    if len(labels) != 28 or len(set(labels)) != 28 or labels[-1] != "neutral":
        raise ValueError("GoEmotions requires the native 28 labels")
    key = digest(
        {
            "files": manifest["files"],
            "revision": manifest["source_commit"],
            "importer": IMPORTER_VERSION,
            "normalization": NORMALIZATION_VERSION,
        }
    )
    existing = store.rows(
        "SELECT * FROM dataset_versions WHERE name=? AND version_key=?",
        ("GoEmotions", key),
    )
    if existing:
        if existing[0]["status"] != "ready":
            raise ValueError("previous import incomplete")
        return existing[0]["dataset_version_id"]
    records, counts, seen = [], {}, set()
    for split in ("train", "dev", "test"):
        rows = list(
            csv.reader(
                io.StringIO(contents[split + ".tsv"].decode()),
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
        )
        counts[split] = len(rows)
        if len(rows) != manifest["expected_rows"][split]:
            raise ValueError(f"unexpected row count: {split}")
        for line, row in enumerate(rows, 1):
            if len(row) != 3:
                raise ValueError(f"invalid TSV row: {split}:{line}")
            text, encoded, source_id = row
            ids = [int(x) for x in encoded.split(",")]
            if (
                not text
                or not source_id
                or source_id in seen
                or not ids
                or len(set(ids)) != len(ids)
                or any(i not in range(28) for i in ids)
            ):
                raise ValueError(f"invalid source sample: {split}:{line}")
            seen.add(source_id)
            records.append(
                (
                    split,
                    line,
                    text,
                    source_id,
                    ids,
                    digest(("\t".join(row) + "\n").encode()),
                )
            )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in sorted(contents.items()):
            archive.writestr(filename, data)
        archive.writestr("manifest.json", manifest_bytes)
    with store.transaction():
        ds = store.insert(
            "dataset_versions",
            name="GoEmotions",
            version_key=key,
            source_repository=manifest["source_repository"],
            source_revision=manifest["source_commit"],
            source_manifest_artifact_id=store.put(
                manifest_bytes, "source_manifest", media_type="application/json"
            ),
            source_bundle_artifact_id=store.put(
                bundle.getvalue(),
                "source_bundle",
                inline=False,
                media_type="application/zip",
            ),
            label_manifest_sha256=digest(contents["emotions.txt"]),
            importer_version=IMPORTER_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            expected_counts_json=manifest["expected_rows"],
            actual_counts_json=counts,
            status="importing",
        )
        for i, label in enumerate(labels):
            store.insert(
                "label_definitions",
                dataset_version_id=ds,
                label_id=i,
                label_name=label,
                source_order=i,
            )
        for split, line, text, source_id, ids, record_hash in records:
            sample = store.insert(
                "samples",
                dataset_version_id=ds,
                source_id=source_id,
                split=split,
                source_line=line,
                language="en",
                raw_text=text,
                text_sha256=digest(text.encode()),
                normalized_text_sha256=digest(normalize(text).encode()),
                source_record_sha256=record_hash,
                metadata_json={"source_file": split + ".tsv"},
            )
            for position, label in enumerate(ids):
                store.insert(
                    "sample_labels",
                    sample_id=sample,
                    dataset_version_id=ds,
                    label_id=label,
                    annotation_source="official-tsv",
                    source_label_position=position,
                )
        store.transition(
            "dataset_versions", "dataset_version_id", ds, "ready", sealed_at=now()
        )
    return ds


def make_set(store, config):
    config = dict(config)
    ds = config.get("dataset_version_id")
    if ds is None:
        versions = store.rows(
            "SELECT dataset_version_id FROM dataset_versions WHERE status='ready'"
        )
        if len(versions) != 1:
            raise ValueError("specify dataset_version_id")
        ds = config["dataset_version_id"] = versions[0]["dataset_version_id"]
    split, purpose = config["split"], config["purpose"]
    if purpose == "corpus" and split != "train":
        raise ValueError("corpus must use train")
    if (purpose in ("dev_pilot", "dev_tuning") and split != "dev") or (
        purpose == "final_test" and split != "test"
    ):
        raise ValueError("sample purpose/split mismatch")
    if purpose not in {"corpus", "dev_pilot", "dev_tuning", "final_test", "diagnostic"}:
        raise ValueError("unsupported sample purpose")
    if (
        store.one(
            "SELECT status FROM dataset_versions WHERE dataset_version_id=?", (ds,)
        )["status"]
        != "ready"
    ):
        raise ValueError("dataset is not sealed")
    rows = store.rows(
        "SELECT sample_id,normalized_text_sha256 FROM samples WHERE dataset_version_id=? AND split=? ORDER BY source_line",
        (ds, split),
    )
    exclusions = config.get('exclude_set_ids', [])
    if not isinstance(exclusions, list) or not all(isinstance(sid, str) and sid for sid in exclusions):
        raise ValueError('exclude_set_ids must be a list of frozen set ids')
    excluded_hashes = set()
    for sid in exclusions:
        previous = store.one('SELECT * FROM sample_sets WHERE sample_set_id=?', (sid,))
        if previous['status'] != 'frozen' or previous['dataset_version_id'] != ds:
            raise ValueError('exclusion set must be frozen and from same dataset')
        excluded_hashes.update(r['normalized_text_sha256'] for r in store.rows(
            'SELECT s.normalized_text_sha256 FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=?', (sid,)))
    rows = [r for r in rows if r['normalized_text_sha256'] not in excluded_hashes]
    if config.get("deduplicate", False):
        seen = set()
        rows = [
            r
            for r in rows
            if r["normalized_text_sha256"] not in seen
            and not seen.add(r["normalized_text_sha256"])
        ]
    limit = config.get("limit")
    if limit is not None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > len(rows)
        ):
            raise ValueError("limit must be between 1 and available samples")
        rows = random.Random(config.get("seed", 42)).sample(rows, limit)
    if not rows:
        raise ValueError("empty sample set")
    members = [r["sample_id"] for r in rows]
    with store.transaction():
        sid = store.insert(
            "sample_sets",
            dataset_version_id=ds,
            name=config["name"],
            purpose=purpose,
            source_split=split,
            selection_config_artifact_id=store.put(config, "sample_selection"),
            members_sha256=digest(members),
            sample_count=len(members),
            status="building",
        )
        for ordinal, sample in enumerate(members, 1):
            store.insert(
                "sample_set_members",
                sample_set_id=sid,
                sample_id=sample,
                dataset_version_id=ds,
                ordinal=ordinal,
                selection_reason="fixed-seed-sample" if limit else "source-order",
            )
        store.db.execute(
            "UPDATE sample_sets SET status='frozen',frozen_at=? WHERE sample_set_id=?",
            (now(), sid),
        )
        aid = store.put(
            {
                "sample_set_id": sid,
                "count": len(members),
                "members_sha256": digest(members),
            },
            "sample_set_frozen",
        )
        store.event("dataset_version_id", ds, "sample_set_frozen", payload=aid)
    return sid
