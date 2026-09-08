"""The analysis must preserve literal dataset quotes just like the importer."""

import json

from emotion_lab.storage import digest
from scripts.audit_train_prompt_v3 import audit


def test_training_audit_preserves_literal_quotes(tmp_path):
    texts = ['"A quoted Reddit comment"'] + [f"plain comment {i}" for i in range(7)]
    (tmp_path / "train.tsv").write_text("".join(f"{text}\t0\tid{i}\n" for i, text in enumerate(texts)))
    (tmp_path / "emotions.txt").write_text("neutral\n")
    manifest = {"files": {name: {"sha256": digest((tmp_path / name).read_bytes())} for name in ("train.tsv", "emotions.txt")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = audit(tmp_path)
    observed = {row["source_id"]: row["text"] for row in result["label_strata"]["neutral"]}
    assert observed == {f"id{i}": text for i, text in enumerate(texts)}
