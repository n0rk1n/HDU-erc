import json
import os
import subprocess
import sys
from test_storage import source_fixture


def test_cli_import_set_and_dry_run_without_credentials(tmp_path):
    root = tmp_path / "research"
    source = source_fixture(tmp_path / "source")
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["EXPERIMENT_DATA_DIR"] = str(root)

    def cli(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "emotion_lab", *args],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    assert cli("db", "init")["initialized"]
    ds = cli("dataset", "import-goemotions", "--source", str(source))[
        "dataset_version_id"
    ]
    sets = {}
    for purpose, split in [("corpus", "train"), ("dev_pilot", "dev")]:
        cfg = tmp_path / (purpose + ".json")
        cfg.write_text(
            json.dumps(
                {
                    "dataset_version_id": ds,
                    "name": purpose,
                    "purpose": purpose,
                    "split": split,
                }
            )
        )
        sets[purpose] = cli("dataset", "make-set", "--config", str(cfg))[
            "sample_set_id"
        ]
    run_cfg = tmp_path / "run.json"
    run_cfg.write_text(
        json.dumps(
            {
                "method": "zero-shot",
                "corpus_set_id": sets["corpus"],
                "evaluation_set_id": sets["dev_pilot"],
            }
        )
    )
    assert cli("run", "--config", str(run_cfg), "--dry-run")["sample_count"] == 2
    assert cli("storage", "verify")["ok"]
    assert cli("db", "status")["call_attempts"] == 0
    cli("storage", "backup", "--output", str(tmp_path / "backup"))
    assert (tmp_path / "backup" / "backup_manifest.json").exists()
