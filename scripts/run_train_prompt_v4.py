"""Run the authorized, fixed dev200 prompt-only comparison with resumable evidence."""

import argparse
import json
from pathlib import Path

from emotion_lab.config import credentials, load_environment, validate_config
from emotion_lab.evaluation import compare, evaluate, export_evaluation
from emotion_lab.preparation import prepare_run
from emotion_lab.runner import execute_run
from emotion_lab.storage import Store, digest, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True)
    args = parser.parse_args()
    load_environment(args.env_file)
    credentials()  # Verify presence without printing or snapshotting credentials.
    root = Path("data/research")
    state_path = root / "train_prompt_v4_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    configs = {
        arm: validate_config(json.loads(Path(f"config/experiments/dev200-prompt-v4-{arm}.json").read_text()))
        for arm in ("control", "candidate")
    }
    changed = {key for key in configs["control"] if configs["control"][key] != configs["candidate"][key]}
    assert changed == {"name", "prompt_version"}, changed
    assert all(c["evaluation_set"] == "goemotions-dev200" and c["k"] == 4 for c in configs.values())
    assert configs["control"]["prompt_version"] == "native-labels-v3"
    assert configs["candidate"]["prompt_version"] == "native-labels-v4"
    if state:
        assert state["config_hash"] == digest(configs), "cannot change a running protocol"
    else:
        state.update(config_hash=digest(configs), started_at=now())

    def save():
        pending = state_path.with_suffix(".pending")
        pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        pending.replace(state_path)

    with Store(root) as store:
        if "driver_artifact_id" not in state:
            audit_id = "a519b66c-e87e-4271-baa4-29cb06c8adea"
            store.read(audit_id)
            with store.transaction():
                aid = store.put({
                    "source": Path(__file__).read_text(), "configs": configs,
                    "development_basis_artifact_id": audit_id,
                    "authorization": "Continue user-authorized GLM and fixed dev200 comparison; prompt-only optimization requested. Two fresh arms, 200 items each, two attempts per item maximum.",
                    "scope": "development iteration on previously inspected dev200; official test unused",
                }, "train_prompt_v4_driver", inline=False)
                store.event("artifact_id", aid, "train_prompt_v4_authorized")
            state.update(driver_artifact_id=aid, development_basis_artifact_id=audit_id)
            save()
        for arm, cfg in configs.items():
            if arm + "_run_id" not in state:
                print(json.dumps({"phase": "prepare_" + arm, "at": now()}), flush=True)
                state[arm + "_run_id"] = prepare_run(store, cfg)
                save()
                with store.transaction():
                    store.event("run_id", state[arm + "_run_id"], "development_basis_linked", payload=state["development_basis_artifact_id"])
                print(json.dumps({"phase": "frozen_" + arm, "run_id": state[arm + "_run_id"], "at": now()}), flush=True)
        for arm in configs:
            run_id = state[arm + "_run_id"]
            status = store.one("SELECT status FROM experiment_runs WHERE run_id=?", (run_id,))["status"]
            if status in ("completed", "completed_with_errors"):
                continue

            def checkpoint(stage, attempt_id):
                if stage == "response_saved":
                    code = store.one("SELECT http_status FROM call_attempts WHERE call_attempt_id=?", (attempt_id,))["http_status"]
                    if code in (400, 401, 403, 404):
                        raise RuntimeError(f"paused after HTTP {code}; response retained")
                if stage == "parsed":
                    n = store.one("SELECT COUNT(*) n FROM v_call_audit WHERE run_id=? AND response_artifact_id IS NOT NULL", (run_id,))["n"]
                    if n <= 2 or n % 20 == 0:
                        print(json.dumps({"phase": arm, "responses_saved": n, "at": now()}), flush=True)

            execute_run(store, run_id, resume=status != "ready", checkpoint=checkpoint)
            final_status = store.one("SELECT status FROM experiment_runs WHERE run_id=?", (run_id,))["status"]
            assert final_status in ("completed", "completed_with_errors"), final_status
            print(json.dumps({"phase": arm, "status": final_status, "at": now()}), flush=True)
        # Score only after both frozen arms finish; retain any failed items in denominator.
        for arm in configs:
            if arm + "_evaluation_id" not in state:
                state[arm + "_evaluation_id"] = evaluate(store, state[arm + "_run_id"], {})
                save()
            output = root / "exports" / ("glm-dev200-prompt-v4-" + arm + "-20260908")
            if not output.exists():
                export_evaluation(store, state[arm + "_evaluation_id"], output)
        if "comparison_id" not in state:
            state["comparison_id"] = compare(store, state["control_evaluation_id"], state["candidate_evaluation_id"], seed=42, repeats=1000)
            save()
        state["completed_at"] = now()
        save()
        with store.transaction():
            aid = store.put(state, "train_prompt_v4_execution_manifest")
            store.event("artifact_id", aid, "train_prompt_v4_executed")
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
