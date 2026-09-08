"""CLI for offline initialization and explicit, separately requested model runs."""

import argparse
import json
import os
import sqlite3
import sys

from .config import load_environment, project_path
from .datasets import import_goemotions, make_set
from .evaluation import evaluate, compare, export_evaluation
from .llm.transport import safe_facts
from .runner import prepare_run, execute_run, reparse, cancel_run, scheduler_lock
from .storage import Store


def parser():
    p = argparse.ArgumentParser(
        prog="emotion_lab", description="GoEmotions 多标签情绪识别实验"
    )
    p.add_argument("--data-dir")
    p.add_argument("--db-path")
    p.add_argument("--env-file")
    sub = p.add_subparsers(dest="command", required=True)
    db = sub.add_parser("db").add_subparsers(dest="action", required=True)
    db.add_parser("init")
    db.add_parser("status")
    dataset = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    imp = dataset.add_parser("import-goemotions")
    imp.add_argument("--source", default="data/benchmarks/goemotions")
    ms = dataset.add_parser("make-set")
    ms.add_argument("--config", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    for cmd in ("resume", "cancel", "reparse", "evaluate"):
        action = sub.add_parser(cmd)
        action.add_argument("--run-id", required=True)
        if cmd == "reparse":
            action.add_argument("--parser-version", required=True)
        if cmd == "evaluate":
            action.add_argument("--config")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--seed", type=int, default=42)
    comparison.add_argument("--repeats", type=int, default=1000)
    export = sub.add_parser("export")
    export.add_argument("--evaluation-id", required=True)
    export.add_argument("--output")
    storage = sub.add_parser("storage").add_subparsers(dest="action", required=True)
    storage.add_parser("verify")
    backup = storage.add_parser("backup")
    backup.add_argument("--output", required=True)
    return p


def load_json(path):
    return json.loads(project_path(path).read_text())


def main(argv=None):
    args = parser().parse_args(argv)
    load_environment(args.env_file)
    root = project_path(
        args.data_dir or os.getenv("EXPERIMENT_DATA_DIR", "data/research")
    )
    db_path = args.db_path or os.getenv("EXPERIMENT_DB_PATH")
    try:
        with Store(
            root,
            project_path(db_path) if db_path else None,
            initialize=args.command == "db" and args.action == "init",
        ) as store:
            if args.command == "db":
                if args.action == "init":
                    result = {"initialized": True, "database": str(store.path)}
                else:
                    result = {
                        t: store.one(f"SELECT COUNT(*) n FROM {t}")["n"]
                        for t in (
                            "samples",
                            "sample_labels",
                            "sample_sets",
                            "experiment_runs",
                            "run_items",
                            "call_attempts",
                            "predictions",
                            "evaluations",
                            "artifacts",
                        )
                    }
                    result["runs"] = store.rows("SELECT * FROM v_run_summary")
            elif args.command == "dataset":
                if args.action == "import-goemotions":
                    result = {
                        "dataset_version_id": import_goemotions(
                            store, project_path(args.source)
                        )
                    }
                else:
                    result = {"sample_set_id": make_set(store, load_json(args.config))}
            elif args.command == "run":
                result = prepare_run(
                    store, load_json(args.config), dry_run=args.dry_run
                )
                if not args.dry_run:
                    execute_run(store, result)
                    result = {
                        "run_id": result,
                        "status": store.one(
                            "SELECT status FROM experiment_runs WHERE run_id=?",
                            (result,),
                        )["status"],
                    }
            elif args.command == "resume":
                execute_run(store, args.run_id, resume=True)
                result = {
                    "run_id": args.run_id,
                    "status": store.one(
                        "SELECT status FROM experiment_runs WHERE run_id=?",
                        (args.run_id,),
                    )["status"],
                }
            elif args.command == "cancel":
                with scheduler_lock(store):
                    cancel_run(store, args.run_id)
                result = {"run_id": args.run_id, "status": "cancelled"}
            elif args.command == "reparse":
                result = {
                    "prediction_ids": reparse(store, args.run_id, args.parser_version)
                }
            elif args.command == "evaluate":
                result = {
                    "evaluation_id": evaluate(
                        store,
                        args.run_id,
                        load_json(args.config) if args.config else {},
                    )
                }
            elif args.command == "compare":
                result = {
                    "comparison_id": compare(
                        store, args.baseline, args.candidate, args.seed, args.repeats
                    )
                }
            elif args.command == "export":
                result = {
                    "output": str(
                        export_evaluation(
                            store,
                            args.evaluation_id,
                            project_path(args.output)
                            if args.output
                            else root / "exports" / args.evaluation_id,
                        )
                    )
                }
            elif args.command == "storage":
                if args.action == "verify":
                    result = store.verify()
                    if not store.read_only:
                        with store.transaction():
                            aid = store.put(result, "verification_report")
                            store.event("artifact_id", aid, "verified")
                else:
                    result = {"output": str(store.backup(project_path(args.output)))}
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 1 if isinstance(result, dict) and result.get("ok") is False else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        secret = os.getenv("EMOTION_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")
        print(
            json.dumps(
                {"error": safe_facts(str(exc), secret), "type": type(exc).__name__},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("运行已中断；已派发请求保留审计记录，请使用 resume。", file=sys.stderr)
        return 130
