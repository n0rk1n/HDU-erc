"""SQLite-backed runtime storage for local chat application state."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from chatbot.core.paths import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RUNTIME_DB_PATH = str(DATA_DIR / "records" / "runtime.sqlite3")


class RuntimeStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._init_schema()
        except sqlite3.Error as exc:
            print(f"Warning: could not initialize runtime database: {exc}")

    def load_json_records(self, namespace: str) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    select payload
                    from runtime_records
                    where namespace = ?
                    order by position
                    """,
                    (namespace,),
                ).fetchall()
        except sqlite3.Error as exc:
            print(f"Warning: could not load {namespace}: {exc}")
            return []

        records = []
        for (payload,) in rows:
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def append_json_record(self, namespace: str, record: dict[str, Any]) -> bool:
        try:
            payload = json.dumps(record, ensure_ascii=False)
            with self._connect() as connection:
                connection.execute(
                    """
                    insert into runtime_records(namespace, payload)
                    values (?, ?)
                    """,
                    (namespace, payload),
                )
            return True
        except (TypeError, sqlite3.Error) as exc:
            print(f"Warning: could not append {namespace}: {exc}")
            return False

    def replace_json_records(self, namespace: str, records: list[dict[str, Any]]) -> bool:
        try:
            payloads = [
                (namespace, json.dumps(record, ensure_ascii=False))
                for record in records
                if isinstance(record, dict)
            ]
            with self._connect() as connection:
                connection.execute(
                    "delete from runtime_records where namespace = ?",
                    (namespace,),
                )
                connection.executemany(
                    """
                    insert into runtime_records(namespace, payload)
                    values (?, ?)
                    """,
                    payloads,
                )
            return True
        except (TypeError, sqlite3.Error) as exc:
            print(f"Warning: could not replace {namespace}: {exc}")
            return False

    def load_profile(self) -> dict[str, str]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    select key, value
                    from profile_entries
                    order by key
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            print(f"Warning: could not load profile: {exc}")
            return {}
        return {key: value for key, value in rows if isinstance(value, str) and value.strip()}

    def replace_profile(self, profile: dict[str, str]) -> bool:
        rows = [
            (str(key), value)
            for key, value in profile.items()
            if isinstance(value, str) and value.strip()
        ]
        try:
            with self._connect() as connection:
                connection.execute("delete from profile_entries")
                connection.executemany(
                    """
                    insert into profile_entries(key, value)
                    values (?, ?)
                    """,
                    rows,
                )
            return True
        except sqlite3.Error as exc:
            print(f"Warning: could not replace profile: {exc}")
            return False

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=0.2)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists runtime_records (
                    position integer primary key autoincrement,
                    namespace text not null,
                    payload text not null
                )
                """
            )
            connection.execute(
                """
                create index if not exists runtime_records_namespace_position_idx
                on runtime_records(namespace, position)
                """
            )
            connection.execute(
                """
                create table if not exists profile_entries (
                    key text primary key,
                    value text not null
                )
                """
            )
