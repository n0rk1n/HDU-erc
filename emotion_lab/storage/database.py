"""Versioned local database; every mutation belongs to an explicit transaction."""

from __future__ import annotations

from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from .artifacts import ArtifactStore

APPLICATION_ID = 0x45584C42
SCHEMA = Path(__file__).with_name("001_initial.sql")


def uid():
    return str(uuid.uuid4())


def now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else json_bytes(value)
    ).hexdigest()


class Store(ArtifactStore):
    def __init__(self, root, db_path=None, *, initialize=False):
        self.root = Path(root).resolve()
        self.path = (
            Path(db_path).resolve() if db_path else self.root / "experiments.sqlite3"
        )
        self.artifact_root = self.root / "artifacts"
        exists = self.path.exists()
        if exists:
            with closing(
                sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
            ) as probe:
                if (
                    probe.execute("PRAGMA application_id").fetchone()[0]
                    != APPLICATION_ID
                ):
                    raise ValueError(
                        "unknown database: refusing to overwrite 未知数据库"
                    )
                migrations = probe.execute(
                    "SELECT version,migration_sha256 FROM schema_migrations"
                ).fetchall()
                if migrations != [(1, digest(SCHEMA.read_bytes()))]:
                    raise ValueError("unknown or modified database migration")
        elif not initialize:
            raise ValueError("database missing; run db init first")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.read_only = (self.root / "backup_manifest.json").is_file()
        self.db = sqlite3.connect(
            self.path.as_uri() + "?mode=ro" if self.read_only else self.path,
            uri=self.read_only,
            isolation_level=None,
            timeout=5,
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._tables = {}
        if not exists:
            try:
                self.db.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + SCHEMA.read_text()
                    + f"\nPRAGMA application_id={APPLICATION_ID};"
                )
                self.db.execute(
                    "INSERT INTO schema_migrations VALUES(?,?,?)",
                    (1, digest(SCHEMA.read_bytes()), now()),
                )
                self.db.commit()
                with self.transaction():
                    a = self.put(
                        {"schema": 1, "application_id": APPLICATION_ID},
                        "initialization",
                    )
                    self.event("artifact_id", a, "initialized")
            except BaseException:
                self.db.rollback()
                self.db.close()
                raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self):
        if self.db.in_transaction:
            raise RuntimeError("nested transactions are not supported")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def rows(self, sql, args=()):
        return [dict(row) for row in self.db.execute(sql, args)]

    def one(self, sql, args=()):
        row = self.db.execute(sql, args).fetchone()
        if row is None:
            raise ValueError("record not found")
        return dict(row)

    def insert(self, table, **values):
        if table not in self._tables:
            # Table names come only from application code, checked before interpolation.
            if not table.isidentifier():
                raise ValueError("invalid table")
            columns = self.rows(f"PRAGMA table_info({table})")
            if not columns:
                raise ValueError("unknown table")
            self._tables[table] = columns
        columns = self._tables[table]
        keys = {c["name"] for c in columns}
        if set(values) - keys:
            raise ValueError(f"unknown columns for {table}: {set(values) - keys}")
        pk = [c for c in columns if c["pk"]]
        key = pk[0]["name"] if len(pk) == 1 else None
        if key and pk[0]["type"] == "TEXT" and key not in values:
            values[key] = uid()
        encoded = {
            k: json_bytes(v).decode()
            if k.endswith("_json") and not isinstance(v, str)
            else v
            for k, v in values.items()
        }
        self.db.execute(
            f"INSERT INTO {table}({','.join(encoded)}) VALUES({','.join('?' for _ in encoded)})",
            tuple(encoded.values()),
        )
        return values.get(key) if key else None

    def event(self, target_key, target, event_type, old=None, new=None, payload=None):
        if not self.db.in_transaction:
            raise RuntimeError("events and state changes require a transaction")
        return self.insert(
            "audit_events",
            **{target_key: target},
            event_type=event_type,
            from_status=old,
            to_status=new,
            actor="emotion_lab",
            payload_artifact_id=payload,
            occurred_at=now(),
        )

    def transition(self, table, key, identity, status, **extra):
        if not self.db.in_transaction:
            raise RuntimeError("state transition requires a transaction")
        old = self.one(f"SELECT status FROM {table} WHERE {key}=?", (identity,))[
            "status"
        ]
        fields = {"status": status, **extra}
        self.db.execute(
            f"UPDATE {table} SET {','.join(k + '=?' for k in fields)} WHERE {key}=?",
            (*fields.values(), identity),
        )
        self.event(key, identity, "state_changed", old, status)
