"""Temporary SQLite-backed local memory provider for existing synchronous callers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chatbot.memory.models import (
    MEMORY_CATEGORIES,
    DisabledMemoryProvider,
    Memory,
    MemoryCandidate,
    MemoryProvider,
    MemoryRuntimeConfig,
)
from chatbot.memory.rules import (
    _tokens,
    can_supersede,
    conflicts,
    is_similar_memory,
    normalize_content,
    ranking_score,
)


class SQLiteLocalMemoryProvider:
    """Compatibility adapter retained until graph nodes use StoreMemoryRepository."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def search(self, query: str, *, limit: int) -> list[Memory]:
        tokens = _tokens(query)
        if not tokens or limit <= 0:
            return []
        try:
            with self._connect() as connection:
                memories = self._active_memories(connection)
                scored = [
                    (score, memory.updated_at, memory.use_count, memory)
                    for memory in memories
                    if (score := ranking_score(memory, query, tokens)) > 0
                ]
                scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
                selected = [item[3] for item in scored[:limit]]
                self._mark_used(connection, selected)
                now = _now_iso()
                return [
                    Memory(
                        id=memory.id,
                        content=memory.content,
                        category=memory.category,
                        source=memory.source,
                        confidence=memory.confidence,
                        created_at=memory.created_at,
                        updated_at=memory.updated_at,
                        last_used_at=now,
                        use_count=memory.use_count + 1,
                    )
                    for memory in selected
                ]
        except sqlite3.Error as exc:
            print(f"Warning: memory search failed: {exc}")
            return []

    def remember(self, candidates: list[MemoryCandidate]) -> list[Memory]:
        stored: list[Memory] = []
        try:
            with self._connect() as connection:
                for candidate in candidates:
                    content = normalize_content(candidate.content)
                    if not content or candidate.category not in MEMORY_CATEGORIES:
                        continue
                    active = self._active_memories(connection)
                    existing = next(
                        (memory for memory in active if is_similar_memory(memory, candidate, content)),
                        None,
                    )
                    if existing is not None:
                        stored.append(self._update(connection, existing, candidate, content))
                        continue
                    conflicting = [
                        memory for memory in active if conflicts(memory, candidate, content)
                    ]
                    if any(not can_supersede(memory, candidate) for memory in conflicting):
                        continue
                    for memory in conflicting:
                        connection.execute("update memories set status = 'superseded' where id = ?", (memory.id,))
                    stored.append(
                        self._insert(
                            connection,
                            candidate,
                            content,
                            supersedes_id=conflicting[0].id if conflicting else None,
                        )
                    )
        except sqlite3.Error as exc:
            print(f"Warning: memory write failed: {exc}")
        return stored

    def get_consolidation_state(self) -> dict[str, str | int | None]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "select last_turn_count, last_message_id from memory_consolidation_state where id = ?",
                    ("default",),
                ).fetchone()
        except sqlite3.Error as exc:
            print(f"Warning: memory consolidation state read failed: {exc}")
            return {"last_turn_count": 0, "last_message_id": None}
        if row is None:
            return {"last_turn_count": 0, "last_message_id": None}
        return {"last_turn_count": int(row[0]), "last_message_id": row[1]}

    def mark_consolidated(self, *, turn_count: int, last_message_id: str | None) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    insert into memory_consolidation_state (id, last_turn_count, last_message_id, updated_at)
                    values (?, ?, ?, ?)
                    on conflict(id) do update set
                        last_turn_count = excluded.last_turn_count,
                        last_message_id = excluded.last_message_id,
                        updated_at = excluded.updated_at
                    """,
                    ("default", turn_count, last_message_id, _now_iso()),
                )
        except sqlite3.Error as exc:
            print(f"Warning: memory consolidation state write failed: {exc}")

    def _active_memories(self, connection: sqlite3.Connection) -> list[Memory]:
        rows = connection.execute(
            """
            select id, content, category, source, confidence, created_at, updated_at, last_used_at, use_count
            from memories where status = 'active' order by created_at, rowid
            """
        ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def _insert(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
        content: str,
        *,
        supersedes_id: str | None,
    ) -> Memory:
        now = _now_iso()
        memory = Memory(
            id=f"mem_{uuid4().hex}", content=content, category=candidate.category,
            source=candidate.source, confidence=candidate.confidence, created_at=now,
            updated_at=now, last_used_at=None, use_count=0,
        )
        connection.execute(
            """
            insert into memories (id, content, category, source, confidence, created_at, updated_at,
                                  last_used_at, use_count, status, supersedes_id, metadata_json)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*memory.__dict__.values(), "active", supersedes_id, "{}"),
        )
        return memory

    def _update(
        self,
        connection: sqlite3.Connection,
        existing: Memory,
        candidate: MemoryCandidate,
        content: str,
    ) -> Memory:
        now = _now_iso()
        confidence = max(existing.confidence, candidate.confidence)
        connection.execute(
            """update memories set content = ?, category = ?, source = ?, confidence = ?, updated_at = ?
               where id = ?""",
            (content, candidate.category, candidate.source, confidence, now, existing.id),
        )
        return Memory(
            id=existing.id, content=content, category=candidate.category, source=candidate.source,
            confidence=confidence, created_at=existing.created_at, updated_at=now,
            last_used_at=existing.last_used_at, use_count=existing.use_count,
        )

    def _mark_used(self, connection: sqlite3.Connection, memories: list[Memory]) -> None:
        now = _now_iso()
        for memory in memories:
            connection.execute(
                "update memories set last_used_at = ?, use_count = use_count + 1 where id = ?",
                (now, memory.id),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=0.2)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists memories (
                    id text primary key, content text not null, category text not null, source text not null,
                    confidence real not null, created_at text not null, updated_at text not null,
                    last_used_at text, use_count integer not null default 0,
                    status text not null default 'active', supersedes_id text,
                    metadata_json text not null default '{}'
                )
                """
            )
            columns = {row[1] for row in connection.execute("pragma table_info(memories)").fetchall()}
            for column, statement in {
                "status": "alter table memories add column status text not null default 'active'",
                "supersedes_id": "alter table memories add column supersedes_id text",
                "metadata_json": "alter table memories add column metadata_json text not null default '{}'",
            }.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute("create index if not exists idx_memories_updated_at on memories(updated_at)")
            connection.execute("create index if not exists idx_memories_status on memories(status)")
            connection.execute(
                """
                create table if not exists memory_consolidation_state (
                    id text primary key, last_turn_count integer not null,
                    last_message_id text, updated_at text not null
                )
                """
            )


def build_memory_provider(config: MemoryRuntimeConfig) -> MemoryProvider:
    return SQLiteLocalMemoryProvider(config.db_path) if config.enabled else DisabledMemoryProvider()


def _memory_from_row(row) -> Memory:
    return Memory(
        id=row[0], content=row[1], category=row[2], source=row[3], confidence=float(row[4]),
        created_at=row[5], updated_at=row[6], last_used_at=row[7], use_count=int(row[8]),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
