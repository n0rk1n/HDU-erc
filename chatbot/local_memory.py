"""SQLite-backed local memory provider."""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chatbot.memory import (
    MEMORY_CATEGORIES,
    DisabledMemoryProvider,
    Memory,
    MemoryCandidate,
    MemoryProvider,
    MemoryRuntimeConfig,
)


class SQLiteLocalMemoryProvider:
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
                rows = connection.execute(
                    """
                    select id, content, category, source, confidence,
                           created_at, updated_at, last_used_at, use_count
                    from memories
                    where status = 'active'
                    """
                ).fetchall()
                scored = []
                for row in rows:
                    memory = _memory_from_row(row)
                    score = _score(memory.content, tokens)
                    if score > 0:
                        scored.append((score, memory.updated_at, memory.use_count, memory))
                scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
                memories = [item[3] for item in scored[:limit]]
                self._mark_used(connection, memories)
                return [
                    Memory(
                        id=memory.id,
                        content=memory.content,
                        category=memory.category,
                        source=memory.source,
                        confidence=memory.confidence,
                        created_at=memory.created_at,
                        updated_at=memory.updated_at,
                        last_used_at=_now_iso(),
                        use_count=memory.use_count + 1,
                    )
                    for memory in memories
                ]
        except sqlite3.Error as exc:
            print(f"Warning: memory search failed: {exc}")
            return []

    def remember(self, candidates: list[MemoryCandidate]) -> list[Memory]:
        stored: list[Memory] = []
        try:
            with self._connect() as connection:
                for candidate in candidates:
                    normalized = _normalize_content(candidate.content)
                    if not normalized or candidate.category not in MEMORY_CATEGORIES:
                        continue
                    existing = self._find_existing(connection, normalized)
                    if existing is None:
                        stored.append(self._insert(connection, candidate, normalized))
                    else:
                        stored.append(self._update(connection, existing, candidate, normalized))
                return stored
        except sqlite3.Error as exc:
            print(f"Warning: memory write failed: {exc}")
            return stored

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=0.2)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists memories (
                    id text primary key,
                    content text not null,
                    category text not null,
                    source text not null,
                    confidence real not null,
                    created_at text not null,
                    updated_at text not null,
                    last_used_at text,
                    use_count integer not null default 0,
                    status text not null default 'active',
                    supersedes_id text,
                    metadata_json text not null default '{}'
                )
                """
            )
            self._migrate_schema(connection)
            connection.execute(
                "create index if not exists idx_memories_updated_at on memories(updated_at)"
            )
            connection.execute(
                "create index if not exists idx_memories_status on memories(status)"
            )

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("pragma table_info(memories)").fetchall()
        }
        migrations = {
            "status": "alter table memories add column status text not null default 'active'",
            "supersedes_id": "alter table memories add column supersedes_id text",
            "metadata_json": (
                "alter table memories add column metadata_json text not null default '{}'"
            ),
        }
        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)

    def _find_existing(self, connection: sqlite3.Connection, normalized: str) -> Memory | None:
        rows = connection.execute(
            """
            select id, content, category, source, confidence,
                   created_at, updated_at, last_used_at, use_count
            from memories
            where status = 'active'
            """
        ).fetchall()
        for row in rows:
            memory = _memory_from_row(row)
            if _normalize_content(memory.content) == normalized:
                return memory
        return None

    def _insert(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
        content: str,
    ) -> Memory:
        now = _now_iso()
        memory = Memory(
            id=f"mem_{uuid4().hex}",
            content=content,
            category=candidate.category,
            source=candidate.source,
            confidence=candidate.confidence,
            created_at=now,
            updated_at=now,
            last_used_at=None,
            use_count=0,
        )
        connection.execute(
            """
            insert into memories (
                id, content, category, source, confidence,
                created_at, updated_at, last_used_at, use_count
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.category,
                memory.source,
                memory.confidence,
                memory.created_at,
                memory.updated_at,
                memory.last_used_at,
                memory.use_count,
            ),
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
            """
            update memories
            set content = ?, category = ?, source = ?, confidence = ?, updated_at = ?
            where id = ?
            """,
            (
                content,
                candidate.category,
                candidate.source,
                confidence,
                now,
                existing.id,
            ),
        )
        return Memory(
            id=existing.id,
            content=content,
            category=candidate.category,
            source=candidate.source,
            confidence=confidence,
            created_at=existing.created_at,
            updated_at=now,
            last_used_at=existing.last_used_at,
            use_count=existing.use_count,
        )

    def _mark_used(self, connection: sqlite3.Connection, memories: list[Memory]) -> None:
        now = _now_iso()
        for memory in memories:
            connection.execute(
                """
                update memories
                set last_used_at = ?, use_count = use_count + 1
                where id = ?
                """,
                (now, memory.id),
            )


def build_memory_provider(config: MemoryRuntimeConfig) -> MemoryProvider:
    if not config.enabled:
        return DisabledMemoryProvider()
    return SQLiteLocalMemoryProvider(config.db_path)


def _memory_from_row(row) -> Memory:
    return Memory(
        id=row[0],
        content=row[1],
        category=row[2],
        source=row[3],
        confidence=float(row[4]),
        created_at=row[5],
        updated_at=row[6],
        last_used_at=row[7],
        use_count=int(row[8]),
    )


def _normalize_content(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _tokens(value: str) -> set[str]:
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]+", value.lower())
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    tokens = set(ascii_tokens)
    for token in cjk_tokens:
        tokens.add(token)
        for index in range(len(token) - 1):
            tokens.add(token[index:index + 2])
    return {token for token in tokens if token}


def _score(content: str, query_tokens: set[str]) -> int:
    content_tokens = _tokens(content)
    return len(content_tokens & query_tokens)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
