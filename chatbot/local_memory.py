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
                    score = _ranking_score(memory, query, tokens)
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
                    existing = self._find_existing(connection, candidate, normalized)
                    if existing is None:
                        conflicts = self._find_conflicts(connection, candidate, normalized)
                        blocked = [
                            conflict
                            for conflict in conflicts
                            if not _can_supersede(conflict, candidate)
                        ]
                        if blocked:
                            continue
                        superseded = [
                            conflict
                            for conflict in conflicts
                            if _can_supersede(conflict, candidate)
                        ]
                        if superseded:
                            for conflict in superseded:
                                self._mark_superseded(connection, conflict)
                            stored.append(
                                self._insert(
                                    connection,
                                    candidate,
                                    normalized,
                                    supersedes_id=superseded[0].id,
                                )
                            )
                        elif conflicts:
                            continue
                        else:
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

    def _find_existing(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
        normalized: str,
    ) -> Memory | None:
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
            if _is_similar_memory(memory, candidate, normalized):
                return memory
        return None

    def _find_conflicts(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
        normalized: str,
    ) -> list[Memory]:
        rows = connection.execute(
            """
            select id, content, category, source, confidence,
                   created_at, updated_at, last_used_at, use_count
            from memories
            where status = 'active'
            order by created_at, rowid
            """
        ).fetchall()
        conflicts = []
        for row in rows:
            memory = _memory_from_row(row)
            if _conflicts(memory, candidate, normalized):
                conflicts.append(memory)
        return conflicts

    def _mark_superseded(self, connection: sqlite3.Connection, memory: Memory) -> None:
        connection.execute(
            "update memories set status = 'superseded' where id = ?",
            (memory.id,),
        )

    def _insert(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
        content: str,
        supersedes_id: str | None = None,
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
                created_at, updated_at, last_used_at, use_count,
                status, supersedes_id, metadata_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "active",
                supersedes_id,
                "{}",
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


def _normalize_for_compare(value: str) -> str:
    normalized = _normalize_content(value).lower()
    normalized = re.sub(r"[。.!！?？]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _has_word(value: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", value) for word in words)


def _has_pattern(value: str, *patterns: str) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def _has_any(value: str, words: tuple[str, ...]) -> bool:
    return any(word in value for word in words)


CONFLICTING_TOPICS = {
    "reply_length:concise": "reply_length:detailed",
    "reply_length:detailed": "reply_length:concise",
    "reply_language:chinese": "reply_language:english",
    "reply_language:english": "reply_language:chinese",
    "reply_tone:formal": "reply_tone:casual",
    "reply_tone:casual": "reply_tone:formal",
}


def _memory_topics(value: str) -> set[str]:
    normalized = _normalize_for_compare(value)
    topics = set()
    chinese_reply_context = ("回答", "回复", "答复", "解释")
    if (
        (
            ("简洁" in normalized or "简短" in normalized)
            and _has_any(normalized, chinese_reply_context)
        )
        or _has_pattern(
            normalized,
            r"\b(?:brief|concise)\s+(?:reply|replies|answer|answers|response|responses)\b",
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:briefly|concisely)\b",
        )
    ):
        topics.add("reply_length:concise")
    if (
        (
            ("详细" in normalized or "展开" in normalized)
            and _has_any(normalized, chinese_reply_context)
        )
        or _has_pattern(
            normalized,
            r"\b(?:detailed|expanded)\s+(?:reply|replies|answer|answers|response|responses)\b",
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:in\s+detail|with\s+detail)\b",
        )
    ):
        topics.add("reply_length:detailed")
    if (
        _has_any(
            normalized,
            (
                "用中文回答",
                "用中文回复",
                "中文回答",
                "中文回复",
                "回答使用中文",
                "回复使用中文",
            ),
        )
        or _has_pattern(
            normalized,
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+in\s+chinese\b",
            r"\buse\s+chinese\s+for\s+(?:reply|replies|answer|answers|response|responses)\b",
            r"\bchinese\s+(?:reply|replies|answer|answers|response|responses)\b",
        )
    ):
        topics.add("reply_language:chinese")
    if (
        _has_any(
            normalized,
            (
                "用英文回答",
                "用英文回复",
                "用英语回答",
                "用英语回复",
                "英文回答",
                "英文回复",
                "英语回答",
                "英语回复",
                "回答使用英文",
                "回复使用英文",
                "回答使用英语",
                "回复使用英语",
            ),
        )
        or _has_pattern(
            normalized,
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+in\s+english\b",
            r"\buse\s+english\s+for\s+(?:reply|replies|answer|answers|response|responses)\b",
            r"\benglish\s+(?:reply|replies|answer|answers|response|responses)\b",
        )
    ):
        topics.add("reply_language:english")
    if (
        _has_any(
            normalized,
            (
                "语气正式",
                "正式语气",
                "风格正式",
                "正式风格",
                "回复正式",
                "回答正式",
                "答复正式",
                "解释正式",
            ),
        )
        or _has_pattern(
            normalized,
            r"\bformal\s+(?:tone|style)\b",
            r"\b(?:tone|style)\s+formal\b",
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+formally\b",
        )
    ):
        topics.add("reply_tone:formal")
    if (
        _has_any(
            normalized,
            (
                "语气随意",
                "随意语气",
                "风格随意",
                "随意风格",
                "回复随意",
                "回答随意",
                "回复随意一点",
                "回答随意一点",
                "语气轻松",
                "轻松语气",
                "风格轻松",
                "轻松风格",
                "回复轻松",
                "回答轻松",
                "回复轻松一点",
                "回答轻松一点",
                "语气自然",
                "语气自然一点",
                "语气自然些",
                "自然语气",
                "风格自然",
                "自然风格",
                "回复自然一点",
                "回答自然一点",
            ),
        )
        or _has_pattern(
            normalized,
            r"\b(?:informal|casual)\s+(?:tone|style)\b",
            r"\b(?:tone|style)\s+(?:informal|casual)\b",
            r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:informally|casually)\b",
        )
    ):
        topics.add("reply_tone:casual")
    if (
        "第三方" in normalized
        or "托管" in normalized
        or _has_word(normalized, "hosted", "third-party", "third party")
    ):
        topics.add("memory_storage:hosted")
    return topics


def _is_similar_memory(existing: Memory, candidate: MemoryCandidate, content: str) -> bool:
    if existing.category != candidate.category:
        return False
    if _normalize_for_compare(existing.content) == _normalize_for_compare(content):
        return True
    existing_topics = _memory_topics(existing.content)
    candidate_topics = _memory_topics(content)
    return bool(existing_topics) and existing_topics == candidate_topics


def _is_negative_request(value: str) -> bool:
    normalized = _normalize_for_compare(value)
    return (
        _has_any(normalized, ("不要", "别", "禁止"))
        or _has_pattern(
            normalized,
            r"\bnot\s+to\b",
            r"\bdo\s+not\b",
            r"\bdon't\b",
            r"\bdont\b",
        )
    )


def _is_reply_language_topic(topic: str) -> bool:
    return topic.startswith("reply_language:")


def _negated_boundary_language_topics(
    category: str,
    content: str,
    topics: set[str],
) -> set[str]:
    if category != "boundary" or not _is_negative_request(content):
        return set()
    return {topic for topic in topics if _is_reply_language_topic(topic)}


def _conflicts(existing: Memory, candidate: MemoryCandidate, content: str) -> bool:
    existing_topics = _memory_topics(existing.content)
    candidate_topics = _memory_topics(content)
    existing_negated_boundary_languages = _negated_boundary_language_topics(
        existing.category,
        existing.content,
        existing_topics,
    )
    candidate_negated_boundary_languages = _negated_boundary_language_topics(
        candidate.category,
        content,
        candidate_topics,
    )
    if {existing.category, candidate.category} == {"boundary", "preference"}:
        if existing_negated_boundary_languages & candidate_topics:
            return True
        if candidate_negated_boundary_languages & existing_topics:
            return True
    for topic in existing_topics:
        opposite = CONFLICTING_TOPICS.get(topic)
        if opposite in candidate_topics:
            if (
                _is_reply_language_topic(topic)
                and (existing.category == "boundary" or candidate.category == "boundary")
                and (
                    _is_negative_request(existing.content)
                    or _is_negative_request(content)
                )
            ):
                continue
            return True
    if (
        "memory_storage:hosted" in existing_topics
        and "memory_storage:hosted" in candidate_topics
        and {existing.category, candidate.category} == {"boundary", "preference"}
    ):
        return True
    return False


def _can_supersede(existing: Memory, candidate: MemoryCandidate) -> bool:
    if existing.category == "boundary" and candidate.category != "boundary":
        return False
    return True


def _tokens(value: str) -> set[str]:
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]+", value.lower())
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    tokens = set(ascii_tokens)
    for token in cjk_tokens:
        tokens.add(token)
        for index in range(len(token) - 1):
            tokens.add(token[index:index + 2])
    return {token for token in tokens if token}


CATEGORY_WEIGHTS = {
    "boundary": 5.0,
    "preference": 3.0,
    "goal": 1.5,
    "profile": 1.0,
}


def _lexical_score(content: str, query: str, query_tokens: set[str]) -> float:
    content_tokens = _tokens(content)
    if not content_tokens:
        return 0.0
    overlap = content_tokens & query_tokens
    score = float(len(overlap))
    normalized_content = _normalize_for_compare(content)
    normalized_query = _normalize_for_compare(query)
    if normalized_query and normalized_query in normalized_content:
        score += 3.0
    for token in overlap:
        if len(token) >= 2 and token in normalized_content:
            score += 0.25
    return score


def _recency_score(value: str) -> float:
    try:
        updated_at = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return 0.0
    if updated_at.tzinfo is None or updated_at.utcoffset() is None:
        return 0.0
    age_seconds = max(0.0, (_now() - updated_at).total_seconds())
    age_days = age_seconds / 86400
    return max(0.0, 1.0 - min(age_days, 30.0) / 30.0)


def _ranking_score(memory: Memory, query: str, query_tokens: set[str]) -> float:
    lexical = _lexical_score(memory.content, query, query_tokens)
    if lexical <= 0:
        return 0.0
    category = CATEGORY_WEIGHTS.get(memory.category, 0.5)
    confidence = max(0.0, min(memory.confidence, 1.0))
    recency = _recency_score(memory.updated_at)
    usage = min(memory.use_count, 10) * 0.05
    return lexical + category + confidence + recency + usage


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")
