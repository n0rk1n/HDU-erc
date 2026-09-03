"""Pure long-term memory normalization, conflict, and ranking rules."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha256

from chatbot.memory.models import Memory, MemoryCandidate


def normalize_content(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def deterministic_memory_key(value: str) -> str:
    normalized = normalize_content(value)
    return "mem_" + sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_for_compare(value: str) -> str:
    normalized = normalize_content(value).lower()
    normalized = re.sub(r"[。.!！?？]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized)


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
PREFERENCE_CONFLICT_CATEGORIES = {"preference", "boundary"}


def _memory_topics(value: str) -> set[str]:
    normalized = _normalize_for_compare(value)
    topics = set()
    chinese_reply_context = ("回答", "回复", "答复", "解释")
    if (("简洁" in normalized or "简短" in normalized) and _has_any(normalized, chinese_reply_context)) or _has_pattern(normalized, r"\b(?:brief|concise)\s+(?:reply|replies|answer|answers|response|responses)\b", r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:briefly|concisely)\b"):
        topics.add("reply_length:concise")
    if (("详细" in normalized or "展开" in normalized) and _has_any(normalized, chinese_reply_context)) or _has_pattern(normalized, r"\b(?:detailed|expanded)\s+(?:reply|replies|answer|answers|response|responses)\b", r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:in\s+detail|with\s+detail)\b"):
        topics.add("reply_length:detailed")
    if _has_any(normalized, ("用中文回答", "用中文回复", "中文回答", "中文回复", "回答使用中文", "回复使用中文")) or _has_pattern(normalized, r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+in\s+chinese\b", r"\buse\s+chinese\s+for\s+(?:reply|replies|answer|answers|response|responses)\b", r"\bchinese\s+(?:reply|replies|answer|answers|response|responses)\b"):
        topics.add("reply_language:chinese")
    if _has_any(normalized, ("用英文回答", "用英文回复", "用英语回答", "用英语回复", "英文回答", "英文回复", "英语回答", "英语回复", "回答使用英文", "回复使用英文", "回答使用英语", "回复使用英语")) or _has_pattern(normalized, r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+in\s+english\b", r"\buse\s+english\s+for\s+(?:reply|replies|answer|answers|response|responses)\b", r"\benglish\s+(?:reply|replies|answer|answers|response|responses)\b"):
        topics.add("reply_language:english")
    if _has_any(normalized, ("语气正式", "正式语气", "风格正式", "正式风格", "回复正式", "回答正式", "答复正式", "解释正式")) or _has_pattern(normalized, r"\bformal\s+(?:tone|style)\b", r"\b(?:tone|style)\s+formal\b", r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+formally\b"):
        topics.add("reply_tone:formal")
    if _has_any(normalized, ("语气随意", "随意语气", "风格随意", "随意风格", "回复随意", "回答随意", "回复随意一点", "回答随意一点", "语气轻松", "轻松语气", "风格轻松", "轻松风格", "回复轻松", "回答轻松", "回复轻松一点", "回答轻松一点", "语气自然", "语气自然一点", "语气自然些", "自然语气", "风格自然", "自然风格", "回复自然一点", "回答自然一点")) or _has_pattern(normalized, r"\b(?:informal|casual)\s+(?:tone|style)\b", r"\b(?:tone|style)\s+(?:informal|casual)\b", r"\b(?:reply|replies|answer|answers|respond|response|responses)\s+(?:informally|casually)\b"):
        topics.add("reply_tone:casual")
    if "第三方" in normalized or "托管" in normalized or _has_word(normalized, "hosted", "third-party", "third party"):
        topics.add("memory_storage:hosted")
    return topics


def is_similar_memory(existing: Memory, candidate: MemoryCandidate, content: str) -> bool:
    if existing.category != candidate.category:
        return False
    if _normalize_for_compare(existing.content) == _normalize_for_compare(content):
        return True
    existing_topics = _memory_topics(existing.content)
    candidate_topics = _memory_topics(content)
    return bool(existing_topics) and existing_topics == candidate_topics


def _is_negative_request(value: str) -> bool:
    normalized = _normalize_for_compare(value)
    return _has_any(normalized, ("不要", "别", "禁止")) or _has_pattern(normalized, r"\bnot\s+to\b", r"\bdo\s+not\b", r"\bdon't\b", r"\bdont\b")


def _is_reply_language_topic(topic: str) -> bool:
    return topic.startswith("reply_language:")


def _negated_boundary_language_topics(category: str, content: str, topics: set[str]) -> set[str]:
    if category != "boundary" or not _is_negative_request(content):
        return set()
    return {topic for topic in topics if _is_reply_language_topic(topic)}


def conflicts(existing: Memory, candidate: MemoryCandidate, content: str) -> bool:
    existing_topics = _memory_topics(existing.content)
    candidate_topics = _memory_topics(content)
    existing_negated = _negated_boundary_language_topics(existing.category, existing.content, existing_topics)
    candidate_negated = _negated_boundary_language_topics(candidate.category, content, candidate_topics)
    if {existing.category, candidate.category} == {"boundary", "preference"}:
        if existing_negated & candidate_topics or candidate_negated & existing_topics:
            return True
    if existing.category in PREFERENCE_CONFLICT_CATEGORIES and candidate.category in PREFERENCE_CONFLICT_CATEGORIES:
        for topic in existing_topics:
            opposite = CONFLICTING_TOPICS.get(topic)
            if opposite in candidate_topics:
                if _is_reply_language_topic(topic) and (existing.category == "boundary" or candidate.category == "boundary") and (_is_negative_request(existing.content) or _is_negative_request(content)):
                    continue
                return True
    return "memory_storage:hosted" in existing_topics and "memory_storage:hosted" in candidate_topics and {existing.category, candidate.category} == {"boundary", "preference"}


def can_supersede(existing: Memory, candidate: MemoryCandidate) -> bool:
    return existing.category != "boundary" or candidate.category == "boundary"


def _tokens(value: str) -> set[str]:
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]+", value.lower())
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    tokens = set(ascii_tokens)
    for token in cjk_tokens:
        tokens.add(token)
        tokens.update(token[index:index + 2] for index in range(len(token) - 1))
    return {token for token in tokens if token}


CATEGORY_WEIGHTS = {"boundary": 5.0, "preference": 3.0, "goal": 1.5, "profile": 1.0}


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
    return score + sum(0.25 for token in overlap if len(token) >= 2 and token in normalized_content)


def _recency_score(value: str) -> float:
    try:
        updated_at = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return 0.0
    if updated_at.tzinfo is None or updated_at.utcoffset() is None:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds()) / 86400
    return max(0.0, 1.0 - min(age_days, 30.0) / 30.0)


def ranking_score(memory: Memory, query: str, query_tokens: set[str]) -> float:
    lexical = _lexical_score(memory.content, query, query_tokens)
    if lexical <= 0:
        return 0.0
    return lexical + CATEGORY_WEIGHTS.get(memory.category, 0.5) + max(0.0, min(memory.confidence, 1.0)) + _recency_score(memory.updated_at) + min(memory.use_count, 10) * 0.05
