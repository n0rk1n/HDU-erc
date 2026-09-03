"""Conservative local extraction of durable memory candidates."""

import re

from chatbot.memory.models import MemoryCandidate


MAX_CANDIDATES_PER_TURN = 3


def extract_memory_candidates(
    user_message: str,
    assistant_reply: str = "",
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for sentence in _split_sentences(user_message):
        candidate = _candidate_from_sentence(sentence)
        if candidate is None:
            continue
        if not _contains_content(candidates, candidate.content):
            candidates.append(candidate)
        if len(candidates) >= MAX_CANDIDATES_PER_TURN:
            break
    return candidates


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[\n。！？!?]+", text)
    return [part.strip(" ，,;；") for part in parts if part.strip(" ，,;；")]


def _candidate_from_sentence(sentence: str) -> MemoryCandidate | None:
    chinese = _candidate_from_chinese(sentence)
    if chinese is not None:
        return chinese
    return _candidate_from_english(sentence)


def _candidate_from_chinese(sentence: str) -> MemoryCandidate | None:
    if sentence.startswith("不要"):
        content = sentence
        return MemoryCandidate(
            content=f"用户要求不要{content[2:]}。",
            category="boundary",
            confidence=0.9,
        )
    for prefix in ("我喜欢", "我希望", "以后请"):
        if sentence.startswith(prefix):
            return MemoryCandidate(
                content=f"用户{sentence[1:]}。",
                category="preference",
                confidence=0.85,
            )
    if sentence.startswith("我的") and "是" in sentence:
        return MemoryCandidate(
            content=f"用户{sentence}。",
            category="profile",
            confidence=0.8,
        )
    return None


def _candidate_from_english(sentence: str) -> MemoryCandidate | None:
    normalized = sentence.strip().rstrip(".").strip()
    lower = normalized.lower()
    if lower.startswith("do not "):
        rest = normalized[7:].strip()
        return MemoryCandidate(
            content=f"User requested not to {rest}.",
            category="boundary",
            confidence=0.9,
        )
    if lower.startswith("i prefer "):
        rest = normalized[9:].strip()
        return MemoryCandidate(
            content=f"User prefers {rest}.",
            category="preference",
            confidence=0.85,
        )
    if lower.startswith("i like "):
        rest = normalized[7:].strip()
        return MemoryCandidate(
            content=f"User likes {rest}.",
            category="preference",
            confidence=0.8,
        )
    if lower.startswith("please remember "):
        rest = normalized[16:].strip()
        return MemoryCandidate(
            content=f"User asked to remember {rest}.",
            category="profile",
            confidence=0.85,
        )
    if lower.startswith("my ") and " is " in lower:
        return MemoryCandidate(
            content=f"User stated that {normalized}.",
            category="profile",
            confidence=0.8,
        )
    return None


def _contains_content(candidates: list[MemoryCandidate], content: str) -> bool:
    normalized = _normalize_for_compare(content)
    return any(_normalize_for_compare(candidate.content) == normalized for candidate in candidates)


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
