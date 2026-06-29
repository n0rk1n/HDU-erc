import json
from typing import Any


ALLOWED_PROFILE_FIELDS = (
    "preferred_name",
    "life_stage",
    "companion_expectation",
    "response_style",
    "avoidance",
)

MAX_PROFILE_VALUE_LENGTH = 200

ONBOARDING_QUESTIONS = [
    {"key": "preferred_name", "question": "希望我怎么称呼你？", "skippable": True},
    {
        "key": "life_stage",
        "question": "你现在大概是什么身份或阶段？比如学生、工作、备考、休息调整中。",
        "skippable": True,
    },
    {
        "key": "companion_expectation",
        "question": "你希望这个聊天机器人主要怎么陪伴你？",
        "skippable": True,
    },
    {
        "key": "response_style",
        "question": "你更喜欢怎样的回应风格？比如简短、温柔、直接、慢慢分析。",
        "skippable": True,
    },
    {
        "key": "avoidance",
        "question": "有哪些话题、表达方式或建议类型是你希望我避免的？",
        "skippable": True,
    },
]


def sanitize_profile(profile: dict[str, Any]) -> dict[str, str]:
    sanitized = {}
    for field in ALLOWED_PROFILE_FIELDS:
        value = profile.get(field)
        if not isinstance(value, str):
            continue

        stripped = value.strip()
        if not stripped:
            continue

        sanitized[field] = stripped[:MAX_PROFILE_VALUE_LENGTH]

    return sanitized


def fallback_profile_draft(answers: list[dict[str, Any]]) -> dict[str, str]:
    profile = {}
    for item in answers:
        key = item.get("key")
        answer = item.get("answer")
        if key in ALLOWED_PROFILE_FIELDS and isinstance(answer, str) and answer.strip():
            profile[key] = answer

    return sanitize_profile(profile)


def draft_profile(llm: Any, answers: list[dict[str, Any]]) -> dict[str, str]:
    fallback = fallback_profile_draft(answers)

    try:
        response = llm.invoke(_draft_prompt(answers))
        content = response.content if hasattr(response, "content") else response
        parsed = json.loads(content)
    except Exception:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    sanitized = sanitize_profile(parsed)
    return sanitized or fallback


def _draft_prompt(answers: list[dict[str, Any]]) -> str:
    allowed_fields = ", ".join(ALLOWED_PROFILE_FIELDS)
    answers_json = json.dumps(answers, ensure_ascii=False)
    return (
        "请根据用户的入门问答草拟一份聊天机器人用户资料。\n"
        "只输出 JSON，不要输出解释、Markdown 或代码块。\n"
        f"只允许使用这些字段：{allowed_fields}。\n"
        "不要编造用户没有明确表达的信息；跳过或不确定的字段请留空字符串或省略。\n"
        f"问答如下：{answers_json}"
    )
