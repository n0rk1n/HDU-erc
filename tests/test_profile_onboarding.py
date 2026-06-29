import json

from chatbot.profile_onboarding import (
    ALLOWED_PROFILE_FIELDS,
    MAX_PROFILE_VALUE_LENGTH,
    ONBOARDING_QUESTIONS,
    draft_profile,
    fallback_profile_draft,
    sanitize_profile,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return FakeResponse(self.content)


def test_onboarding_questions_are_fixed_and_skippable():
    assert ALLOWED_PROFILE_FIELDS == (
        "preferred_name",
        "life_stage",
        "companion_expectation",
        "response_style",
        "avoidance",
    )
    assert [question["key"] for question in ONBOARDING_QUESTIONS] == list(
        ALLOWED_PROFILE_FIELDS
    )
    assert ONBOARDING_QUESTIONS == [
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
    assert all(question["skippable"] is True for question in ONBOARDING_QUESTIONS)


def test_sanitize_profile_filters_empty_unknown_and_long_values():
    profile = {
        "preferred_name": " 小明 ",
        "life_stage": "",
        "companion_expectation": "   ",
        "response_style": "x" * 500,
        "avoidance": 123,
        "unknown": "保留我",
    }

    sanitized = sanitize_profile(profile)

    assert sanitized == {
        "preferred_name": "小明",
        "response_style": "x" * MAX_PROFILE_VALUE_LENGTH,
    }
    assert len(sanitized["response_style"]) == 200


def test_fallback_profile_draft_maps_answers_to_allowed_fields():
    answers = [
        {"key": "preferred_name", "answer": " 小明 "},
        {"key": "life_stage", "answer": "学生"},
        {"key": "unknown", "answer": "不要出现"},
        {"key": "response_style", "answer": ""},
        {"key": "avoidance", "answer": None},
    ]

    assert fallback_profile_draft(answers) == {
        "preferred_name": "小明",
        "life_stage": "学生",
    }


def test_draft_profile_uses_valid_llm_json():
    llm = FakeLlm(
        json.dumps(
            {
                "preferred_name": "小明",
                "response_style": " 温柔一点 ",
                "unknown": "不要出现",
            },
            ensure_ascii=False,
        )
    )

    draft = draft_profile(llm, [{"key": "preferred_name", "answer": "明明"}])

    assert draft == {"preferred_name": "小明", "response_style": "温柔一点"}
    assert len(llm.prompts) == 1
    assert "只输出 JSON" in llm.prompts[0]
    assert "preferred_name" in llm.prompts[0]


def test_draft_profile_falls_back_for_invalid_json():
    llm = FakeLlm("不是 JSON")
    answers = [{"key": "preferred_name", "answer": " 小明 "}]

    assert draft_profile(llm, answers) == {"preferred_name": "小明"}


def test_draft_profile_falls_back_when_llm_raises():
    llm = FakeLlm(error=RuntimeError("boom"))
    answers = [{"key": "life_stage", "answer": "备考"}]

    assert draft_profile(llm, answers) == {"life_stage": "备考"}
