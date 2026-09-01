import chatbot.emotion.analysis as emotion
import pytest
from chatbot.emotion import (
    analyze_emotion_async,
    build_emotion_prompt,
    parse_emotion_output,
)


def test_build_emotion_prompt_uses_recent_history_and_current_input():
    records = [
        {"role": "human", "content": "first question"},
        {"role": "ai", "content": "first answer"},
        {"role": "human", "content": "second question"},
        {"role": "ai", "content": "second answer"},
    ]

    prompt = build_emotion_prompt(
        records,
        "current question",
        previous_emotion="anxious",
        max_turns=1,
    )

    assert "Emotion labels:" in prompt
    assert "Response Format: Return exactly one JSON object" in prompt
    assert '"primary_emotion": "anxious"' in prompt
    assert '"reply_strategy": "brief guidance for the next chatbot reply"' in prompt
    assert "More likely emotion labels: anxious" in prompt
    assert "Dynamic EICL examples:" in prompt
    assert "True emotion label: anxious" in prompt
    assert "recent-emotion-prior" in prompt
    assert "Dialogue context:" in prompt
    assert "first question" not in prompt
    assert "second question</s>second answer</s>current question" in prompt


def test_build_emotion_prompt_accepts_multiple_likely_emotions():
    prompt = build_emotion_prompt(
        [],
        "I feel let down but still nervous about tomorrow.",
        previous_emotion="anxious",
        likely_emotions=["disappointed", "sad", "unknown", "anxious"],
    )

    assert "More likely emotion labels: anxious, disappointed, sad" in prompt
    assert "True emotion label: anxious" in prompt
    assert "True emotion label: disappointed" in prompt


def test_build_emotion_prompt_uses_previous_and_likely_emotions_for_dynamic_examples():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="angry",
        likely_emotions=["grateful"],
    )

    assert "More likely emotion labels: angry, grateful" in prompt
    assert "True emotion label: angry" in prompt
    assert "True emotion label: grateful" in prompt


def test_build_emotion_prompt_can_disable_dynamic_and_static_examples():
    prompt = build_emotion_prompt(
        [],
        "I feel nervous about tomorrow.",
        previous_emotion="anxious",
        example_mode="none",
    )

    assert "Dynamic EICL examples:" not in prompt
    assert "Labeled examples:" not in prompt
    assert "More likely emotion labels: anxious" in prompt


def test_build_emotion_prompt_can_disable_emotion_history():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="angry",
        likely_emotions=["grateful"],
        include_emotion_history=False,
    )

    assert "More likely emotion labels:" not in prompt
    assert "recent-emotion-prior" not in prompt


def test_build_emotion_prompt_supports_static_examples_mode():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="grateful",
        example_mode="static",
    )

    assert "Dynamic EICL examples:" not in prompt
    assert "Labeled examples:" in prompt
    assert "True emotion label: grateful" in prompt


def test_build_emotion_prompt_passes_prompt_variant_through():
    full_prompt = build_emotion_prompt([], "thank you")
    treatment_prompt = build_emotion_prompt(
        [], "thank you", prompt_variant="prompt_coarse_to_fine"
    )

    assert treatment_prompt != full_prompt


def test_parse_emotion_output_accepts_known_label():
    assert parse_emotion_output("Emotion: anxious") == "anxious"


def test_parse_emotion_output_accepts_extra_text():
    output = "Emotion: joyful\nThe user sounds upbeat."

    assert parse_emotion_output(output) == "joyful"


def test_parse_emotion_output_rejects_unknown_label():
    assert parse_emotion_output("Emotion: confused") is None


def test_parse_emotion_output_rejects_missing_format():
    assert parse_emotion_output("The emotion is anxious.") is None


@pytest.mark.asyncio
async def test_analyze_emotion_async_returns_structured_state_without_persistence():
    class AsyncFakeLlm:
        async def ainvoke(self, prompt):
            return type("Response", (), {"content": (
                '{"primary_emotion":"anxious","confidence":0.8,'
                '"secondary_emotions":["apprehensive"],'
                '"evidence":"The user is worried.",'
                '"reply_strategy":"Be calm.",'
                '"trajectory_note":"","safety_level":"normal"}'
            )})()

    result = await analyze_emotion_async(
        AsyncFakeLlm(),
        [],
        "我有点担心明天。",
        turn_count=1,
        emotion_interval=5,
    )

    assert result.success is True
    assert result.state is not None
    assert result.state.primary_emotion == "anxious"


@pytest.mark.asyncio
async def test_analyze_emotion_async_preserves_legacy_label_fallback():
    class AsyncFakeLlm:
        async def ainvoke(self, prompt):
            return type(
                "Response",
                (),
                {"content": "Model note: positive tone. Emotion: joyful"},
            )()

    result = await analyze_emotion_async(
        AsyncFakeLlm(),
        [],
        "That went well.",
        turn_count=1,
        emotion_interval=5,
    )

    assert result.success is True
    assert result.emotion == "joyful"
    assert result.state == emotion.EmotionState(primary_emotion="joyful")


def test_emotion_labels_are_shared_from_label_module():
    from chatbot.emotion.labels import EMOTION_LABELS as SHARED_LABELS
    from chatbot.emotion.labels import EMOTION_LABEL_SET as SHARED_LABEL_SET

    assert emotion.EMOTION_LABELS is SHARED_LABELS
    assert emotion.EMOTION_LABEL_SET is SHARED_LABEL_SET
    assert "anxious" in SHARED_LABEL_SET
