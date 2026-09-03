from string import Formatter

import pytest

from chatbot.emotion.prompt import build_emotion_analysis_prompt
from chatbot.emotion.prompt_variants import (
    PROMPT_VARIANT_NAMES,
    resolve_emotion_prompt_template,
)


def test_full_prompt_variant_preserves_existing_default_prompt():
    kwargs = {
        "emotion_labels": ["anxious", "lonely"],
        "emotion_label_set": {"anxious", "lonely"},
        "dialogue_context": "I feel alone tonight",
        "current_input": "I feel alone tonight",
    }
    assert build_emotion_analysis_prompt(**kwargs) == build_emotion_analysis_prompt(
        **kwargs,
        prompt_variant="full",
    )


def test_prompt_variants_render_distinct_model_inputs():
    prompts = {
        build_emotion_analysis_prompt(
            emotion_labels=["anxious", "lonely"],
            emotion_label_set={"anxious", "lonely"},
            dialogue_context="I feel alone tonight",
            current_input="I feel alone tonight",
            prompt_variant=variant,
        )
        for variant in PROMPT_VARIANT_NAMES
    }

    assert len(prompts) == len(PROMPT_VARIANT_NAMES)


def test_no_label_guidance_variant_omits_guidance_placeholder_only():
    template = resolve_emotion_prompt_template("prompt_no_label_guidance")
    placeholders = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }

    assert "label_guidance" not in placeholders
    assert placeholders == {
        "dialogue_context",
        "emotion_labels",
        "example_block",
        "likely_line",
        "response_block",
    }


def test_unknown_prompt_variant_is_rejected_before_rendering():
    with pytest.raises(ValueError):
        build_emotion_analysis_prompt(
            emotion_labels=["anxious"],
            emotion_label_set={"anxious"},
            dialogue_context="test",
            prompt_variant="unknown",
        )


def test_build_emotion_analysis_prompt_renders_examples_and_candidates():
    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "sad", "grateful"],
        emotion_label_set={"anxious", "sad", "grateful"},
        dialogue_context="I cannot sleep",
        current_input="I feel anxious",
        previous_emotion="sad",
        likely_emotions=["anxious", "unknown", "sad"],
    )

    assert "Emotion labels: anxious, sad, grateful" in prompt
    assert "anxious: uneasy worry about an uncertain outcome" in prompt
    assert "sad: general unhappiness or sorrow" in prompt
    assert "Response Format: Return exactly one JSON object" in prompt
    assert '"primary_emotion": "anxious"' in prompt
    assert '"confidence": 0.0' in prompt
    assert '"secondary_emotions": []' in prompt
    assert '"evidence": "short phrase from the dialogue"' in prompt
    assert '"reply_strategy": "brief guidance for the next chatbot reply"' in prompt
    assert '"trajectory_note": "optional change from prior emotion"' in prompt
    assert '"safety_level": "normal"' in prompt
    assert "More likely emotion labels: sad, anxious" in prompt
    assert "Labeled examples:" in prompt
    assert "True emotion label: anxious" in prompt
    assert "Dialogue context: I cannot sleep" in prompt


def test_build_emotion_analysis_prompt_renders_dynamic_examples():
    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "lonely"],
        emotion_label_set={"anxious", "lonely"},
        dialogue_context="I feel alone tonight",
        examples=[
            {
                "dialogue": "I feel alone tonight.",
                "emotion": "lonely",
                "score": 2.0,
                "reason": "weighted-overlap=alone:1.00,tonight:1.00",
            }
        ],
    )

    assert "Dynamic EICL examples:" in prompt
    assert "- Dialogue: I feel alone tonight." in prompt
    assert "True emotion label: lonely" in prompt
    assert "Selection reason: weighted-overlap=alone:1.00,tonight:1.00 (score=2.00)" in prompt
    assert "Response Format: Return exactly one JSON object" in prompt


def test_build_emotion_analysis_prompt_can_disable_examples():
    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "sad"],
        emotion_label_set={"anxious", "sad"},
        dialogue_context="I cannot sleep",
        current_input="I cannot sleep",
        examples=[],
        include_static_examples=False,
    )

    assert "Dynamic EICL examples:" not in prompt
    assert "Labeled examples:" not in prompt
    assert "Emotion labels: anxious, sad" in prompt
    assert "Dialogue context: I cannot sleep" in prompt


def test_build_emotion_analysis_prompt_uses_prompt_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / "prompts.json"
    config_file.write_text(
        '{"emotion_analysis": "Custom labels={emotion_labels}\\n'
        'Examples:{example_block}\\nLikely:{likely_line}\\nContext:{dialogue_context}"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(config_file))

    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "sad"],
        emotion_label_set={"anxious", "sad"},
        dialogue_context="I cannot sleep",
        previous_emotion="sad",
    )

    assert "Custom labels=anxious, sad" in prompt
    assert "Infer the user's current emotion" not in prompt
    assert "Labeled examples:" in prompt
    assert "Likely:\n- More likely emotion labels: sad" in prompt
    assert "Context:I cannot sleep" in prompt
