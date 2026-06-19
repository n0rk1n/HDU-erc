from chatbot.emotion_prompt import build_emotion_analysis_prompt


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
