from chatbot.emotion_labels import (
    EMOTION_FAMILIES,
    EMOTION_LABEL_GUIDANCE,
    EMOTION_LABEL_SET,
    emotion_family,
)


def test_every_supported_label_has_guidance_and_diagnostic_family():
    assert set(EMOTION_LABEL_GUIDANCE) == EMOTION_LABEL_SET
    assert set(EMOTION_FAMILIES) == EMOTION_LABEL_SET


def test_adjacent_intensity_labels_share_expected_family():
    assert emotion_family("afraid") == emotion_family("terrified")
    assert emotion_family("annoyed") == emotion_family("angry")
    assert emotion_family("angry") == emotion_family("furious")
    assert emotion_family("anxious") == emotion_family("apprehensive")
