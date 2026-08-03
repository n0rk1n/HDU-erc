"""Shared emotion labels."""

EMOTION_LABELS = [
    "surprised",
    "excited",
    "annoyed",
    "proud",
    "angry",
    "sad",
    "grateful",
    "lonely",
    "impressed",
    "afraid",
    "disgusted",
    "confident",
    "terrified",
    "hopeful",
    "anxious",
    "disappointed",
    "joyful",
    "prepared",
    "guilty",
    "furious",
    "nostalgic",
    "jealous",
    "anticipating",
    "embarrassed",
    "content",
    "devastated",
    "sentimental",
    "caring",
    "trusting",
    "ashamed",
    "apprehensive",
    "faithful",
]

EMOTION_LABEL_SET = set(EMOTION_LABELS)

# Short operational definitions make adjacent labels explicit to both the model and
# evaluators.  They intentionally describe the situation appraisal, which is how
# EmpatheticDialogues collected its conversation-level labels.
EMOTION_LABEL_GUIDANCE = {
    "surprised": "an unexpected event, without assuming it is good or bad",
    "excited": "high-energy positive anticipation or enthusiasm",
    "annoyed": "mild irritation, less intense than angry or furious",
    "proud": "satisfaction in one's own or another person's achievement",
    "angry": "clear anger at a wrong or obstacle, stronger than annoyed",
    "sad": "general unhappiness or sorrow",
    "grateful": "appreciation for help, kindness, or a benefit received",
    "lonely": "distress from lacking companionship or connection",
    "impressed": "admiration caused by something notably skillful or good",
    "afraid": "fear of a concrete threat, less overwhelming than terrified",
    "disgusted": "strong revulsion or moral/physical aversion",
    "confident": "belief in one's ability or a favorable outcome",
    "terrified": "overwhelming or extreme fear",
    "hopeful": "desire and belief that a positive outcome may happen",
    "anxious": "uneasy worry about an uncertain outcome",
    "disappointed": "sadness because an expectation was not met",
    "joyful": "strong happiness caused by a positive event",
    "prepared": "readiness because planning or practice is complete",
    "guilty": "remorse for one's own action or responsibility",
    "furious": "extreme, highly intense anger",
    "nostalgic": "longing for or warmly recalling the past",
    "jealous": "fear or resentment about losing status, attention, or a bond",
    "anticipating": "looking forward to an upcoming event; valence may be mixed",
    "embarrassed": "social discomfort after awkward exposure or attention",
    "content": "calm satisfaction, lower arousal than joyful or excited",
    "devastated": "overwhelming sadness after severe loss or bad news",
    "sentimental": "tender emotional attachment to a person, object, or memory",
    "caring": "concern for another person's wellbeing",
    "trusting": "willingness to rely on another person or believe them",
    "ashamed": "painful negative judgment of oneself, broader than guilt about an act",
    "apprehensive": "cautious fear about a possible future problem",
    "faithful": "loyal commitment to a person, relationship, or belief",
}

EMOTION_FAMILIES = {
    "surprised": "surprise",
    "excited": "positive_anticipation",
    "annoyed": "anger",
    "proud": "competence",
    "angry": "anger",
    "sad": "sadness_loss",
    "grateful": "social_bond",
    "lonely": "sadness_loss",
    "impressed": "admiration",
    "afraid": "acute_fear",
    "disgusted": "aversion",
    "confident": "competence",
    "terrified": "acute_fear",
    "hopeful": "positive_anticipation",
    "anxious": "future_fear",
    "disappointed": "sadness_loss",
    "joyful": "joy_contentment",
    "prepared": "competence",
    "guilty": "self_conscious",
    "furious": "anger",
    "nostalgic": "memory_attachment",
    "jealous": "jealousy",
    "anticipating": "positive_anticipation",
    "embarrassed": "self_conscious",
    "content": "joy_contentment",
    "devastated": "sadness_loss",
    "sentimental": "memory_attachment",
    "caring": "social_bond",
    "trusting": "social_bond",
    "ashamed": "self_conscious",
    "apprehensive": "future_fear",
    "faithful": "social_bond",
}


def emotion_family(label: str) -> str:
    """Return a stable diagnostic family while preserving exact labels as primary."""
    return EMOTION_FAMILIES.get(label.strip().lower(), label.strip().lower())


def format_emotion_label_guidance(labels: list[str]) -> str:
    return "\n".join(
        f"  - {label}: {EMOTION_LABEL_GUIDANCE[label]}"
        for label in labels
        if label in EMOTION_LABEL_GUIDANCE
    )
