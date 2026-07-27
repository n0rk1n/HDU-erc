from chatbot.emotion_examples import EmotionExample
from chatbot.emotion_retrieval import select_dynamic_examples


EXAMPLES = [
    {"dialogue": "I am scared about the exam.", "emotion": "anxious"},
    {"dialogue": "Thank you for helping me.", "emotion": "grateful"},
    {"dialogue": "I feel alone tonight.", "emotion": "lonely"},
    {"dialogue": "I am so excited about this offer.", "emotion": "excited"},
]


def test_select_dynamic_examples_scores_relevant_example_first():
    selected = select_dynamic_examples(
        examples=EXAMPLES,
        dialogue_context="I am scared and worried about tomorrow's exam.",
        likely_emotions=[],
        limit=3,
    )

    assert selected[0]["emotion"] == "anxious"
    assert selected[0]["score"] > 0
    assert selected[0]["reason"]


def test_select_dynamic_examples_boosts_likely_emotions_but_keeps_diversity():
    selected = select_dynamic_examples(
        examples=EXAMPLES,
        dialogue_context="Thanks, but I still feel alone.",
        likely_emotions=["grateful", "lonely"],
        limit=3,
    )

    emotions = [item["emotion"] for item in selected]
    assert "grateful" in emotions
    assert "lonely" in emotions
    assert len(set(emotions)) == len(emotions)


def test_select_dynamic_examples_accepts_emotion_example_dataclasses():
    selected = select_dynamic_examples(
        examples=[EmotionExample("I feel alone tonight.", "lonely")],
        dialogue_context="alone tonight",
        likely_emotions=[],
        limit=1,
    )

    assert selected == [{
        "index": 0,
        "dialogue": "I feel alone tonight.",
        "emotion": "lonely",
        "score": 2.0,
        "reason": "weighted-overlap=alone:1.00,tonight:1.00",
    }]


def test_select_dynamic_examples_ignores_common_stopwords():
    selected = select_dynamic_examples(
        examples=[
            {"dialogue": "I am in the room with you.", "emotion": "content"},
            {"dialogue": "The exam deadline makes me worried.", "emotion": "anxious"},
        ],
        dialogue_context="I am worried about the exam.",
        limit=1,
    )

    assert selected[0]["emotion"] == "anxious"
    assert "exam" in selected[0]["reason"]
