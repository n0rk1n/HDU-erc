import pytest
from tokenizers import Tokenizer
from emotion_lab.llm.counter import TokenCounter
from emotion_lab.llm.deepseek_tokens import TOKENIZER_PATH as DEEPSEEK
from emotion_lab.llm.glm53_tokens import TOKENIZER_PATH as GLM
from emotion_lab.taxonomy import parse_content


def test_preserved_deepseek_counter_includes_full_role_template():
    counter = TokenCounter("deepseek-v4-flash")
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "今天很开心😊"},
    ]
    prompt = (
        "<｜begin▁of▁sentence｜>你是助手。<｜User｜>你好<｜Assistant｜></think>你好！"
        "<｜end▁of▁sentence｜><｜User｜>今天很开心😊<｜Assistant｜><think>"
    )
    assert counter.count(messages) == len(
        Tokenizer.from_file(str(DEEPSEEK)).encode(prompt, add_special_tokens=False).ids
    )


def test_preserved_glm_counter_includes_reasoning_effort_template():
    counter = TokenCounter("ZHIPU/GLM-5.3")
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好"},
    ]
    tokenizer = Tokenizer.from_file(str(GLM))
    prompts = [
        f"[gMASK]<sop><|system|>Reasoning Effort: {effort}<|system|>你是助手。<|user|>你好<|assistant|><think>"
        for effort in ("Low", "High", "Max")
    ]
    assert counter.count(messages) == max(
        len(tokenizer.encode(p, add_special_tokens=False).ids) for p in prompts
    )


@pytest.mark.parametrize(
    "body,code",
    [
        ("not json", "invalid_json"),
        ('{"labels":[]}', "empty_labels"),
        ('{"labels":["joy","joy"]}', "duplicate_labels"),
        ('{"labels":["fake"]}', "unknown_labels"),
        ('{"labels":[1]}', "label_must_be_string"),
        ('{"labels":"joy"}', "labels_array_required"),
        ('{"labels":["joy"],"labels":["neutral"]}', "invalid_json"),
        ('{"labels":["joy"],"scores":{"joy":NaN}}', "invalid_json"),
    ],
)
def test_invalid_predictions_are_diagnostic_failures(body, code):
    _, errors = parse_content(body, {"joy", "neutral"})
    assert code in [e["code"] for e in errors]
