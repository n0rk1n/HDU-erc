from chatbot.memory.extractor import extract_memory_candidates


def test_extracts_chinese_preference():
    candidates = extract_memory_candidates("我希望以后都用中文回答。")

    assert len(candidates) == 1
    assert candidates[0].content == "用户希望以后都用中文回答。"
    assert candidates[0].category == "preference"


def test_extracts_chinese_boundary():
    candidates = extract_memory_candidates("不要把记忆存到第三方托管服务。")

    assert len(candidates) == 1
    assert candidates[0].content == "用户要求不要把记忆存到第三方托管服务。"
    assert candidates[0].category == "boundary"


def test_extracts_english_preference():
    candidates = extract_memory_candidates("I prefer concise answers.")

    assert len(candidates) == 1
    assert candidates[0].content == "User prefers concise answers."
    assert candidates[0].category == "preference"


def test_extracts_english_boundary():
    candidates = extract_memory_candidates("Do not use hosted memory storage.")

    assert len(candidates) == 1
    assert candidates[0].content == "User requested not to use hosted memory storage."
    assert candidates[0].category == "boundary"


def test_ignores_transient_chat():
    assert extract_memory_candidates("今天有点累。") == []


def test_limits_to_three_candidates():
    message = (
        "我喜欢中文回答。\n"
        "我希望回答简洁。\n"
        "不要使用第三方存储。\n"
        "我的项目是情绪识别聊天机器人。"
    )

    candidates = extract_memory_candidates(message)

    assert len(candidates) == 3
