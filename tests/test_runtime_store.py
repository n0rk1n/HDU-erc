from chatbot.runtime_store import RuntimeStore


def test_runtime_store_appends_and_replaces_json_records(tmp_path):
    store = RuntimeStore(str(tmp_path / "runtime.sqlite3"))

    store.append_json_record("chat_history", {"role": "human", "content": "hello"})
    store.append_json_record("chat_history", {"role": "ai", "content": "hi"})

    assert store.load_json_records("chat_history") == [
        {"role": "human", "content": "hello"},
        {"role": "ai", "content": "hi"},
    ]

    store.replace_json_records("chat_history", [{"role": "human", "content": "reset"}])

    assert store.load_json_records("chat_history") == [
        {"role": "human", "content": "reset"},
    ]


def test_runtime_store_loads_profile_values(tmp_path):
    store = RuntimeStore(str(tmp_path / "runtime.sqlite3"))

    store.replace_profile({"name": "Alice", "age": "", "mbti": "INTP"})

    assert store.load_profile() == {"name": "Alice", "mbti": "INTP"}
