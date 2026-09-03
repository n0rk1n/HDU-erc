from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from collections.abc import AsyncIterator, Sequence
from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import empty_checkpoint
from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.core.errors import ConfigError
from chatbot.db.messages import MessageRepository
from chatbot.llm.types import ModelDelta
from chatbot.services.identity import IdentityService
from chatbot.web import create_app


class OfflineModel:
    provider = "offline-test"
    parameters = {"model": "offline-test", "temperature": 0}

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        yield ModelDelta(content="ok")


class _ElementIdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.element_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.element_ids.add(identifier)


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    return AppConfig(
        llm_api_key=SecretStr("sk-test-secret"),
        llm_model="offline-test",
        llm_base_url=None,
        llm_temperature=0,
        llm_timeout_seconds=1,
        context_message_limit=20,
        sqlite_db_path=tmp_path / "chatbot.sqlite3",
    )


@pytest.fixture
def offline_model() -> OfflineModel:
    return OfflineModel()


def _pragma(connection, name: str):
    return connection.execute(f"PRAGMA {name}").fetchone()[0]


def test_lifespan_creates_official_saver_tables_and_applies_saver_pragmas(
    app_config, offline_model
) -> None:
    """Catches a separate saver connection missing contention or integrity PRAGMAs."""
    app = create_app(config=app_config, model=offline_model)
    with TestClient(app) as client:
        checkpointer = app.state.checkpointer

        async def read_saver_pragmas():
            values = {}
            for name in ("journal_mode", "foreign_keys", "busy_timeout", "synchronous"):
                row = await (await checkpointer.conn.execute(f"PRAGMA {name}")).fetchone()
                values[name] = row[0]
            return values

        saver_pragmas = client.portal.call(read_saver_pragmas)
        assert app.state.database.path == app_config.sqlite_db_path
        assert app.state.recovery_report.failed_messages == 0

    assert saver_pragmas == {
        "journal_mode": "wal",
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "synchronous": 1,
    }
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"users", "conversations", "messages", "checkpoints", "writes"} <= tables
        assert _pragma(connection, "journal_mode") == "wal"


def test_restart_fails_only_interrupted_business_turn_and_resets_its_checkpoint(
    app_config, offline_model
) -> None:
    """Catches restart deleting all threads or leaving stale running facts active."""
    first_app = create_app(config=app_config, model=offline_model)
    with TestClient(first_app) as client:
        async def seed():
            identity = IdentityService(first_app.state.database)
            normal = await identity.resolve("normal")
            stale = await identity.resolve("stale")
            messages = MessageRepository(first_app.state.database)
            normal_turn = await messages.reserve_turn(
                normal.conversation.id,
                "00000000-0000-4000-8000-000000000001",
                "normal",
            )
            await messages.mark_streaming(normal_turn.assistant.id)
            await messages.complete_assistant(
                normal_turn.assistant.id,
                content="done",
                reasoning_content="private",
                trace={"schema_version": 1},
                prompt=[],
                provider="test",
                model="test",
                parameters={},
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                latency_ms=1,
                finish_reason="stop",
            )
            await messages.reserve_turn(
                stale.conversation.id,
                "00000000-0000-4000-8000-000000000002",
                "stale",
            )
            for conversation in (normal.conversation, stale.conversation):
                await first_app.state.checkpointer.aput(
                    {
                        "configurable": {
                            "thread_id": conversation.thread_id,
                            "checkpoint_ns": "",
                        }
                    },
                    empty_checkpoint(),
                    {"source": "input", "step": -1, "parents": {}},
                    {},
                )
            return normal, stale

        normal, stale = client.portal.call(seed)

    restarted = create_app(config=app_config, model=offline_model)
    with TestClient(restarted) as client:
        async def inspect_restart():
            messages = MessageRepository(restarted.state.database)
            normal_history = await messages.list_visible(normal.conversation.id, limit=10)
            stale_history = await messages.list_visible(stale.conversation.id, limit=10)
            normal_checkpoint = await restarted.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": normal.conversation.thread_id}}
            )
            stale_checkpoint = await restarted.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": stale.conversation.thread_id}}
            )
            return normal_history, stale_history, normal_checkpoint, stale_checkpoint

        normal_history, stale_history, normal_checkpoint, stale_checkpoint = (
            client.portal.call(inspect_restart)
        )
        report = restarted.state.recovery_report

    assert [message["status"] for message in normal_history] == [
        "completed",
        "completed",
    ]
    assert [message["status"] for message in stale_history] == [
        "completed",
        "failed",
    ]
    assert stale_history[-1]["error_code"] == "process_interrupted"
    assert normal_checkpoint is not None
    assert stale_checkpoint is None
    assert (report.failed_messages, report.reset_threads) == (1, 1)


def test_create_app_without_config_is_import_safe_and_loads_env_at_startup(
    monkeypatch,
) -> None:
    """Catches module import eagerly requiring secrets or returning raw config details."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    app = create_app()

    with pytest.raises(ConfigError, match="LLM_API_KEY is required"):
        with TestClient(app):
            pass

    from chatbot.main import app as module_app

    assert module_app is not None


def test_shutdown_closes_saver_even_when_coordinator_shutdown_fails(
    app_config, offline_model
) -> None:
    """Catches shutdown exceptions leaking the independently owned saver connection."""
    app = create_app(config=app_config, model=offline_model)
    client = TestClient(app)
    client.__enter__()
    events: list[str] = []
    checkpointer = app.state.checkpointer
    original_close = checkpointer.conn.close

    async def failing_shutdown(timeout_seconds: float) -> None:
        events.append("coordinator")
        raise RuntimeError("shutdown failure")

    async def recording_close() -> None:
        events.append("saver")
        await original_close()

    app.state.coordinator.shutdown = failing_shutdown
    checkpointer.conn.close = recording_close

    with pytest.raises(RuntimeError, match="shutdown failure"):
        client.__exit__(None, None, None)

    assert events == ["coordinator", "saver"]


def test_recovery_failure_prevents_half_available_app(
    app_config, offline_model, monkeypatch
) -> None:
    """Catches a failed startup recovery being ignored while routes become available."""
    async def fail_recovery(messages, checkpointer):
        raise RuntimeError("recovery failed")

    monkeypatch.setattr("chatbot.web.recover_interrupted_turns", fail_recovery)
    app = create_app(config=app_config, model=offline_model)

    with pytest.raises(RuntimeError, match="recovery failed"):
        with TestClient(app):
            pass
    assert not hasattr(app.state, "coordinator")


def test_chat_page_serves_accessible_controls_assets_and_keeps_api_reachable(
    app_config, offline_model
) -> None:
    """Catches a missing SPA shell, incorrectly typed assets, or a static mount masking API routes."""
    app = create_app(config=app_config, model=offline_model)
    with TestClient(app) as client:
        index = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
        resolved = client.post("/api/users/resolve", json={"identifier": "web-user"})

    parser = _ElementIdParser()
    parser.feed(index.text)
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert {
        "identity-view",
        "identity-form",
        "identifier-input",
        "identity-status",
        "chat-view",
        "current-identifier",
        "switch-user",
        "message-list",
        "chat-status",
        "message-form",
        "message-input",
        "send-message",
    } <= parser.element_ids
    assert 'for="identifier-input"' in index.text
    assert 'for="message-input"' in index.text
    assert 'aria-live="polite"' in index.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith(("application/javascript", "text/javascript"))
    assert resolved.status_code == 200
    assert resolved.json()["user"]["identifier"] == "web-user"


def test_chat_script_has_the_minimal_safe_post_sse_contract(
    app_config, offline_model
) -> None:
    """Catches a UI that cannot safely drive the documented POST SSE protocol."""
    app = create_app(config=app_config, model=offline_model)
    with TestClient(app) as client:
        script = client.get("/static/app.js")

    assert script.status_code == 200
    source = script.text
    for required_fragment in (
        "crypto.randomUUID()",
        "method: \"POST\"",
        "response.body.getReader()",
        "new TextDecoder()",
        "AbortController",
        "event:",
        "data:",
        "sequence_no",
        "!Number.isInteger(siblingSequence)",
        "textContent",
    ):
        assert required_fragment in source
    for private_field in (
        "reasoning_content",
        "prompt_json",
        "parameters_json",
        "trace_json",
        "thread_id",
    ):
        assert private_field not in source
    assert (
        'setStatus(chatStatus, "回复已完成。", "completed");\n'
        "      temporaryAssistant = null;"
    ) in source


def test_chat_script_executes_recovery_ordering_and_initial_history_failures() -> None:
    """Catches a one-shot recovery, unsequenced done DOM, or a hidden initial-load error."""
    script_path = Path(__file__).parents[2] / "chatbot" / "static" / "app.js"
    harness = dedent(
        r'''
        const assert = require("node:assert/strict");
        const crypto = require("node:crypto");
        const fs = require("node:fs");
        const vm = require("node:vm");

        class Element {
          constructor(tag = "div") {
            this.tagName = tag;
            this.children = [];
            this.dataset = {};
            this.listeners = new Map();
            this.parentNode = null;
            this.textContent = "";
            this.value = "";
            this.disabled = false;
            this.hidden = false;
          }
          append(...nodes) {
            for (const node of nodes) {
              if (!(node instanceof Element)) continue;
              if (node.parentNode) node.parentNode.removeChild(node);
              node.parentNode = this;
              this.children.push(node);
            }
          }
          replaceChildren(...nodes) {
            for (const child of this.children) child.parentNode = null;
            this.children = [];
            this.append(...nodes);
          }
          removeChild(node) {
            const index = this.children.indexOf(node);
            if (index >= 0) this.children.splice(index, 1);
            node.parentNode = null;
          }
          insertBefore(node, reference) {
            if (node.parentNode) node.parentNode.removeChild(node);
            node.parentNode = this;
            const index = reference ? this.children.indexOf(reference) : -1;
            if (index < 0) this.children.push(node);
            else this.children.splice(index, 0, node);
          }
          querySelector(selector) {
            for (const child of this.children) {
              if (child.tagName === selector) return child;
              const nested = child.querySelector(selector);
              if (nested) return nested;
            }
            return null;
          }
          setAttribute() {}
          addEventListener(name, listener) { this.listeners.set(name, listener); }
          dispatch(name) {
            return this.listeners.get(name)({ preventDefault() {} });
          }
          focus() {}
        }

        function makeDocument() {
          const ids = [
            "identity-view", "identity-form", "identifier-input", "identity-status",
            "chat-view", "current-identifier", "switch-user", "message-list",
            "chat-status", "message-form", "message-input", "send-message",
          ];
          const nodes = Object.fromEntries(ids.map((id) => [id, new Element()]));
          return {
            nodes,
            querySelector(selector) { return nodes[selector.slice(1)]; },
            createElement(tag) { return new Element(tag); },
          };
        }

        function sse(events) {
          const text = events.map(({ name, data }) =>
            `event: ${name}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`
          ).join("");
          const chunks = [text.slice(0, 11), text.slice(11, 29), text.slice(29)];
          return {
            ok: true,
            body: {
              getReader() {
                let index = 0;
                return { read: async () => index < chunks.length
                  ? { value: Buffer.from(chunks[index++]), done: false }
                  : { value: undefined, done: true } };
              },
            },
          };
        }

        async function settle(rounds = 12) {
          for (let index = 0; index < rounds; index += 1) await Promise.resolve();
        }

        async function boot(fetch) {
          const document = makeDocument();
          const timers = [];
          const listeners = new Map();
          let nextTimerId = 1;
          const window = {
            addEventListener(name, listener) { listeners.set(name, listener); },
            dispatch(name) { return listeners.get(name)?.(); },
            setTimeout(callback) {
              callback.timerId = nextTimerId;
              nextTimerId += 1;
              timers.push(callback);
              return callback.timerId;
            },
            clearTimeout(timerId) {
              const index = timers.findIndex((callback) => callback.timerId === timerId);
              if (index >= 0) timers.splice(index, 1);
            },
          };
          const context = {
            AbortController,
            Buffer,
            TextDecoder,
            crypto: { randomUUID: () => "00000000-0000-4000-8000-000000000001" },
            document,
            fetch,
            window,
          };
          vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context, { filename: "app.js" });
          return { document, timers, window };
        }

        async function enter(app) {
          app.document.nodes["identifier-input"].value = "alice";
          await app.document.nodes["identity-form"].dispatch("submit");
          await settle();
        }

        async function runSlowRecovery() {
          let historyReads = 0;
          let posts = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              const assistant = historyReads === 1 ? [] : historyReads === 2
                ? [{ id: "user-1", role: "user", status: "completed", sequence_no: 1, content: "hello" }, { id: "assistant-1", request_id: "00000000-0000-4000-8000-000000000001", role: "assistant", status: "streaming", sequence_no: 2, content: "partial" }]
                : [{ id: "user-1", role: "user", status: "completed", sequence_no: 1, content: "hello" }, { id: "assistant-1", request_id: "00000000-0000-4000-8000-000000000001", role: "assistant", status: "completed", sequence_no: 2, content: "final" }];
              return { ok: true, json: async () => ({ messages: assistant }) };
            }
            if (url.endsWith(":stream")) {
              posts += 1;
              return sse([
                { name: "run_started", data: {} },
                { name: "user_message", data: { id: "user-1", sequence_no: 1, content: "hello" } },
                { name: "token", data: { content: "partial" } },
                { name: "error", data: { code: "model_error", message: "model generation failed" } },
              ]);
            }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle();
          assert.equal(app.document.nodes["message-input"].disabled, true, "recovery keeps sending disabled");
          for (let index = 0; index < 8; index += 1) {
            await settle(20);
            const timer = app.timers.shift();
            if (timer) timer();
          }
          const messages = app.document.nodes["message-list"].children;
          assert.equal(posts, 1, "recovery never reposts");
          assert.deepEqual(messages.map((item) => item.dataset.sequence), ["1", "2"]);
          assert.equal(messages[1].dataset.state, "completed");
          assert.equal(messages[1].querySelector("p").textContent, "final");
          assert.equal(app.document.nodes["message-input"].disabled, false);
        }

        async function runTwoTurns() {
          let turn = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) return { ok: true, json: async () => ({ messages: [] }) };
            if (url.endsWith(":stream")) {
              turn += 1;
              const userSequence = turn * 2 - 1;
              const assistantSequence = userSequence + 1;
              return sse([
                { name: "run_started", data: {} },
                { name: "user_message", data: { id: `user-${turn}`, sequence_no: userSequence, content: `question ${turn}` } },
                { name: "token", data: { content: "draft" } },
                { name: "done", data: { message: { id: `assistant-${turn}`, sequence_no: assistantSequence, status: "completed", content: `answer ${turn}` } } },
              ]);
            }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          for (const content of ["question 1", "question 2"]) {
            app.document.nodes["message-input"].value = content;
            app.document.nodes["message-form"].dispatch("submit");
            await settle(40);
          }
          const messages = app.document.nodes["message-list"].children;
          assert.deepEqual(messages.map((item) => item.dataset.sequence), ["1", "2", "3", "4"]);
          assert.deepEqual(messages.map((item) => item.dataset.messageId), ["user-1", "assistant-1", "user-2", "assistant-2"]);
          assert.deepEqual(messages.map((item) => item.dataset.state), ["completed", "completed", "completed", "completed"]);
          assert.deepEqual(messages.map((item) => item.querySelector("p").textContent), ["question 1", "answer 1", "question 2", "answer 2"]);
        }

        async function runFailedRecovery() {
          let historyReads = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              const messages = historyReads === 1 ? [] : [
                { id: "user-1", role: "user", status: "completed", sequence_no: 1, content: "hello" },
                { id: "assistant-1", request_id: "00000000-0000-4000-8000-000000000001", role: "assistant", status: "failed", sequence_no: 2, content: "partial" },
              ];
              return { ok: true, json: async () => ({ messages }) };
            }
            if (url.endsWith(":stream")) return sse([
              { name: "run_started", data: {} },
              { name: "user_message", data: { id: "user-1", sequence_no: 1, content: "hello" } },
              { name: "error", data: { code: "model_error", message: "model generation failed" } },
            ]);
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          app.timers.shift()();
          await settle(20);
          assert.equal(app.document.nodes["message-list"].children[1].dataset.state, "failed");
          assert.equal(app.document.nodes["chat-status"].dataset.state, "failed");
          assert.equal(app.document.nodes["message-input"].disabled, false);
        }

        async function runRejectedPost() {
          let historyReads = 0;
          let posts = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              return { ok: true, json: async () => ({ messages: [
                { id: "existing", role: "assistant", status: "completed", sequence_no: 2, content: "existing history" },
              ] }) };
            }
            if (url.endsWith(":stream")) {
              posts += 1;
              return { ok: false, json: async () => ({ error: { code: "turn_in_progress", message: "untrusted internal detail" } }) };
            }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          for (let index = 0; index < 4; index += 1) {
            const timer = app.timers.shift();
            if (timer) timer();
            await settle(20);
          }
          assert.equal(posts, 1);
          assert.equal(historyReads, 1, "header rejection does not poll an unaccepted request id");
          assert.equal(app.document.nodes["message-list"].children.length, 1);
          assert.equal(app.document.nodes["message-list"].children[0].querySelector("p").textContent, "existing history");
          assert.equal(app.document.nodes["message-input"].disabled, false);
          assert.equal(app.document.nodes["send-message"].disabled, false);
          assert.equal(app.document.nodes["switch-user"].disabled, false);
          assert.equal(app.document.nodes["chat-status"].dataset.state, "failed");
          assert.match(app.document.nodes["chat-status"].textContent, /已有消息正在生成/);
          assert.doesNotMatch(app.document.nodes["chat-status"].textContent, /untrusted internal detail/);
        }

        async function runDelayedRejectedPostDoesNotOverwriteNewUser() {
          let resolveRejectedJson;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") {
              const identifier = JSON.parse(options.body).identifier;
              return { ok: true, json: async () => ({ user: { id: identifier === "bob" ? 2 : 1, identifier } }) };
            }
            if (url.endsWith("/messages") && !options.method) {
              const messages = url.includes("/2/")
                ? [{ id: "bob-history", role: "assistant", status: "completed", sequence_no: 2, content: "bob history" }]
                : [];
              return { ok: true, json: async () => ({ messages }) };
            }
            if (url.endsWith(":stream")) return {
              ok: false,
              json: () => new Promise((resolve) => { resolveRejectedJson = resolve; }),
            };
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "alice message";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          app.document.nodes["switch-user"].dispatch("click");
          app.document.nodes["identifier-input"].value = "bob";
          await app.document.nodes["identity-form"].dispatch("submit");
          await settle(20);
          resolveRejectedJson({ error: { code: "turn_in_progress", message: "old response" } });
          await settle(20);
          assert.equal(app.document.nodes["chat-view"].hidden, false);
          assert.equal(app.document.nodes["current-identifier"].textContent, "bob");
          assert.equal(app.document.nodes["chat-status"].textContent, "可以开始聊天。");
          assert.equal(app.document.nodes["chat-status"].dataset.state, "completed");
          assert.equal(app.document.nodes["message-list"].children[0].querySelector("p").textContent, "bob history");
          assert.equal(app.document.nodes["message-input"].disabled, false);
        }

        async function runDelayedRejectedPostDoesNotDisplayAfterPagehide() {
          let resolveRejectedJson;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) return { ok: true, json: async () => ({ messages: [] }) };
            if (url.endsWith(":stream")) return {
              ok: false,
              json: () => new Promise((resolve) => { resolveRejectedJson = resolve; }),
            };
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          app.window.dispatch("pagehide");
          resolveRejectedJson({ error: { code: "turn_in_progress", message: "old response" } });
          await settle(20);
          assert.equal(app.document.nodes["chat-status"].textContent, "正在发送消息…");
          assert.equal(app.document.nodes["chat-status"].dataset.state, "pending");
          assert.equal(app.timers.length, 0);
        }

        async function runSwitchIgnoresLateHistory() {
          let resolveLateHistory;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              if (!resolveLateHistory) return { ok: true, json: async () => ({ messages: [] }) };
              return new Promise((resolve) => { resolveLateHistory = resolve; });
            }
            if (url.endsWith(":stream")) return sse([
              { name: "run_started", data: {} },
              { name: "user_message", data: { id: "user-1", sequence_no: 1, content: "hello" } },
            ]);
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          resolveLateHistory = () => {};
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          app.timers.shift()();
          await settle(20);
          app.document.nodes["switch-user"].dispatch("click");
          resolveLateHistory({ ok: true, json: async () => ({ messages: [
            { id: "old", role: "assistant", status: "completed", sequence_no: 2, content: "old user history" },
          ] }) });
          await settle(20);
          assert.equal(app.document.nodes["identity-view"].hidden, false);
          assert.equal(app.document.nodes["chat-view"].hidden, true);
          assert.equal(app.document.nodes["message-list"].children.length, 0);
          assert.equal(app.timers.length, 0);
        }

        async function runPagehideStopsPolling() {
          let historyReads = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              return { ok: true, json: async () => ({ messages: [] }) };
            }
            if (url.endsWith(":stream")) return sse([
              { name: "run_started", data: {} },
              { name: "user_message", data: { id: "user-1", sequence_no: 1, content: "hello" } },
            ]);
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "hello";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(20);
          assert.equal(app.timers.length, 1);
          app.window.dispatch("pagehide");
          await settle(20);
          assert.equal(app.timers.length, 0);
          assert.equal(historyReads, 1);
          assert.notEqual(app.document.nodes["chat-status"].dataset.state, "failed");
        }

        async function runInitialHistoryFailure() {
          const app = await boot(async (url) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages")) throw new Error("offline");
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          assert.equal(app.document.nodes["chat-view"].hidden, false);
          assert.equal(app.document.nodes["message-input"].disabled, true);
          assert.equal(app.document.nodes["send-message"].disabled, true);
          assert.equal(app.document.nodes["switch-user"].disabled, false);
          assert.equal(app.document.nodes["chat-status"].dataset.state, "failed");
          assert.match(app.document.nodes["chat-status"].textContent, /历史加载失败/);
        }

        Promise.resolve()
          .then(runSlowRecovery)
          .then(runTwoTurns)
          .then(runFailedRecovery)
          .then(runRejectedPost)
          .then(runDelayedRejectedPostDoesNotOverwriteNewUser)
          .then(runDelayedRejectedPostDoesNotDisplayAfterPagehide)
          .then(runSwitchIgnoresLateHistory)
          .then(runPagehideStopsPolling)
          .then(runInitialHistoryFailure)
          .then(() => process.stdout.write("frontend scenarios passed\n"))
          .catch((error) => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
        '''
    )
    completed = subprocess.run(
        ["node", "-e", harness, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "frontend scenarios passed\n"
