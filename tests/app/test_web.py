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
from tests.emotion.helpers import create_app


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
    async def fail_recovery(messages, checkpointer, *, emotions, gates):
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
            this.scrollHeight = 0;
            this.style = {};
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
            if (reference && !this.children.includes(reference)) throw new Error("NotFoundError: reference is not a direct child");
            if (node.parentNode) node.parentNode.removeChild(node);
            node.parentNode = this;
            const index = reference ? this.children.indexOf(reference) : -1;
            if (index < 0) this.children.push(node);
            else this.children.splice(index, 0, node);
          }
          querySelector(selector) {
            for (const child of this.children) {
              if (child.tagName === selector || (selector.startsWith(".") &&
                  (child.className || "").split(" ").includes(selector.slice(1)))) return child;
              const nested = child.querySelector(selector);
              if (nested) return nested;
            }
            return null;
          }
          setAttribute() {}
          addEventListener(name, listener) { this.listeners.set(name, listener); }
          dispatch(name, event = {}) {
            return this.listeners.get(name)?.({ preventDefault() {}, ...event });
          }
          requestSubmit() { return this.dispatch("submit"); }
          focus() {}
        }

        function makeDocument() {
          const ids = [
            "identity-view", "identity-form", "identifier-input", "identity-status",
            "chat-view", "current-identifier", "switch-user", "message-list",
            "chat-status", "message-form", "message-input", "send-message",
            "emotion-label", "emotion-meta", "emotion-badge", "conversation-stage",
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

        async function settle() {
          // Let the real microtask queue drain; display timers stay manually controlled.
          await new Promise(setImmediate);
        }

        async function boot(fetch) {
          const document = makeDocument();
          const timers = [];
          const listeners = new Map();
          let nextTimerId = 1;
          const window = {
            addEventListener(name, listener) { listeners.set(name, listener); },
            dispatch(name, event = {}) { return listeners.get(name)?.(event); },
            setTimeout(callback, milliseconds) {
              callback.milliseconds = milliseconds;
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
          const presentationPath = require("node:path").join(require("node:path").dirname(process.argv[1]), "presentation.js");
          vm.runInNewContext(fs.readFileSync(presentationPath, "utf8"), context, { filename: "presentation.js" });
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

        async function runSwitchBackToStreamingHistoryReconcilesWithoutPost() {
          let historyReads = 0;
          let posts = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              const messages = historyReads === 1 ? [] : historyReads === 2 ? [
                { id: "user-active", request_id: "active-request", role: "user", status: "completed", sequence_no: 1, content: "question" },
                { id: "assistant-active", request_id: "active-request", role: "assistant", status: "streaming", sequence_no: 2, content: "partial" },
              ] : [
                { id: "user-active", request_id: "active-request", role: "user", status: "completed", sequence_no: 1, content: "question" },
                { id: "assistant-active", request_id: "active-request", role: "assistant", status: "completed", sequence_no: 2, content: "final" },
              ];
              return { ok: true, json: async () => ({ messages }) };
            }
            if (url.endsWith(":stream")) { posts += 1; throw new Error("must not post"); }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          app.document.nodes["switch-user"].dispatch("click");
          await enter(app);
          assert.equal(app.document.nodes["message-input"].disabled, true, "streaming history keeps send disabled");
          assert.equal(app.timers.length, 1, "streaming history starts one reconciliation timer");
          app.timers.shift()();
          await settle(30);
          const assistant = app.document.nodes["message-list"].children[1];
          assert.equal(posts, 0, "history recovery never creates a new turn");
          assert.equal(assistant.dataset.state, "completed");
          assert.equal(assistant.querySelector("p").textContent, "final");
          assert.equal(app.document.nodes["message-input"].disabled, false);
          assert.equal(app.timers.length, 0);
        }

        async function runBfcachePageshowRestoresStreamingHistoryWithoutPost() {
          let historyReads = 0;
          let posts = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) {
              historyReads += 1;
              const terminal = historyReads >= 3;
              return { ok: true, json: async () => ({ messages: [
                { id: "user-bfcache", request_id: "bfcache-request", role: "user", status: "completed", sequence_no: 1, content: "question" },
                { id: "assistant-bfcache", request_id: "bfcache-request", role: "assistant", status: terminal ? "completed" : "streaming", sequence_no: 2, content: terminal ? "final" : "partial" },
              ] }) };
            }
            if (url.endsWith(":stream")) { posts += 1; throw new Error("must not post"); }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          assert.equal(app.timers.length, 1);
          app.window.dispatch("pagehide", { persisted: true });
          await settle(20);
          assert.equal(app.timers.length, 0, "pagehide removes the old reconciliation timer");
          app.window.dispatch("pageshow", { persisted: true });
          await settle(30);
          assert.equal(app.timers.length, 1, "pageshow creates only one fresh timer");
          app.timers.shift()();
          await settle(30);
          const assistant = app.document.nodes["message-list"].children[1];
          assert.equal(posts, 0);
          assert.equal(assistant.dataset.state, "completed");
          assert.equal(assistant.querySelector("p").textContent, "final");
          assert.equal(app.document.nodes["message-input"].disabled, false);
          assert.equal(app.timers.length, 0);
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

        async function runComposerKeyboardAndAutoResize() {
          let posts = 0;
          const app = await boot(async (url, options = {}) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages") && !options.method) return { ok: true, json: async () => ({ messages: [] }) };
            if (url.endsWith(":stream")) {
              posts += 1;
              return sse([
                { name: "run_started", data: {} },
                { name: "user_message", data: { id: "user-keyboard", sequence_no: 1, content: "line 1" } },
                { name: "done", data: { message: { id: "assistant-keyboard", sequence_no: 2, status: "completed", content: "done" } } },
              ]);
            }
            throw new Error(`unexpected ${url}`);
          });
          await enter(app);
          const input = app.document.nodes["message-input"];
          input.value = "line 1\nline 2\nline 3";
          input.scrollHeight = 96;
          input.dispatch("input");
          assert.equal(input.style.height, "96px", "typing grows the composer to its content height");

          const draft = input.value;
          for (const keys of [
            {},
            { shiftKey: true },
            { ctrlKey: true, isComposing: true },
            { metaKey: true, isComposing: true },
            { ctrlKey: true, keyCode: 229 },
            { metaKey: true, keyCode: 229 },
          ]) {
            let prevented = false;
            input.dispatch("keydown", {
              key: "Enter", shiftKey: false, isComposing: false,
              ...keys,
              preventDefault() { prevented = true; },
            });
            await settle();
            assert.equal(prevented, false, "newline and IME confirmation keep their default behavior");
            assert.equal(posts, 0, "unfinished drafts must not be sent");
            assert.equal(input.value, draft, "unfinished drafts must not be cleared");
          }

          for (const modifier of ["ctrlKey", "metaKey"]) {
            input.value = "line 1";
            let prevented = false;
            input.dispatch("keydown", {
              key: "Enter", [modifier]: true, shiftKey: false, isComposing: false,
              preventDefault() { prevented = true; },
            });
            await settle(40);
            assert.equal(prevented, true, "send shortcut prevents a newline");
            assert.equal(input.value, "");
            assert.equal(input.style.height, "", "sending restores the default composer height");
          }
          assert.equal(posts, 2, "Ctrl+Enter and Command+Enter each send once");
        }

        function descendantText(element) {
          return [element.textContent, ...element.children.map(descendantText)].join(" ");
        }

        async function runEmotionPresentation() {
          const emotion = { label: "sad", display_label: "难过", confidence: .9,
            evidence: "<img src=x onerror=alert(1)>", source_message_id: "u1",
            sequence_no: 1, analyzed_at: "2026-09-07T03:00:00+00:00" };
          const processing = { emotion_status: "completed", emotion_invoked: true,
            emotion, elapsed_ms: 2400, started_at: "2026-09-07T02:59:58+00:00",
            steps: [{ id: "received", status: "completed" },
              { id: "decision", status: "completed" }, { id: "emotion", status: "completed" },
              { id: "response", status: "completed" }] };
          let user = 0;
          const app = await boot(async (url) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({
              user: { id: ++user, identifier: user === 1 ? "alice" : "bob" }, conversation: { id: "c" } }) };
            if (url.endsWith("/messages")) return { ok: true, json: async () => ({
              latest_emotion: user === 1 ? emotion : null,
              messages: user === 1 ? [{ id: "a1", request_id: "r1", sequence_no: 2,
                role: "assistant", content: "回答", status: "completed", processing }] : [] }) };
            return sse([
              { name: "run_started", data: {} },
              { name: "user_message", data: { id: "u2", sequence_no: 3, content: "继续聊聊" } },
              { name: "progress", data: { processing: { ...processing, emotion: null,
                emotion_status: "skipped", steps: [{ id: "emotion", status: "skipped" },
                  { id: "response", status: "running" }] }, latest_emotion: emotion } },
              { name: "token", data: { content: "新回答" } },
              { name: "done", data: { message: { id: "a2", sequence_no: 4, status: "completed",
                content: "新回答", processing: { ...processing, emotion_status: "skipped", emotion: null,
                  steps: [{ id: "emotion", status: "skipped" }, { id: "response", status: "completed" }] } },
                latest_emotion: emotion } },
            ]);
          });
          await enter(app);
          const first = app.document.nodes["message-list"].children[0];
          assert.equal(app.document.nodes["emotion-label"].textContent, "难过");
          assert.match(descendantText(first), /90%/);
          assert.match(descendantText(first), /<img src=x onerror=alert\(1\)>/);
          assert.equal(first.querySelector("img"), null, "evidence is plain text, never executable HTML");
          assert.equal(first.querySelector("p").textContent, "回答", "card does not replace reply text");
          const card = first.querySelector("details");
          assert.equal(card.open, false);
          app.document.nodes["message-input"].value = "继续聊聊";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(60);
          const last = app.document.nodes["message-list"].children[2];
          assert.equal(last.querySelector("p").textContent, "新回答");
          assert.match(descendantText(last), /本轮未调用情绪识别/);
          assert.doesNotMatch(descendantText(last), /90%/);
          assert.equal(app.document.nodes["emotion-label"].textContent, "难过");
          app.document.nodes["switch-user"].dispatch("click");
          assert.equal(app.document.nodes["emotion-label"].textContent, "尚未识别");
          await enter(app);
          assert.equal(app.document.nodes["emotion-label"].textContent, "尚未识别");
          assert.equal(app.document.nodes["message-list"].children.length, 0);
        }

        async function runBubblePacingAndHistory() {
          const reply = { id: "a", request_id: "r", role: "assistant", sequence_no: 2,
            status: "completed", content: "第一条。\n\n第二条。\n这里仍是同一个气泡。",
            bubbles: ["第一条。", "第二条。\n这里仍是同一个气泡。"] };
          let posted = false;
          const app = await boot(async (url) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            if (url.endsWith("/messages")) return { ok: true, json: async () => ({ messages: posted ? [reply] : [] }) };
            posted = true;
            return sse([
              { name: "run_started", data: { assistant_message_id: "a", bubble_gap_ms: 750 } },
              { name: "user_message", data: { id: "u", sequence_no: 1, content: "你好" } },
              { name: "token", data: { content: "不可逐字显示的内部流" } },
              { name: "bubble", data: { assistant_message_id: "a", index: 0, content: reply.bubbles[0] } },
              { name: "bubble", data: { assistant_message_id: "a", index: 0, content: reply.bubbles[0] } },
              { name: "bubble", data: { assistant_message_id: "a", index: 1, content: reply.bubbles[1] } },
              { name: "done", data: { message: reply } },
            ]);
          });
          await enter(app);
          app.document.nodes["message-input"].value = "你好";
          app.document.nodes["message-form"].dispatch("submit");
          await settle(60);
          const assistant = app.document.nodes["message-list"].children[1];
          let body = assistant.querySelector(".message-bubbles");
          assert.ok(body, "reply must own a bubble container");
          assert.deepEqual(body.children.map((node) => node.textContent), ["第一条。"]);
          assert.equal(app.document.nodes["message-input"].disabled, true);
          assert.equal(app.timers.length, 1, "only the second distinct bubble waits");
          assert.ok(app.timers[0].milliseconds > 0 && app.timers[0].milliseconds <= 750);
          app.timers.shift()();
          await settle(60);
          body = assistant.querySelector(".message-bubbles");
          assert.deepEqual(body.children.map((node) => node.textContent), reply.bubbles);
          assert.equal(app.document.nodes["message-input"].disabled, false, "done must wait for visible bubbles");
          app.document.nodes["switch-user"].dispatch("click");
          await enter(app);
          body = app.document.nodes["message-list"].children[0].querySelector(".message-bubbles");
          assert.deepEqual(body.children.map((node) => node.textContent), reply.bubbles, "history preserves boundaries");
          assert.equal(app.timers.length, 0, "history does not replay delays");
        }

        async function runHistoryLoadingKeepsSendDisabled() {
          let releaseHistory;
          const app = await boot(async (url) => {
            if (url === "/api/users/resolve") return { ok: true, json: async () => ({ user: { id: 1, identifier: "alice" } }) };
            return new Promise(resolve => { releaseHistory = () => resolve({ ok: true, json: async () => ({ messages: [] }) }); });
          });
          app.document.nodes["identifier-input"].value = "alice";
          const entering = app.document.nodes["identity-form"].dispatch("submit");
          await settle();
          assert.equal(app.document.nodes["chat-view"].hidden, false);
          assert.equal(app.document.nodes["message-input"].disabled, true, "late history must not overwrite a new reply");
          releaseHistory();
          await entering;
          assert.equal(app.document.nodes["message-input"].disabled, false);
        }

        Promise.resolve()
          .then(runHistoryLoadingKeepsSendDisabled)
          .then(runBubblePacingAndHistory)
          .then(runEmotionPresentation)
          .then(runSlowRecovery)
          .then(runTwoTurns)
          .then(runFailedRecovery)
          .then(runRejectedPost)
          .then(runDelayedRejectedPostDoesNotOverwriteNewUser)
          .then(runDelayedRejectedPostDoesNotDisplayAfterPagehide)
          .then(runSwitchIgnoresLateHistory)
          .then(runPagehideStopsPolling)
          .then(runSwitchBackToStreamingHistoryReconcilesWithoutPost)
          .then(runBfcachePageshowRestoresStreamingHistoryWithoutPost)
          .then(runInitialHistoryFailure)
          .then(runComposerKeyboardAndAutoResize)
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
