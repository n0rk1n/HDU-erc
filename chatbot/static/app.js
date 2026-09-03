(() => {
  "use strict";

  const identityView = document.querySelector("#identity-view");
  const identityForm = document.querySelector("#identity-form");
  const identifierInput = document.querySelector("#identifier-input");
  const identityStatus = document.querySelector("#identity-status");
  const chatView = document.querySelector("#chat-view");
  const currentIdentifier = document.querySelector("#current-identifier");
  const switchUserButton = document.querySelector("#switch-user");
  const messageList = document.querySelector("#message-list");
  const chatStatus = document.querySelector("#chat-status");
  const messageForm = document.querySelector("#message-form");
  const messageInput = document.querySelector("#message-input");
  const sendMessageButton = document.querySelector("#send-message");

  let activeSession = null;
  let streamController = null;
  let identityController = null;
  let generation = 0;
  let temporaryAssistant = null;

  function setStatus(element, text, state = "") {
    element.textContent = text;
    if (state) {
      element.dataset.state = state;
    } else {
      delete element.dataset.state;
    }
  }

  function showIdentityView() {
    identityView.hidden = false;
    chatView.hidden = true;
    identifierInput.focus();
  }

  function showChatView() {
    identityView.hidden = true;
    chatView.hidden = false;
    messageInput.focus();
  }

  function setSending(sending) {
    messageInput.disabled = sending;
    sendMessageButton.disabled = sending;
    switchUserButton.disabled = false;
  }

  function clearChat() {
    messageList.replaceChildren();
    temporaryAssistant = null;
    messageInput.value = "";
    setStatus(chatStatus, "");
    setSending(false);
  }

  function createMessage(message) {
    const item = document.createElement("li");
    const article = document.createElement("article");
    const body = document.createElement("p");
    const detail = document.createElement("small");
    const role = message.role === "user" ? "user" : "assistant";
    const state = typeof message.status === "string" ? message.status : "completed";

    item.className = `message message--${role}`;
    item.dataset.state = state;
    if (Number.isInteger(message.sequence_no)) {
      item.dataset.sequence = String(message.sequence_no);
    }
    article.setAttribute("aria-label", role === "user" ? "用户消息" : "助手消息");
    body.textContent = typeof message.content === "string" ? message.content : "";
    detail.textContent = state === "failed" ? "生成未完成" : "";
    article.append(body, detail);
    item.append(article);
    return item;
  }

  function appendMessage(message) {
    const item = createMessage(message);
    const sequence = Number(item.dataset.sequence);
    const siblings = [...messageList.children];
    const following = siblings.find((sibling) => {
      const siblingSequence = Number(sibling.dataset.sequence);
      return !Number.isInteger(siblingSequence) || siblingSequence > sequence;
    });
    messageList.insertBefore(item, following || null);
    return item;
  }

  function renderHistory(messages) {
    const publicMessages = Array.isArray(messages)
      ? messages
          .filter((message) => message && Number.isInteger(message.sequence_no))
          .sort((left, right) => left.sequence_no - right.sequence_no)
      : [];
    messageList.replaceChildren(...publicMessages.map(createMessage));
    temporaryAssistant = null;
  }

  function sessionIsCurrent(expectedGeneration, expectedUserId) {
    return (
      expectedGeneration === generation &&
      activeSession !== null &&
      activeSession.user_id === expectedUserId
    );
  }

  async function loadHistory(expectedGeneration = generation) {
    const session = activeSession;
    if (!session) return;
    const response = await fetch(`/api/users/${session.user_id}/messages`);
    if (!response.ok) throw new Error("history unavailable");
    const payload = await response.json();
    if (!sessionIsCurrent(expectedGeneration, session.user_id)) return;
    renderHistory(payload.messages);
  }

  function scheduleHistoryRefresh(expectedGeneration, expectedUserId) {
    window.setTimeout(() => {
      if (!sessionIsCurrent(expectedGeneration, expectedUserId)) return;
      loadHistory(expectedGeneration).catch(() => {
        if (sessionIsCurrent(expectedGeneration, expectedUserId)) {
          setStatus(chatStatus, "连接中断，暂时无法同步消息。", "failed");
        }
      });
    }, 300);
  }

  function ensureTemporaryAssistant() {
    if (!temporaryAssistant) {
      temporaryAssistant = appendMessage({ role: "assistant", status: "streaming", content: "" });
    }
    return temporaryAssistant;
  }

  function updateTemporaryAssistant(content, state) {
    const item = ensureTemporaryAssistant();
    item.dataset.state = state;
    const body = item.querySelector("p");
    const detail = item.querySelector("small");
    body.textContent = content;
    detail.textContent = state === "failed" ? "生成未完成" : "";
  }

  function parseFrame(frame) {
    let name = "message";
    const dataLines = [];
    for (const line of frame.split(/\r\n|\n|\r/)) {
      if (line.startsWith("event:")) name = line.slice(6).trimStart();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) return null;
    try {
      return { name, data: JSON.parse(dataLines.join("\n")) };
    } catch {
      return null;
    }
  }

  function consumeFrames(buffer, onEvent) {
    const boundary = /\r\n\r\n|\n\n|\r\r/;
    let match = boundary.exec(buffer);
    while (match) {
      const event = parseFrame(buffer.slice(0, match.index));
      if (event) onEvent(event);
      buffer = buffer.slice(match.index + match[0].length);
      match = boundary.exec(buffer);
    }
    return buffer;
  }

  function handleStreamEvent(event) {
    if (event.name === "run_started") {
      ensureTemporaryAssistant();
      setStatus(chatStatus, "正在生成回复…", "streaming");
    } else if (event.name === "user_message") {
      if (event.data && Number.isInteger(event.data.sequence_no)) {
        appendMessage({
          role: "user",
          status: "completed",
          content: event.data.content,
          sequence_no: event.data.sequence_no,
        });
      }
    } else if (event.name === "token" && event.data && typeof event.data.content === "string") {
      const item = ensureTemporaryAssistant();
      const body = item.querySelector("p");
      body.textContent += event.data.content;
      setStatus(chatStatus, "正在生成回复…", "streaming");
    } else if (event.name === "done" && event.data && event.data.message) {
      const message = event.data.message;
      updateTemporaryAssistant(typeof message.content === "string" ? message.content : "", "completed");
      setStatus(chatStatus, "回复已完成。", "completed");
      temporaryAssistant = null;
      return true;
    } else if (event.name === "error") {
      updateTemporaryAssistant(
        temporaryAssistant ? temporaryAssistant.querySelector("p").textContent : "",
        "failed"
      );
      setStatus(chatStatus, "回复未能完成，请稍后再试。", "failed");
      temporaryAssistant = null;
      return true;
    }
    return false;
  }

  async function sendMessage(content) {
    const session = activeSession;
    if (!session || !content.trim()) return;
    const expectedGeneration = generation;
    const controller = new AbortController();
    streamController = controller;
    setSending(true);
    setStatus(chatStatus, "正在发送消息…", "pending");
    let terminal = false;
    try {
      const response = await fetch(`/api/users/${session.user_id}/messages:stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ request_id: crypto.randomUUID(), content }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error("stream unavailable");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = consumeFrames(buffer, (event) => {
          terminal = handleStreamEvent(event) || terminal;
        });
      }
      buffer += decoder.decode();
      consumeFrames(buffer, (event) => {
        terminal = handleStreamEvent(event) || terminal;
      });
    } catch {
      if (sessionIsCurrent(expectedGeneration, session.user_id) && !controller.signal.aborted) {
        setStatus(chatStatus, "连接中断，正在同步已保存的消息。", "failed");
      }
    } finally {
      if (streamController === controller) streamController = null;
      if (!sessionIsCurrent(expectedGeneration, session.user_id)) return;
      setSending(false);
      if (!terminal) scheduleHistoryRefresh(expectedGeneration, session.user_id);
    }
  }

  identityForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const identifier = identifierInput.value.trim();
    if (!identifier) {
      setStatus(identityStatus, "请输入用户标识。", "failed");
      return;
    }
    identityController?.abort();
    const controller = new AbortController();
    identityController = controller;
    const expectedGeneration = ++generation;
    setStatus(identityStatus, "正在进入聊天…", "pending");
    try {
      const response = await fetch("/api/users/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("resolve unavailable");
      const payload = await response.json();
      if (expectedGeneration !== generation || !payload.user || !Number.isInteger(payload.user.id)) return;
      activeSession = { user_id: payload.user.id, identifier: payload.user.identifier };
      currentIdentifier.textContent = activeSession.identifier;
      clearChat();
      showChatView();
      await loadHistory(expectedGeneration);
      if (sessionIsCurrent(expectedGeneration, activeSession.user_id)) {
        setStatus(chatStatus, "可以开始聊天。", "completed");
      }
    } catch {
      if (!controller.signal.aborted && expectedGeneration === generation) {
        setStatus(identityStatus, "暂时无法进入聊天，请稍后再试。", "failed");
      }
    } finally {
      if (identityController === controller) identityController = null;
    }
  });

  messageForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (messageInput.disabled) return;
    const content = messageInput.value;
    if (!content.trim()) {
      setStatus(chatStatus, "请输入消息。", "failed");
      return;
    }
    messageInput.value = "";
    void sendMessage(content);
  });

  switchUserButton.addEventListener("click", () => {
    generation += 1;
    streamController?.abort();
    identityController?.abort();
    streamController = null;
    identityController = null;
    activeSession = null;
    clearChat();
    setStatus(identityStatus, "");
    showIdentityView();
  });

  window.addEventListener("pagehide", () => streamController?.abort());
  showIdentityView();
})();
