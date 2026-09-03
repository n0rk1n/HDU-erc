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
  let reconciliationController = null;
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

  function resizeMessageInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${messageInput.scrollHeight}px`;
  }

  function resetMessageInput() {
    messageInput.value = "";
    messageInput.style.height = "";
  }

  function clearChat() {
    messageList.replaceChildren();
    temporaryAssistant = null;
    resetMessageInput();
    setStatus(chatStatus, "");
    setSending(false);
  }

  function applyMessage(item, message) {
    const state = typeof message.status === "string" ? message.status : "completed";
    const body = item.querySelector("p");
    const detail = item.querySelector("small");

    item.dataset.state = state;
    if (typeof message.id === "string") item.dataset.messageId = message.id;
    if (Number.isInteger(message.sequence_no)) item.dataset.sequence = String(message.sequence_no);
    body.textContent = typeof message.content === "string" ? message.content : "";
    detail.textContent = state === "failed" ? "生成未完成" : "";
  }

  function createMessage(message) {
    const item = document.createElement("li");
    const article = document.createElement("article");
    const body = document.createElement("p");
    const detail = document.createElement("small");
    const role = message.role === "user" ? "user" : "assistant";

    item.className = `message message--${role}`;
    article.setAttribute("aria-label", role === "user" ? "用户消息" : "助手消息");
    article.append(body, detail);
    item.append(article);
    applyMessage(item, message);
    return item;
  }

  function placeMessage(item) {
    const sequence = Number(item.dataset.sequence);
    if (!Number.isInteger(sequence)) {
      messageList.append(item);
      return;
    }
    const siblings = [...messageList.children].filter((sibling) => sibling !== item);
    const following = siblings.find((sibling) => {
      const siblingSequence = Number(sibling.dataset.sequence);
      return !Number.isInteger(siblingSequence) || siblingSequence > sequence;
    });
    messageList.insertBefore(item, following ?? null);
  }

  function appendMessage(message) {
    const item = createMessage(message);
    placeMessage(item);
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

  function sendIsCurrent(controller, expectedGeneration, expectedUserId) {
    return (
      !controller.signal.aborted &&
      sessionIsCurrent(expectedGeneration, expectedUserId)
    );
  }

  async function loadHistory(expectedGeneration = generation, signal) {
    const session = activeSession;
    if (!session) return;
    const response = await fetch(`/api/users/${session.user_id}/messages`, { signal });
    if (!response.ok) throw new Error("history unavailable");
    const payload = await response.json();
    if (!sessionIsCurrent(expectedGeneration, session.user_id)) return;
    renderHistory(payload.messages);
    return Array.isArray(payload.messages) ? payload.messages : [];
  }

  function waitForDelay(milliseconds, signal) {
    return new Promise((resolve) => {
      if (signal.aborted) {
        resolve(false);
        return;
      }
      let timerId;
      const stop = () => {
        window.clearTimeout(timerId);
        signal.removeEventListener("abort", stop);
        resolve(false);
      };
      signal.addEventListener("abort", stop, { once: true });
      timerId = window.setTimeout(() => {
        signal.removeEventListener("abort", stop);
        resolve(!signal.aborted);
      }, milliseconds);
    });
  }

  function findTerminalAssistant(messages, requestId) {
    return messages.find(
      (message) =>
        message &&
        message.role === "assistant" &&
        message.request_id === requestId &&
        (message.status === "completed" || message.status === "failed")
    );
  }

  function findActiveAssistant(messages) {
    return messages
      .filter(
        (message) =>
          message &&
          message.role === "assistant" &&
          typeof message.request_id === "string" &&
          (message.status === "pending" || message.status === "streaming")
      )
      .sort((left, right) => right.sequence_no - left.sequence_no)[0];
  }

  function resumeActiveHistory(messages, expectedGeneration, expectedUserId) {
    const activeAssistant = findActiveAssistant(messages);
    if (!activeAssistant) return false;
    setSending(true);
    setStatus(chatStatus, "消息仍在生成，正在同步已保存的内容。", "pending");
    void reconcileHistory(
      expectedGeneration,
      expectedUserId,
      activeAssistant.request_id
    );
    return true;
  }

  async function reconcileHistory(expectedGeneration, expectedUserId, requestId) {
    reconciliationController?.abort();
    const controller = new AbortController();
    reconciliationController = controller;
    let delay = 300;
    try {
      while (!controller.signal.aborted && sessionIsCurrent(expectedGeneration, expectedUserId)) {
        const shouldContinue = await waitForDelay(delay, controller.signal);
        if (!shouldContinue || !sessionIsCurrent(expectedGeneration, expectedUserId)) return;
        try {
          const messages = await loadHistory(expectedGeneration, controller.signal);
          if (!sessionIsCurrent(expectedGeneration, expectedUserId)) return;
          const terminalAssistant = findTerminalAssistant(messages ?? [], requestId);
          if (terminalAssistant) {
            if (terminalAssistant.status === "failed") {
              setStatus(chatStatus, "回复未能完成。", "failed");
            } else {
              setStatus(chatStatus, "回复已同步。", "completed");
            }
            setSending(false);
            return;
          }
          setStatus(chatStatus, "连接中断，仍在同步已保存的消息。", "pending");
        } catch {
          if (!controller.signal.aborted && sessionIsCurrent(expectedGeneration, expectedUserId)) {
            setStatus(chatStatus, "连接中断，仍在同步已保存的消息。", "failed");
          }
        }
        delay = Math.min(delay * 2, 3000);
      }
    } finally {
      if (reconciliationController === controller) reconciliationController = null;
    }
  }

  function ensureTemporaryAssistant(message = {}) {
    if (!temporaryAssistant) {
      temporaryAssistant = appendMessage({
        id: message.assistant_message_id,
        role: "assistant",
        status: "streaming",
        content: "",
      });
    }
    return temporaryAssistant;
  }

  function updateTemporaryAssistant(message) {
    const item = ensureTemporaryAssistant();
    applyMessage(item, message);
    placeMessage(item);
  }

  function removeTemporaryAssistant() {
    if (temporaryAssistant?.parentNode) {
      temporaryAssistant.parentNode.removeChild(temporaryAssistant);
    }
    temporaryAssistant = null;
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
      ensureTemporaryAssistant(event.data ?? {});
      setStatus(chatStatus, "正在生成回复…", "streaming");
    } else if (event.name === "user_message") {
      if (event.data && Number.isInteger(event.data.sequence_no)) {
        appendMessage({
          id: event.data.id,
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
      updateTemporaryAssistant({
        id: message.id,
        role: "assistant",
        sequence_no: message.sequence_no,
        status: message.status,
        content: message.content,
      });
      setStatus(chatStatus, "回复已完成。", "completed");
      temporaryAssistant = null;
      return "done";
    } else if (event.name === "error") {
      removeTemporaryAssistant();
      setStatus(chatStatus, "回复未能完成，正在同步已保存的消息。", "failed");
      return "reconcile";
    }
    return false;
  }

  async function rejectedRequestMessage(response) {
    try {
      const payload = await response.json();
      const code = payload?.error?.code;
      if (code === "turn_in_progress") {
        return "当前用户已有消息正在生成，请等待完成后再试。";
      }
      if (code === "invalid_identifier_or_message") {
        return "消息格式无效，请检查后重试。";
      }
      if (code === "user_not_found") {
        return "当前用户不可用，请切换用户后重试。";
      }
    } catch {
      // The server rejected the request but did not provide a parseable public error.
    }
    return "暂时无法发送消息，请稍后再试。";
  }

  async function sendMessage(content) {
    const session = activeSession;
    if (!session || !content.trim()) return;
    const expectedGeneration = generation;
    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    streamController = controller;
    setSending(true);
    setStatus(chatStatus, "正在发送消息…", "pending");
    let requestAccepted = false;
    let streamStarted = false;
    let rejectedBeforeStream = false;
    let terminal = false;
    try {
      const response = await fetch(`/api/users/${session.user_id}/messages:stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ request_id: requestId, content }),
        signal: controller.signal,
      });
      if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
      if (!response.ok) {
        const message = await rejectedRequestMessage(response);
        if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
        rejectedBeforeStream = true;
        setStatus(chatStatus, message, "failed");
        return;
      }
      requestAccepted = true;
      if (!response.body) throw new Error("stream unavailable");
      const reader = response.body.getReader();
      streamStarted = true;
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = consumeFrames(buffer, (event) => {
          terminal = handleStreamEvent(event) === "done" || terminal;
        });
      }
      buffer += decoder.decode();
      consumeFrames(buffer, (event) => {
        terminal = handleStreamEvent(event) === "done" || terminal;
      });
    } catch {
      if (sendIsCurrent(controller, expectedGeneration, session.user_id)) {
        setStatus(chatStatus, "连接中断，正在同步已保存的消息。", "failed");
      }
    } finally {
      if (streamController === controller) streamController = null;
      if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
      if (terminal) {
        setSending(false);
      } else if (rejectedBeforeStream) {
        removeTemporaryAssistant();
        setSending(false);
      } else if (!controller.signal.aborted && (requestAccepted || !streamStarted)) {
        void reconcileHistory(expectedGeneration, session.user_id, requestId);
      }
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
      try {
        const messages = await loadHistory(expectedGeneration, controller.signal);
        if (sessionIsCurrent(expectedGeneration, activeSession.user_id)) {
          if (!resumeActiveHistory(messages ?? [], expectedGeneration, activeSession.user_id)) {
            setSending(false);
            setStatus(chatStatus, "可以开始聊天。", "completed");
          }
        }
      } catch {
        if (sessionIsCurrent(expectedGeneration, activeSession.user_id)) {
          setStatus(chatStatus, "历史加载失败，请切换用户后重试。", "failed");
          setSending(true);
        }
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
    resetMessageInput();
    void sendMessage(content);
  });

  messageInput.addEventListener("input", resizeMessageInput);

  messageInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    if (!messageInput.disabled) messageForm.requestSubmit();
  });

  switchUserButton.addEventListener("click", () => {
    generation += 1;
    streamController?.abort();
    identityController?.abort();
    reconciliationController?.abort();
    streamController = null;
    identityController = null;
    reconciliationController = null;
    activeSession = null;
    clearChat();
    setStatus(identityStatus, "");
    showIdentityView();
  });

  window.addEventListener("pagehide", () => {
    generation += 1;
    streamController?.abort();
    identityController?.abort();
    reconciliationController?.abort();
    streamController = null;
    identityController = null;
    reconciliationController = null;
  });

  window.addEventListener("pageshow", () => {
    const session = activeSession;
    if (!session) return;
    streamController?.abort();
    identityController?.abort();
    reconciliationController?.abort();
    streamController = null;
    identityController = null;
    reconciliationController = null;
    const expectedGeneration = ++generation;
    setSending(true);
    setStatus(chatStatus, "正在恢复消息状态…", "pending");
    void loadHistory(expectedGeneration)
      .then((messages) => {
        if (!sessionIsCurrent(expectedGeneration, session.user_id)) return;
        if (!resumeActiveHistory(messages ?? [], expectedGeneration, session.user_id)) {
          setSending(false);
          setStatus(chatStatus, "消息状态已恢复。", "completed");
        }
      })
      .catch(() => {
        if (!sessionIsCurrent(expectedGeneration, session.user_id)) return;
        setStatus(chatStatus, "历史加载失败，请切换用户后重试。", "failed");
        setSending(true);
      });
  });
  showIdentityView();
})();
