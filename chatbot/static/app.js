(() => {
  "use strict";

  const presentation = window.ChatPresentation;

  const identityView = document.querySelector("#identity-view");
  const identityForm = document.querySelector("#identity-form");
  const identifierInput = document.querySelector("#identifier-input");
  const identityStatus = document.querySelector("#identity-status");
  const chatView = document.querySelector("#chat-view");
  const currentIdentifier = document.querySelector("#current-identifier");
  const switchUserButton = document.querySelector("#switch-user");
  const conversationStage = document.querySelector("#conversation-stage");
  const messageList = document.querySelector("#message-list");
  const chatStatus = document.querySelector("#chat-status");
  const messageForm = document.querySelector("#message-form");
  const messageInput = document.querySelector("#message-input");
  const sendMessageButton = document.querySelector("#send-message");

  const feedbackControllers = new Set();
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
    presentation.updateBadge(null);
    temporaryAssistant = null;
    resetMessageInput();
    setStatus(chatStatus, "");
    setSending(false);
  }

  function applyMessage(item, message) {
    const state = typeof message.status === "string" ? message.status : "completed";
    const body = item.querySelector(".message-bubbles");
    const detail = item.querySelector("small");

    item.dataset.state = state;
    if (typeof message.id === "string") item.dataset.messageId = message.id;
    if (Number.isInteger(message.sequence_no)) item.dataset.sequence = String(message.sequence_no);
    const parts = Array.isArray(message.bubbles) ? message.bubbles : [message.content || ""];
    item.replyBubbles = Array.isArray(message.bubbles) ? [...message.bubbles] : [];
    body.replaceChildren(...parts.map(createBubble));
    detail.textContent = state === "failed" ? "生成未完成" : "";
    presentation.render(item, message.processing, state);
    renderReplyFeedback(item, message);
  }

  function renderReplyFeedback(item, message) {
    const previous = item.querySelector(".reply-feedback");
    if (previous) previous.parentNode.removeChild(previous);
    if (message.role !== "assistant" || message.status !== "completed" || !message.id || !activeSession) return;
    const controls = document.createElement("div");
    controls.className = "reply-feedback";
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", "评价这条回复");
    const status = document.createElement("span");
    status.className = "reply-feedback-status";
    status.setAttribute("role", "status");
    let rating = message.feedback;
    let pending = false;
    const buttons = [];
    const expectedGeneration = generation;
    const userId = activeSession.user_id;
    const isCurrent = () => sessionIsCurrent(expectedGeneration, userId) && item.parentNode === messageList;
    const update = () => {
      buttons.forEach((button) => {
        button.disabled = pending || Boolean(rating);
        button.hidden = Boolean(rating);
        button.setAttribute("aria-pressed", String(button.dataset.rating === rating));
      });
      status.textContent = pending ? "保存中…" : rating === "like" ? "已赞" : rating === "dislike" ? "已踩" : "";
    };
    for (const value of ["like", "dislike"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "reply-feedback-button";
      button.dataset.rating = value;
      button.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M7 10v11H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3Zm0 0 5-7a3 3 0 0 1 3 3l-1 4h5a3 3 0 0 1 2.9 3.7l-1.4 5A3 3 0 0 1 17.6 21H7" />
      </svg>`;
      button.setAttribute("aria-label", value === "like" ? "点赞这条回复" : "点踩这条回复");
      button.addEventListener("click", async () => {
        if (pending || rating || !isCurrent()) return;
        const controller = new AbortController();
        feedbackControllers.add(controller);
        pending = true;
        update();
        try {
          const response = await fetch(`/api/users/${userId}/messages/${encodeURIComponent(message.id)}/feedback`, {
            method: "PATCH", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({feedback: value}), signal: controller.signal,
          });
          const payload = await response.json();
          if (!isCurrent()) return;
          if (response.ok && payload.message_id === message.id && ["like", "dislike"].includes(payload.feedback)) {
            rating = payload.feedback;
          } else if (response.status === 409 && payload.error?.code === "already_rated") {
            // Another tab or a lost response may have saved the first rating.
            const history = await fetch(`/api/users/${userId}/messages`, {signal: controller.signal});
            if (!history.ok) throw new Error("history unavailable");
            const saved = (await history.json()).messages?.find((row) => row.id === message.id);
            if (!saved || !["like", "dislike"].includes(saved.feedback)) throw new Error("rating unavailable");
            rating = saved.feedback;
          } else {
            throw new Error("feedback rejected");
          }
          pending = false;
          if (isCurrent()) update();
        } catch {
          pending = false;
          if (isCurrent() && !controller.signal.aborted) {
            update();
            status.textContent = "评价保存失败，请重试。";
          }
        } finally {
          feedbackControllers.delete(controller);
        }
      });
      buttons.push(button);
      controls.append(button);
    }
    controls.append(status);
    update();
    item.querySelector("article").append(controls);
  }

  function abortFeedback() {
    feedbackControllers.forEach((controller) => controller.abort());
    feedbackControllers.clear();
  }

  function createBubble(content) {
    const body = document.createElement("p");
    body.className = "message-body";
    body.textContent = typeof content === "string" ? content : "";
    return body;
  }

  function createMessage(message) {
    const item = document.createElement("li");
    const article = document.createElement("article");
    const body = document.createElement("div");
    const detail = document.createElement("small");
    const role = message.role === "user" ? "user" : "assistant";

    item.className = `message message--${role}`;
    article.setAttribute("aria-label", role === "user" ? "用户消息" : "助手消息");
    body.className = "message-bubbles";
    detail.className = "message-error";
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

  function followConversation(update, force = false) {
    const follow = force || conversationStage.scrollHeight - conversationStage.scrollTop - conversationStage.clientHeight < 100;
    update();
    if (follow) conversationStage.scrollTop = conversationStage.scrollHeight;
  }

  function appendMessage(message) {
    const item = createMessage(message);
    followConversation(() => placeMessage(item), true);
    return item;
  }

  function renderHistory(messages) {
    const publicMessages = Array.isArray(messages)
      ? messages
          .filter((message) => message && Number.isInteger(message.sequence_no))
          .sort((left, right) => left.sequence_no - right.sequence_no)
      : [];
    followConversation(() => messageList.replaceChildren(...publicMessages.map(createMessage)), messageList.children.length === 0);
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
    presentation.updateBadge(payload.latest_emotion);
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
        bubbles: Number.isInteger(message.bubble_gap_ms) ? [] : null,
      });
      temporaryAssistant.dataset.bubbleMode = String(Number.isInteger(message.bubble_gap_ms));
    }
    return temporaryAssistant;
  }

  function updateTemporaryAssistant(message) {
    const item = ensureTemporaryAssistant();
    followConversation(() => {
      applyMessage(item, message);
      placeMessage(item);
    });
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
      setStatus(chatStatus, "正在处理你的消息…", "streaming");
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
    } else if (event.name === "progress" && event.data?.processing) {
      const item = ensureTemporaryAssistant(event.data);
      followConversation(() => presentation.render(item, event.data.processing, "streaming"));
      presentation.updateBadge(event.data.latest_emotion);
      setStatus(chatStatus, presentation.headline(event.data.processing, "streaming"), "streaming");
    } else if (event.name === "token" && event.data && typeof event.data.content === "string") {
      const item = ensureTemporaryAssistant();
      if (item.dataset.bubbleMode === "true") return false;
      const body = item.querySelector("p");
      followConversation(() => { body.textContent += event.data.content; });
      setStatus(chatStatus, "正在生成回复…", "streaming");
    } else if (event.name === "bubble" && event.data && typeof event.data.content === "string") {
      const item = ensureTemporaryAssistant(event.data);
      if (item.dataset.messageId !== event.data.assistant_message_id ||
          event.data.index !== item.replyBubbles.length || !event.data.content.trim()) return false;
      item.replyBubbles.push(event.data.content);
      followConversation(() => item.querySelector(".message-bubbles").append(createBubble(event.data.content)));
      setStatus(chatStatus, "正在回复…", "streaming");
    } else if (event.name === "done" && event.data && event.data.message) {
      const message = event.data.message;
      updateTemporaryAssistant({
        id: message.id,
        role: "assistant",
        sequence_no: message.sequence_no,
        status: message.status,
        content: message.content,
        bubbles: message.bubbles,
        feedback: message.feedback,
        processing: message.processing,
      });
      presentation.updateBadge(event.data.latest_emotion);
      setStatus(chatStatus, "回复已完成。", "completed");
      temporaryAssistant = null;
      return "done";
    } else if (event.name === "error") {
      if (temporaryAssistant) {
        temporaryAssistant.dataset.state = "failed";
        temporaryAssistant.querySelector("small").textContent = "生成未完成";
      }
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
    let eventChain = Promise.resolve();
    let bubbleGapMs = 0;
    let lastBubbleAt = null;
    const queueEvent = (event) => {
      eventChain = eventChain.then(async () => {
        if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
        if (event.name === "run_started") {
          bubbleGapMs = Number.isInteger(event.data?.bubble_gap_ms)
            ? Math.max(0, Math.min(10000, event.data.bubble_gap_ms)) : 0;
        }
        const isNewBubble = event.name === "bubble" && temporaryAssistant &&
          event.data?.assistant_message_id === temporaryAssistant.dataset.messageId &&
          event.data?.index === temporaryAssistant.replyBubbles.length;
        if (isNewBubble && lastBubbleAt !== null) {
          const remaining = bubbleGapMs - (Date.now() - lastBubbleAt);
          if (remaining > 0 && !await waitForDelay(remaining, controller.signal)) return;
        }
        if (!sendIsCurrent(controller, expectedGeneration, session.user_id)) return;
        terminal = handleStreamEvent(event) === "done" || terminal;
        if (isNewBubble) lastBubbleAt = Date.now();
      });
    };
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
        buffer = consumeFrames(buffer, queueEvent);
      }
      buffer += decoder.decode();
      consumeFrames(buffer, queueEvent);
    } catch {
      if (sendIsCurrent(controller, expectedGeneration, session.user_id)) {
        setStatus(chatStatus, "连接中断，正在同步已保存的消息。", "failed");
      }
    } finally {
      // Network reads remain independent from display pacing. Drain complete
      // bubbles before finalizing or polling; abort also cancels pending delays.
      try { await eventChain; } catch { terminal = false; }
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
      setSending(true);
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
    // IME confirmation can report isComposing=false after compositionend.
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    if (!messageInput.disabled) messageForm.requestSubmit();
  });

  switchUserButton.addEventListener("click", () => {
    generation += 1;
    abortFeedback();
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
    abortFeedback();
    streamController?.abort();
    identityController?.abort();
    reconciliationController?.abort();
    streamController = null;
    identityController = null;
    reconciliationController = null;
  });

  window.addEventListener("pageshow", () => {
    abortFeedback();
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
