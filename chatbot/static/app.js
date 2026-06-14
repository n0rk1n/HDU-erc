const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const buttonEl = document.querySelector("#send-button");
const emotionStatusEl = document.querySelector("#emotion-status");
const regenerationReasons = ["不准确", "不完整", "没有理解我的问题", "语气不合适", "其他"];

function setLocked(locked) {
  inputEl.disabled = locked;
  buttonEl.disabled = locked;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function shouldShowFeedback(metadata) {
  return Boolean(metadata.id) && !metadata.feedback;
}

function collapseRegeneratedMessage(wrapper, reason) {
  wrapper.className = `${wrapper.className} regenerated`;
  wrapper.setAttribute("data-regeneration-reason", reason);
  const summary = document.createElement("div");
  summary.className = "regeneration-summary";
  summary.textContent = `已重新生成：${reason}`;
  wrapper.appendChild(summary);
}

function insertMessageAfter(referenceWrapper, role, content, metadata = {}) {
  const inserted = createMessageElement(role, content, metadata);
  messagesEl.insertBefore(inserted.wrapper, referenceWrapper.nextSibling);
  scrollToBottom();
  return inserted;
}

function renderFeedbackControls(wrapper, metadata) {
  if (!shouldShowFeedback(metadata)) {
    return;
  }

  const controls = document.createElement("div");
  controls.className = "feedback-controls";

  const likeButton = document.createElement("button");
  likeButton.type = "button";
  likeButton.className = "feedback-button";
  likeButton.textContent = "Good";
  likeButton.setAttribute("aria-label", "Good");

  const dislikeButton = document.createElement("button");
  dislikeButton.type = "button";
  dislikeButton.className = "feedback-button";
  dislikeButton.textContent = "Bad";
  dislikeButton.setAttribute("aria-label", "Bad");

  const regenerateButton = document.createElement("button");
  regenerateButton.type = "button";
  regenerateButton.className = "feedback-button";
  regenerateButton.textContent = "Regenerate";
  regenerateButton.setAttribute("aria-label", "Regenerate");

  const status = document.createElement("span");
  status.className = "feedback-status";

  const buttons = [likeButton, dislikeButton, regenerateButton];
  likeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "like", controls, status, buttons)
  ));
  dislikeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "dislike", controls, status, buttons)
  ));
  regenerateButton.addEventListener("click", () => (
    renderRegenerationReasons(wrapper, metadata, controls, status, buttons)
  ));

  controls.appendChild(likeButton);
  controls.appendChild(dislikeButton);
  controls.appendChild(regenerateButton);
  controls.appendChild(status);
  wrapper.appendChild(controls);
}

async function submitFeedback(messageId, feedback, controls, status, buttons = []) {
  buttons.forEach((button) => {
    button.disabled = true;
  });
  status.textContent = "";

  try {
    const response = await fetch(`/api/messages/${encodeURIComponent(messageId)}/feedback`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({feedback}),
    });
    if (!response.ok) {
      throw new Error("Feedback request failed.");
    }
    controls.remove();
  } catch (error) {
    buttons.forEach((button) => {
      button.disabled = false;
    });
    status.textContent = "评价保存失败";
  }
}

function findChildByClass(parent, className) {
  return Array.from(parent.children).find((child) => (
    child.className && child.className.split(" ").includes(className)
  ));
}

function renderRegenerationReasons(wrapper, metadata, controls, status, buttons) {
  let reasons = findChildByClass(controls, "regeneration-reasons");
  if (reasons) {
    reasons.remove();
    return;
  }

  reasons = document.createElement("div");
  reasons.className = "regeneration-reasons";
  const reasonButtons = [];
  regenerationReasons.forEach((reason) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "feedback-button regeneration-reason";
    button.textContent = reason;
    button.addEventListener("click", () => (
      submitRegeneration(
        wrapper,
        metadata.id,
        reason,
        controls,
        status,
        [...buttons, ...reasonButtons],
      )
    ));
    reasonButtons.push(button);
    reasons.appendChild(button);
  });
  controls.insertBefore(reasons, status);
}

async function submitRegeneration(wrapper, messageId, reason, controls, status, buttons = []) {
  buttons.forEach((button) => {
    button.disabled = true;
  });
  status.textContent = "";

  try {
    const response = await fetch(`/api/messages/${encodeURIComponent(messageId)}/regenerate`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({reason}),
    });
    if (!response.ok) {
      throw new Error("Regeneration request failed.");
    }
    const payload = await response.json();
    controls.remove();
    collapseRegeneratedMessage(wrapper, payload.reason);
    insertMessageAfter(wrapper, "ai", payload.content, {
      id: payload.message_id,
      feedback: null,
      regenerated_from: payload.original_message_id,
    });
  } catch (error) {
    buttons.forEach((button) => {
      button.disabled = false;
    });
    status.textContent = "重新生成失败";
  }
}

function createMessageElement(role, content = "", metadata = {}) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(bubble);
  if (role === "ai") {
    if (metadata.regeneration) {
      collapseRegeneratedMessage(wrapper, metadata.regeneration.reason);
    } else {
      renderFeedbackControls(wrapper, metadata);
    }
  }
  return {wrapper, bubble};
}

function addMessage(role, content = "", metadata = {}) {
  const message = createMessageElement(role, content, metadata);
  messagesEl.appendChild(message.wrapper);
  scrollToBottom();
  return message;
}

function renderEmotion(emotion) {
  if (emotion && emotion.emotion) {
    emotionStatusEl.textContent = `情感状态：${emotion.emotion}`;
    return;
  }
  emotionStatusEl.textContent = "情感状态：暂无";
}

async function loadSession() {
  const response = await fetch("/api/session?limit=10");
  if (!response.ok) {
    emotionStatusEl.textContent = "历史加载失败";
    return;
  }
  const payload = await response.json();
  messagesEl.innerHTML = "";
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    addMessage(role, message.content, message);
  });
  renderEmotion(payload.emotion);
}

async function initialize() {
  setLocked(true);
  try {
    await loadSession();
  } finally {
    setLocked(false);
  }
}

function streamMessage(message) {
  setLocked(true);
  let aiMessage = null;
  const source = new EventSource(`/api/chat/stream?message=${encodeURIComponent(message)}`);

  source.addEventListener("user_message", (event) => {
    const payload = JSON.parse(event.data);
    addMessage("human", payload.content);
  });

  source.addEventListener("emotion_start", () => {
    emotionStatusEl.textContent = "情感状态：正在分析情绪…";
  });

  source.addEventListener("emotion_done", (event) => {
    const payload = JSON.parse(event.data);
    renderEmotion(payload);
  });

  source.addEventListener("emotion_error", () => {
    emotionStatusEl.textContent = "情感状态：情感分析失败，本轮继续回复";
  });

  source.addEventListener("token", (event) => {
    const payload = JSON.parse(event.data);
    if (!aiMessage) {
      aiMessage = addMessage("ai", "");
    }
    aiMessage.bubble.textContent += payload.content;
    scrollToBottom();
  });

  source.addEventListener("done", (event) => {
    const payload = event.data ? JSON.parse(event.data) : {};
    if (aiMessage && payload.message_id) {
      renderFeedbackControls(aiMessage.wrapper, {id: payload.message_id, feedback: null});
    }
    source.close();
    setLocked(false);
    inputEl.focus();
  });

  source.addEventListener("error", (event) => {
    if (!aiMessage) {
      aiMessage = addMessage("ai", "");
    }
    if (event.data) {
      const payload = JSON.parse(event.data);
      aiMessage.bubble.textContent = payload.message;
    } else {
      aiMessage.bubble.textContent = "连接中断，请稍后重试";
    }
    source.close();
    setLocked(false);
  });
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) {
    return;
  }
  inputEl.value = "";
  streamMessage(message);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

initialize();
