const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const buttonEl = document.querySelector("#send-button");
const emotionStatusEl = document.querySelector("#emotion-status");
const safetyStatusEl = document.querySelector("#safety-status");
const emotionTimelineEl = document.querySelector("#emotion-timeline");
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

function nextSiblingOf(wrapper) {
  if ("nextSibling" in wrapper) {
    return wrapper.nextSibling;
  }
  if (!wrapper.parent) {
    return null;
  }
  const siblings = Array.from(wrapper.parent.children || []);
  const index = siblings.indexOf(wrapper);
  return siblings[index + 1] || null;
}

function insertMessageAfter(referenceWrapper, role, content, metadata = {}) {
  const inserted = createMessageElement(role, content, metadata);
  messagesEl.insertBefore(inserted.wrapper, nextSiblingOf(referenceWrapper));
  scrollToBottom();
  return inserted;
}

function allControlButtons(controls) {
  const buttons = [];
  const visit = (element) => {
    Array.from(element.children || []).forEach((child) => {
      const tagName = child.tagName ? child.tagName.toLowerCase() : "";
      if (tagName === "button" || child.name === "button") {
        buttons.push(child);
      }
      visit(child);
    });
  };
  visit(controls);
  return buttons;
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
    submitFeedback(metadata.id, "like", controls, status)
  ));
  dislikeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "dislike", controls, status)
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

async function submitFeedback(messageId, feedback, controls, status) {
  const pendingButtons = allControlButtons(controls);
  pendingButtons.forEach((button) => {
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
    pendingButtons.forEach((button) => {
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
  regenerationReasons.forEach((reason) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "feedback-button regeneration-reason";
    button.textContent = reason;
    button.addEventListener("click", () => (
      submitRegeneration(wrapper, metadata.id, reason, controls, status)
    ));
    reasons.appendChild(button);
  });
  controls.insertBefore(reasons, status);
}

async function submitRegeneration(wrapper, messageId, reason, controls, status) {
  const pendingButtons = allControlButtons(controls);
  pendingButtons.forEach((button) => {
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
    pendingButtons.forEach((button) => {
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

function clearSafetyStatus() {
  if (!safetyStatusEl) {
    return;
  }
  safetyStatusEl.hidden = true;
  safetyStatusEl.textContent = "";
}

function renderEmotion(emotion) {
  if (emotion && emotion.emotion) {
    emotionStatusEl.textContent = `情感状态：${emotion.emotion}`;
    clearSafetyStatus();
    return;
  }
  emotionStatusEl.textContent = "情感状态：暂无";
  clearSafetyStatus();
}

function renderEmotionState(state) {
  if (!state || !state.primary_emotion) {
    emotionStatusEl.textContent = "情感状态：暂无";
    clearSafetyStatus();
    return;
  }
  const confidence = typeof state.confidence === "number" ? ` ${(state.confidence * 100).toFixed(0)}%` : "";
  emotionStatusEl.textContent = `情感状态：${state.primary_emotion}${confidence}`;
  if (safetyStatusEl && state.safety_level && state.safety_level !== "normal") {
    safetyStatusEl.hidden = false;
    safetyStatusEl.textContent = `安全提示：${state.safety_level}`;
  } else if (safetyStatusEl) {
    clearSafetyStatus();
  }
}

function renderTimeline(timeline) {
  if (!emotionTimelineEl) {
    return;
  }
  emotionTimelineEl.innerHTML = "";
  timeline.slice(-5).forEach((item) => {
    const row = document.createElement("li");
    row.textContent = item.trajectory_note || `${item.turn_count}: ${item.primary_emotion}`;
    emotionTimelineEl.appendChild(row);
  });
}

async function loadEmotionTimeline() {
  if (!emotionTimelineEl) {
    return;
  }
  const response = await fetch("/api/emotion/timeline?limit=5");
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  renderTimeline(payload.timeline || []);
}

async function loadSession() {
  const response = await fetch("/api/session?limit=10");
  if (!response.ok) {
    emotionStatusEl.textContent = "历史加载失败";
    return;
  }
  const payload = await response.json();
  messagesEl.innerHTML = "";
  const renderedById = {};
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    const rendered = createMessageElement(role, message.content, message);
    const original = message.regenerated_from ? renderedById[message.regenerated_from] : null;
    if (original) {
      messagesEl.insertBefore(rendered.wrapper, nextSiblingOf(original.wrapper));
    } else {
      messagesEl.appendChild(rendered.wrapper);
    }
    if (message.id) {
      renderedById[message.id] = rendered;
    }
  });
  scrollToBottom();
  renderEmotion(payload.emotion);
  await loadEmotionTimeline();
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
    clearSafetyStatus();
  });

  source.addEventListener("emotion_done", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.state) {
      renderEmotionState(payload.state);
    } else {
      renderEmotion(payload);
    }
    loadEmotionTimeline();
  });

  source.addEventListener("emotion_error", () => {
    emotionStatusEl.textContent = "情感状态：情感分析失败，本轮继续回复";
    clearSafetyStatus();
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
