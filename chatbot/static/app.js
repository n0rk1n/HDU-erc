const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const buttonEl = document.querySelector("#send-button");
const emotionStatusEl = document.querySelector("#emotion-status");

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

  const status = document.createElement("span");
  status.className = "feedback-status";

  const buttons = [likeButton, dislikeButton];
  likeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "like", controls, status, buttons)
  ));
  dislikeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "dislike", controls, status, buttons)
  ));

  controls.appendChild(likeButton);
  controls.appendChild(dislikeButton);
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

function addMessage(role, content = "", metadata = {}) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(bubble);
  if (role === "ai") {
    renderFeedbackControls(wrapper, metadata);
  }
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return {wrapper, bubble};
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
