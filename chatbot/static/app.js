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

function addMessage(role, content = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=10");
  if (!response.ok) {
    emotionStatusEl.textContent = "历史加载失败";
    return;
  }
  const payload = await response.json();
  messagesEl.innerHTML = "";
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    addMessage(role, message.content);
  });
}

async function initialize() {
  setLocked(true);
  try {
    await loadHistory();
  } finally {
    setLocked(false);
  }
}

function streamMessage(message) {
  setLocked(true);
  let aiBubble = null;
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
    emotionStatusEl.textContent = `情感状态：${payload.emotion}`;
  });

  source.addEventListener("emotion_error", () => {
    emotionStatusEl.textContent = "情感状态：情感分析失败，本轮继续回复";
  });

  source.addEventListener("token", (event) => {
    const payload = JSON.parse(event.data);
    if (!aiBubble) {
      aiBubble = addMessage("ai", "");
    }
    aiBubble.textContent += payload.content;
    scrollToBottom();
  });

  source.addEventListener("done", () => {
    source.close();
    setLocked(false);
    inputEl.focus();
  });

  source.addEventListener("error", (event) => {
    if (event.data) {
      const payload = JSON.parse(event.data);
      if (!aiBubble) {
        aiBubble = addMessage("ai", "");
      }
      aiBubble.textContent = payload.message;
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
