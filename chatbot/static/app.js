const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButtonEl = document.querySelector("#send-button");
const emotionStatusEl = document.querySelector("#emotion-status");
const safetyStatusEl = document.querySelector("#safety-status");
const emotionTimelineEl = document.querySelector("#emotion-timeline");
const threadListEl = document.querySelector("#thread-list");
const newThreadButtonEl = document.querySelector("#new-thread-button");
const deleteThreadButtonEl = document.querySelector("#delete-thread-button");
const profileButtonEl = document.querySelector("#profile-button");
const profilePanelEl = document.querySelector("#profile-panel");
const profileBackdropEl = document.querySelector("#profile-backdrop");
const profileCloseEl = document.querySelector("#profile-close");
const profilePanelBodyEl = document.querySelector("#profile-panel-body");
const profilePromptEl = document.querySelector("#profile-onboarding-prompt");
const profilePromptStartEl = document.querySelector("#profile-onboarding-start");
const profilePromptSkipEl = document.querySelector("#profile-onboarding-skip");

const profileFields = [
  ["preferred_name", "称呼"], ["life_stage", "身份或阶段"],
  ["companion_expectation", "陪伴期待"], ["response_style", "回应风格"],
  ["avoidance", "希望避免"],
];
const regenerationReasons = ["不准确", "不完整", "没有理解我的问题", "语气不合适", "其他"];
const emotionFeedbackChoices = [
  ["准确", "accurate"], ["过于积极", "too_positive"],
  ["过于消极", "too_negative"], ["情绪判断不对", "wrong_emotion"],
];
const state = {
  clientId: null, threadId: null, threads: [], activeRun: null,
  navigationEpoch: 0, streamEpoch: 0, navigationBusy: false,
  authRecovery: null,
};
const profileState = {profile: {}, questions: [], answers: [], questionIndex: 0};

function clientPath(suffix = "", clientId = state.clientId) {
  return `/api/clients/${encodeURIComponent(clientId)}${suffix}`;
}

function threadPath(suffix = "", clientId = state.clientId, threadId = state.threadId) {
  return clientPath(`/threads/${encodeURIComponent(threadId)}${suffix}`, clientId);
}

function setLocked(locked) {
  if (inputEl) inputEl.disabled = locked;
  if (sendButtonEl) sendButtonEl.disabled = locked;
}

function updateInteractionLocks() {
  const streaming = Boolean(state.activeRun);
  setLocked(state.navigationBusy || streaming);
  if (newThreadButtonEl) newThreadButtonEl.disabled = state.navigationBusy || streaming;
  if (deleteThreadButtonEl) deleteThreadButtonEl.disabled = state.navigationBusy || streaming;
  if (threadListEl) {
    Array.from(threadListEl.children || []).forEach((item) => {
      Array.from(item.children || []).forEach((button) => { button.disabled = state.navigationBusy; });
    });
  }
}

function showUiError(message) {
  if (emotionStatusEl) emotionStatusEl.textContent = message;
}

function abortActiveStream() {
  state.streamEpoch += 1;
  const active = state.activeRun;
  if (active) active.controller.abort();
  if (state.activeRun === active) state.activeRun = null;
  updateInteractionLocks();
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = new Error(`Request failed: ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response;
}

async function fetchJson(url, options = {}) {
  const response = await request(url, options);
  return response.json();
}

async function bootstrapRaw() {
  const payload = await fetchJson("/api/clients/bootstrap", {method: "POST"});
  state.clientId = payload.client_id;
  state.threadId = payload.thread.thread_id;
  state.threads = [payload.thread];
  localStorage.setItem("hdu_erc_client_id", state.clientId);
  localStorage.setItem("hdu_erc_thread_id", state.threadId);
  return payload;
}

async function recoverIdentity() {
  if (state.authRecovery) return state.authRecovery;
  let recoveryEpoch = 0;
  state.authRecovery = (async () => {
    const epoch = ++state.navigationEpoch; recoveryEpoch = epoch;
    abortActiveStream();
    state.navigationBusy = true; updateInteractionLocks();
    localStorage.removeItem("hdu_erc_client_id");
    localStorage.removeItem("hdu_erc_thread_id");
    state.clientId = null; state.threadId = null; state.threads = [];
    const payload = await bootstrapRaw();
    renderThreads();
    const clientId = state.clientId; const threadId = state.threadId;
    const [snapshot, timeline] = await Promise.all([
      fetchJson(threadPath("", clientId, threadId)),
      fetchJson(threadPath("/emotion-timeline?limit=5", clientId, threadId)),
    ]);
    if (navigationIsCurrent(epoch, clientId, threadId)) {
      renderSnapshot(snapshot); renderTimeline(timeline.timeline || []);
    }
    if (epoch === state.navigationEpoch) { state.navigationBusy = false; updateInteractionLocks(); }
    return payload;
  })();
  try { return await state.authRecovery; } finally {
    state.authRecovery = null;
    if (state.navigationEpoch === recoveryEpoch) { state.navigationBusy = false; updateInteractionLocks(); }
  }
}

async function readJson(buildUrl, options = {}) {
  try { return await fetchJson(buildUrl(), options); }
  catch (error) {
    if (error.status !== 401) throw error;
    await recoverIdentity();
    return fetchJson(buildUrl(), options);
  }
}

async function handleMutationUnauthorized(error) {
  if (error.status !== 401) return false;
  await recoverIdentity();
  showUiError("身份已更新，请重试刚才的操作");
  return true;
}

async function mutateJson(url, options) {
  try { return await fetchJson(url, options); }
  catch (error) {
    await handleMutationUnauthorized(error);
    throw error;
  }
}

async function listThreads() {
  const payload = await readJson(() => clientPath("/threads"));
  state.threads = payload.threads || [];
  return state.threads;
}

function renderThreads() {
  if (!threadListEl) return;
  threadListEl.replaceChildren();
  state.threads.forEach((thread) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = thread.thread_id === state.threadId ? "thread-button active" : "thread-button";
    button.textContent = thread.title || "新对话";
    button.setAttribute("data-thread-id", thread.thread_id);
    button.setAttribute("aria-current", thread.thread_id === state.threadId ? "true" : "false");
    button.disabled = state.navigationBusy;
    button.addEventListener("click", () => {
      if (state.navigationBusy) return;
      selectThread(thread.thread_id).catch(() => showUiError("对话切换失败，请重试"));
    });
    item.appendChild(button);
    threadListEl.appendChild(item);
  });
}

async function initializeIdentity() {
  state.clientId = localStorage.getItem("hdu_erc_client_id");
  state.threadId = localStorage.getItem("hdu_erc_thread_id");
  if (!state.clientId) {
    await bootstrapRaw();
    return;
  }
  try {
    await listThreads();
  } catch (error) {
    if (error.status !== 401) throw error;
    await recoverIdentity();
    return;
  }
  if (!state.threads.length) {
    await createThread();
    return;
  }
  if (!state.threads.some((thread) => thread.thread_id === state.threadId)) {
    state.threadId = state.threads[0].thread_id;
    localStorage.setItem("hdu_erc_thread_id", state.threadId);
  }
}

async function createThread() {
  if (state.navigationBusy) return null;
  const previous = {threadId: state.threadId, threads: state.threads.slice()};
  const operationEpoch = ++state.navigationEpoch;
  abortActiveStream();
  state.navigationBusy = true; updateInteractionLocks();
  try {
    const payload = await fetchJson(clientPath("/threads"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title: "新对话"}),
    });
    state.threads = [...state.threads, payload.thread];
    if (state.navigationEpoch !== operationEpoch) { renderThreads(); return payload.thread; }
    state.navigationBusy = false;
    await selectThread(payload.thread.thread_id);
    return payload.thread;
  } catch (error) {
    if (state.navigationEpoch === operationEpoch) {
      state.threadId = previous.threadId; state.threads = previous.threads;
      if (previous.threadId) localStorage.setItem("hdu_erc_thread_id", previous.threadId);
      renderThreads();
    }
    if (!(await handleMutationUnauthorized(error))) showUiError("新建对话失败，请重试");
    throw error;
  } finally {
    if (state.navigationEpoch === operationEpoch) { state.navigationBusy = false; updateInteractionLocks(); }
  }
}

async function deleteCurrentThread() {
  if (!state.threadId || state.navigationBusy) return;
  const deletedThread = state.threadId;
  const operationEpoch = ++state.navigationEpoch;
  abortActiveStream();
  state.navigationBusy = true; updateInteractionLocks();
  try {
    await request(threadPath(), {method: "DELETE"});
    state.threads = state.threads.filter((item) => item.thread_id !== deletedThread);
    if (state.navigationEpoch === operationEpoch && state.threadId === deletedThread) {
      state.threadId = null;
      localStorage.removeItem("hdu_erc_thread_id");
    }
    try { await listThreads(); } catch (error) {
      showUiError("对话已删除，正在重新同步");
      throw error;
    }
    if (state.navigationEpoch !== operationEpoch) { renderThreads(); return; }
    state.navigationBusy = false;
    if (!state.threads.length) { await createThread(); return; }
    await selectThread(state.threads[0].thread_id);
  } catch (error) {
    if (state.navigationEpoch === operationEpoch && state.threadId === deletedThread) localStorage.setItem("hdu_erc_thread_id", deletedThread);
    if (!(await handleMutationUnauthorized(error))) showUiError(state.threadId ? "删除对话失败，请重试" : "对话已删除，请刷新以重新同步");
    throw error;
  } finally {
    if (state.navigationEpoch === operationEpoch) { state.navigationBusy = false; updateInteractionLocks(); }
  }
}

async function selectThread(threadId) {
  if (!threadId) return;
  const previousThreadId = state.threadId;
  const clientId = state.clientId;
  const epoch = ++state.navigationEpoch;
  abortActiveStream();
  state.navigationBusy = true;
  state.threadId = threadId;
  localStorage.setItem("hdu_erc_thread_id", threadId);
  renderSafety({level: "normal", guidance: ""});
  if (messagesEl) messagesEl.replaceChildren();
  renderThreads();
  updateInteractionLocks();
  try {
    await loadCurrentThread(epoch, clientId, threadId);
  } catch (error) {
    if (epoch === state.navigationEpoch && state.clientId === clientId && state.threadId === threadId) {
      state.threadId = previousThreadId;
      if (previousThreadId) localStorage.setItem("hdu_erc_thread_id", previousThreadId);
      else localStorage.removeItem("hdu_erc_thread_id");
      renderThreads(); showUiError("对话切换失败，请重试");
    }
    throw error;
  } finally {
    if (epoch === state.navigationEpoch) { state.navigationBusy = false; updateInteractionLocks(); }
  }
}

function messageText(content, role) {
  if (role === "system") return "[系统消息]";
  if (role === "tool") return "[工具消息]";
  if (role !== "human" && role !== "ai") return "[内部消息]";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = content.map((part) => {
      if (typeof part === "string") return part;
      if (part && part.type === "text" && typeof part.text === "string") return part.text;
      return "";
    }).join("");
    return text || (role === "ai" ? "[工具调用]" : "[复杂消息]");
  }
  if (content == null) return role === "ai" ? "[工具调用]" : "";
  return role === "ai" ? "[复杂回复]" : "[复杂消息]";
}

function scrollToBottom() {
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMessageElement(role, content, metadata = {}) {
  const wrapper = document.createElement("article");
  const displayRole = role === "human" ? "human" : role === "ai" ? "ai" : "system";
  wrapper.className = `message ${displayRole}`;
  if (metadata.id) wrapper.setAttribute("data-message-id", metadata.id);
  if (metadata.feedback) wrapper.setAttribute("data-feedback", metadata.feedback);
  if (metadata.regenerated) wrapper.setAttribute("data-regenerated", "true");
  if (metadata.regeneration_reason) wrapper.setAttribute("data-regeneration-reason", metadata.regeneration_reason);
  if (metadata.original_content != null) wrapper.setAttribute("data-original-content", messageText(metadata.original_content, role));
  if (metadata.predicted_emotion) wrapper.setAttribute("data-predicted-emotion", metadata.predicted_emotion);
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = messageText(content, role);
  wrapper.appendChild(bubble);
  if (displayRole === "ai" && metadata.id) renderFeedbackControls(wrapper, metadata);
  return {wrapper, bubble};
}

function addMessage(role, content, metadata = {}) {
  const message = createMessageElement(role, content, metadata);
  messagesEl.appendChild(message.wrapper);
  scrollToBottom();
  return message;
}

function renderSnapshot(payload) {
  messagesEl.replaceChildren();
  (payload.messages || []).forEach((message) => {
    addMessage(message.role || "system", message.content, message);
  });
  renderEmotionState(payload.emotion);
  const latestAi = (payload.messages || []).slice().reverse().find((message) => message.role === "ai");
  const persistedSafety = latestAi && ["normal", "supportive", "crisis"].includes(latestAi.safety_level)
    ? latestAi.safety_level : "normal";
  renderSafety({level: persistedSafety});
}

function renderEmotionState(emotion) {
  if (!emotionStatusEl) return;
  if (!emotion || !emotion.primary_emotion) {
    emotionStatusEl.textContent = "情感状态：暂无";
    return;
  }
  const confidence = typeof emotion.confidence === "number" ? ` ${(emotion.confidence * 100).toFixed(0)}%` : "";
  emotionStatusEl.textContent = `情感状态：${emotion.primary_emotion}${confidence}`;
}

function renderSafety(data) {
  if (!safetyStatusEl) return;
  if (!data || data.level === "normal") {
    safetyStatusEl.hidden = true; safetyStatusEl.textContent = ""; return;
  }
  safetyStatusEl.hidden = false;
  safetyStatusEl.className = `safety-status ${data.level}`;
  safetyStatusEl.textContent = data.level === "crisis"
    ? "如果你正面临即时危险，请立即联系当地紧急服务或身边可信任的人。"
    : "检测到你可能需要更多支持，我会谨慎回应。";
}

function renderTimeline(timeline) {
  if (!emotionTimelineEl) return;
  emotionTimelineEl.replaceChildren();
  (timeline || []).slice(-5).forEach((item) => {
    const row = document.createElement("li");
    row.textContent = item.trajectory_note || `${item.turn_count || ""}: ${item.primary_emotion || ""}`;
    emotionTimelineEl.appendChild(row);
  });
}

function navigationIsCurrent(epoch, clientId, threadId) {
  return epoch === state.navigationEpoch && clientId === state.clientId && threadId === state.threadId;
}

async function loadEmotionTimeline(epoch = state.navigationEpoch, clientId = state.clientId, threadId = state.threadId) {
  const payload = await readJson(() => threadPath("/emotion-timeline?limit=5", state.clientId, state.threadId));
  if (!navigationIsCurrent(epoch, clientId, threadId)) return false;
  renderTimeline(payload.timeline || []);
  return true;
}

async function loadCurrentThread(epoch = state.navigationEpoch, clientId = state.clientId, threadId = state.threadId) {
  if (!clientId || !threadId) return false;
  const payload = await readJson(() => threadPath("", state.clientId, state.threadId));
  if (!navigationIsCurrent(epoch, clientId, threadId)) return false;
  renderSnapshot(payload);
  await loadEmotionTimeline(epoch, clientId, threadId);
  return navigationIsCurrent(epoch, clientId, threadId);
}

function parseSseFrame(frame) {
  let event = "message";
  const data = [];
  frame.split(/\r?\n/).forEach((line) => {
    if (!line || line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  });
  if (!data.length) return null;
  return {event, data: JSON.parse(data.join("\n"))};
}

async function consumeSseChunks(chunks, onFrame) {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += decoder.decode(chunk, {stream: true});
    let match;
    while ((match = /\r?\n\r?\n/.exec(buffer)) !== null) {
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const parsed = parseSseFrame(frame);
      if (parsed) await onFrame(parsed);
    }
  }
  buffer += decoder.decode();
  const parsed = parseSseFrame(buffer);
  if (parsed) await onFrame(parsed);
}

async function collectSseFrames(chunks) {
  const frames = [];
  await consumeSseChunks(chunks, (frame) => frames.push(frame));
  return frames;
}

async function consumeSseResponse(response, onFrame) {
  if (!response.body || !response.body.getReader) throw new Error("Streaming response unavailable");
  const reader = response.body.getReader();
  async function* chunks() {
    try {
      while (true) {
        const {value, done} = await reader.read();
        if (done) return;
        yield value;
      }
    } finally {
      if (reader.releaseLock) reader.releaseLock();
    }
  }
  await consumeSseChunks(chunks(), onFrame);
}

async function runStream(url, body, handlers) {
  abortActiveStream();
  const controller = new AbortController();
  const run = {
    controller, epoch: ++state.streamEpoch, clientId: state.clientId,
    threadId: state.threadId, requestId: body.request_id,
  };
  state.activeRun = run;
  updateInteractionLocks();
  let completed = false;
  let started = false;
  try {
    let response;
    try {
      response = await request(url, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body), signal: controller.signal,
      });
    } catch (error) {
      if (error.status === 401) {
        await recoverIdentity();
        showUiError("身份已更新，请重新发送");
      }
      throw error;
    }
    await consumeSseResponse(response, async ({event, data}) => {
      if (state.activeRun !== run || run.epoch !== state.streamEpoch || run.clientId !== state.clientId || run.threadId !== state.threadId) return;
      if (event === "run_started") {
        if (data.request_id !== run.requestId || data.thread_id !== run.threadId) throw new Error("Mismatched stream identity");
        started = true; return;
      }
      if (!started) throw new Error("Stream started without run identity");
      if (event === "done") completed = true;
      if (handlers[event]) await handlers[event](data);
    });
    if (state.activeRun !== run) {
      const superseded = new Error("Stream superseded"); superseded.name = "AbortError"; throw superseded;
    }
    if (!started || !completed) throw new Error("Stream ended before done");
    return true;
  } finally {
    if (state.activeRun === run) {
      state.activeRun = null;
      updateInteractionLocks();
      if (inputEl) inputEl.focus();
    }
  }
}

async function streamMessage(message) {
  const streamClient = state.clientId;
  const streamThread = state.threadId;
  const navigationEpoch = state.navigationEpoch;
  const optimistic = addMessage("human", message);
  let aiMessage = null;
  try {
    await runStream(threadPath("/messages:stream"), {message, request_id: crypto.randomUUID()}, {
      user_message(data) {
        optimistic.wrapper.setAttribute("data-message-id", data.message_id);
      },
      emotion_start() { emotionStatusEl.textContent = "情感状态：正在分析情绪…"; },
      emotion_done(data) { renderEmotionState(data.state); },
      emotion_error() { emotionStatusEl.textContent = "情感状态：情感分析失败，本轮继续回复"; },
      safety(data) { renderSafety(data); },
      token(data) {
        if (!aiMessage) aiMessage = addMessage("ai", "");
        aiMessage.bubble.textContent += data.content;
        scrollToBottom();
      },
      error(data) {
        throw new Error(data.error_code || "stream_failed");
      },
      done() {},
    });
    if (streamClient === state.clientId && streamThread === state.threadId) {
      await loadCurrentThread(navigationEpoch, streamClient, streamThread);
    }
  } catch (error) {
    if (error.name !== "AbortError" && streamClient === state.clientId && streamThread === state.threadId) {
      if (!aiMessage) aiMessage = addMessage("ai", "");
      aiMessage.bubble.textContent = "发送失败，请稍后重试";
    }
  }
}

function allButtons(container) {
  const output = [];
  const visit = (node) => (node.children || []).forEach((child) => {
    if ((child.name || child.tagName || "").toString().toLowerCase() === "button") output.push(child);
    visit(child);
  });
  visit(container);
  return output;
}

function renderFeedbackControls(wrapper, metadata) {
  if (metadata.feedback) return;
  const controls = document.createElement("div"); controls.className = "feedback-controls";
  const status = document.createElement("span"); status.className = "feedback-status";
  [["有帮助", "like", "将回复评价为有帮助"], ["没帮助", "dislike", "将回复评价为没帮助"]].forEach(([label, value, ariaLabel]) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "feedback-button"; button.textContent = label;
    button.setAttribute("aria-label", ariaLabel);
    button.addEventListener("click", () => submitFeedback(metadata.id, value, controls, status)); controls.appendChild(button);
  });
  const regenerate = document.createElement("button"); regenerate.type = "button"; regenerate.className = "feedback-button"; regenerate.textContent = "重新生成"; regenerate.setAttribute("aria-label", "重新生成回复");
  regenerate.addEventListener("click", () => renderRegenerationReasons(wrapper, metadata.id, controls, status)); controls.appendChild(regenerate);
  const emotion = document.createElement("button"); emotion.type = "button"; emotion.className = "feedback-button"; emotion.textContent = "情绪判断"; emotion.setAttribute("aria-label", "反馈情绪判断是否准确");
  emotion.addEventListener("click", () => renderEmotionFeedbackChoices(metadata, controls, status)); controls.appendChild(emotion);
  controls.appendChild(status); wrapper.appendChild(controls);
}

async function submitFeedback(messageId, feedback, controls, status) {
  const buttons = allButtons(controls); buttons.forEach((button) => { button.disabled = true; });
  try {
    await mutateJson(threadPath(`/messages/${encodeURIComponent(messageId)}/feedback`), {
      method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({feedback}),
    });
    controls.remove();
  } catch (error) {
    buttons.forEach((button) => { button.disabled = false; }); status.textContent = "评价保存失败";
  }
}

async function submitEmotionFeedback(metadata, feedback, status) {
  const predicted = metadata.predicted_emotion || (metadata.emotion_state && metadata.emotion_state.primary_emotion) || "";
  try {
    await mutateJson(threadPath("/emotion-feedback"), {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message_id: metadata.id, feedback, predicted_emotion: predicted, turn_count: metadata.turn_count || null})});
    status.textContent = "情绪反馈已保存";
  } catch (error) { status.textContent = "情绪反馈保存失败"; }
}

function renderEmotionFeedbackChoices(metadata, controls, status) {
  const choices = document.createElement("div");
  choices.className = "emotion-feedback-choices";
  emotionFeedbackChoices.forEach(([label, feedback]) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "feedback-button"; button.textContent = label;
    button.addEventListener("click", () => submitEmotionFeedback(metadata, feedback, status));
    choices.appendChild(button);
  });
  controls.insertBefore(choices, status);
}

function renderRegenerationReasons(wrapper, messageId, controls, status) {
  const reasons = document.createElement("div"); reasons.className = "regeneration-reasons";
  regenerationReasons.forEach((reason) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "feedback-button"; button.textContent = reason;
    button.addEventListener("click", () => submitRegeneration(wrapper, messageId, reason, controls, status)); reasons.appendChild(button);
  });
  controls.insertBefore(reasons, status);
}

async function submitRegeneration(wrapper, messageId, reason, controls, status) {
  const buttons = allButtons(controls); buttons.forEach((button) => { button.disabled = true; });
  const bubble = wrapper.children[0];
  const originalContent = bubble.textContent;
  let streamed = "";
  try {
    await runStream(threadPath(`/messages/${encodeURIComponent(messageId)}/regenerate:stream`), {reason, request_id: crypto.randomUUID()}, {
      token(data) { streamed += data.content; bubble.textContent = streamed; },
      done(data) { bubble.textContent = data.content; wrapper.setAttribute("data-message-id", data.message_id); },
      error(data) { throw new Error(data.error_code || "regeneration_failed"); },
    });
    await loadCurrentThread();
  } catch (error) {
    bubble.textContent = originalContent;
    if (error.name !== "AbortError") { buttons.forEach((button) => { button.disabled = false; }); status.textContent = "重新生成失败"; }
  }
}

async function loadProfile() {
  const payload = await readJson(() => clientPath("/profile")); profileState.profile = payload.profile || {}; return payload;
}
async function saveProfile(profile) {
  const payload = await mutateJson(clientPath("/profile"), {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({profile})});
  profileState.profile = payload.profile || {}; return payload;
}
async function requestProfileDraft() {
  const payload = await mutateJson(clientPath("/profile/draft"), {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({thread_id: state.threadId, answers: profileState.answers})});
  return payload.draft || {};
}

function closeProfilePanel() {
  if (profilePanelEl) profilePanelEl.hidden = true;
  if (profileBackdropEl) profileBackdropEl.hidden = true;
  if (profileState.previousFocus && profileState.previousFocus.focus) profileState.previousFocus.focus();
  profileState.previousFocus = null;
}
function clearProfileBody() { if (profilePanelBodyEl) profilePanelBodyEl.replaceChildren(); }
function profileStatus(text, error = false) { const element = document.createElement("p"); element.className = error ? "profile-status error" : "profile-status"; element.textContent = text; return element; }
function renderProfileForm(profile) {
  clearProfileBody(); const form = document.createElement("form"); form.className = "profile-form";
  profileFields.forEach(([key, label]) => { const field = document.createElement("label"); field.textContent = label; const input = document.createElement("textarea"); input.name = key; input.value = profile[key] || ""; field.appendChild(input); form.appendChild(field); });
  const save = document.createElement("button"); save.type = "submit"; save.textContent = "保存"; const status = profileStatus(); form.appendChild(save); form.appendChild(status);
  form.addEventListener("submit", async (event) => { event.preventDefault(); const next = {}; profileFields.forEach(([key]) => { const input = form.elements && form.elements[key]; if (input && input.value.trim()) next[key] = input.value.trim(); }); try { const result = await saveProfile(next); renderProfileForm(result.profile); } catch (error) { status.textContent = "保存失败"; } });
  profilePanelBodyEl.appendChild(form);
}
async function openProfilePanel() {
  if (!profilePanelEl || !profileBackdropEl || !profilePanelBodyEl) return;
  profileState.previousFocus = document.activeElement || null;
  profilePanelEl.hidden = false; profileBackdropEl.hidden = false; clearProfileBody(); profilePanelBodyEl.appendChild(profileStatus("正在加载画像…"));
  if (profileCloseEl) profileCloseEl.focus();
  try { const payload = await loadProfile(); renderProfileForm(payload.profile || {}); } catch (error) { clearProfileBody(); profilePanelBodyEl.appendChild(profileStatus("画像加载失败", true)); }
}
async function startProfileOnboarding() {
  profileState.previousFocus = document.activeElement || null;
  if (profilePromptEl) profilePromptEl.hidden = true; if (profilePanelEl) profilePanelEl.hidden = false; if (profileBackdropEl) profileBackdropEl.hidden = false;
  if (profileCloseEl) profileCloseEl.focus();
  try {
    const payload = await fetchJson("/api/profile/onboarding/questions");
    profileState.questions = payload.questions || [];
    clearProfileBody();
    const form = document.createElement("form"); form.className = "profile-form";
    profileState.questions.forEach((question) => {
      const field = document.createElement("label"); field.textContent = question.question;
      const input = document.createElement("textarea"); input.name = question.key;
      field.appendChild(input); form.appendChild(field);
    });
    const submit = document.createElement("button"); submit.type = "submit"; submit.textContent = "生成草稿"; form.appendChild(submit);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      profileState.answers = profileState.questions.map((question) => ({key: question.key, answer: (form.elements[question.key].value || "").trim()}));
      try { renderProfileForm(await requestProfileDraft()); } catch (error) { renderProfileForm({}); }
    });
    profilePanelBodyEl.appendChild(form);
  } catch (error) { renderProfileForm({}); }
}
async function maybeShowProfilePrompt() {
  if (!profilePromptEl || sessionStorage.getItem("profileOnboardingSkipped") === "true") return;
  try { profilePromptEl.hidden = !(await loadProfile()).is_empty; } catch (error) { profilePromptEl.hidden = true; }
}

async function initialize() {
  setLocked(true);
  try {
    await initializeIdentity(); renderThreads(); await loadCurrentThread(); maybeShowProfilePrompt();
  } catch (error) {
    if (emotionStatusEl) emotionStatusEl.textContent = "加载失败，请刷新重试";
  } finally { updateInteractionLocks(); }
}

if (formEl) formEl.addEventListener("submit", (event) => { event.preventDefault(); const message = inputEl.value.trim(); if (!message) return; inputEl.value = ""; streamMessage(message); });
if (inputEl) inputEl.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); formEl.requestSubmit(); } });
if (newThreadButtonEl) newThreadButtonEl.addEventListener("click", () => createThread().catch(() => showUiError("新建对话失败，请重试")));
if (deleteThreadButtonEl) deleteThreadButtonEl.addEventListener("click", () => deleteCurrentThread().catch(() => {}));
if (profileButtonEl) profileButtonEl.addEventListener("click", () => openProfilePanel().catch(() => showUiError("画像加载失败")));
if (profileCloseEl) profileCloseEl.addEventListener("click", closeProfilePanel);
if (profileBackdropEl) profileBackdropEl.addEventListener("click", closeProfilePanel);
if (profilePromptStartEl) profilePromptStartEl.addEventListener("click", startProfileOnboarding);
if (profilePromptSkipEl) profilePromptSkipEl.addEventListener("click", () => { sessionStorage.setItem("profileOnboardingSkipped", "true"); profilePromptEl.hidden = true; });

globalThis.__HDU_ERC_TEST__ = {
  collectSseFrames, createThread, deleteCurrentThread, selectThread,
  streamMessage, submitRegeneration, renderSnapshot,
  setStateForTest(clientId, threadId) { state.clientId = clientId; state.threadId = threadId; },
};
initialize();
