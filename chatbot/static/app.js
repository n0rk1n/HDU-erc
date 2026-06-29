const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const buttonEl = document.querySelector("#send-button");
const emotionStatusEl = document.querySelector("#emotion-status");
const safetyStatusEl = document.querySelector("#safety-status");
const emotionTimelineEl = document.querySelector("#emotion-timeline");
const profileButtonEl = document.querySelector("#profile-button");
const profilePanelEl = document.querySelector("#profile-panel");
const profileBackdropEl = document.querySelector("#profile-backdrop");
const profileCloseEl = document.querySelector("#profile-close");
const profilePanelBodyEl = document.querySelector("#profile-panel-body");
const profilePromptEl = document.querySelector("#profile-onboarding-prompt");
const profilePromptStartEl = document.querySelector("#profile-onboarding-start");
const profilePromptSkipEl = document.querySelector("#profile-onboarding-skip");
const regenerationReasons = ["不准确", "不完整", "没有理解我的问题", "语气不合适", "其他"];
const emotionFeedbackChoices = [
  ["Accurate", "accurate"],
  ["Too positive", "too_positive"],
  ["Too negative", "too_negative"],
  ["Wrong", "wrong_emotion"],
];
const profileFields = [
  ["preferred_name", "称呼"],
  ["life_stage", "身份或阶段"],
  ["companion_expectation", "陪伴期待"],
  ["response_style", "回应风格"],
  ["avoidance", "希望避免"],
];
const profileState = {
  profile: {},
  questions: [],
  answers: [],
  questionIndex: 0,
};
let currentEmotionState = null;

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

  const emotionButton = document.createElement("button");
  emotionButton.type = "button";
  emotionButton.className = "feedback-button";
  emotionButton.textContent = "Emotion?";
  emotionButton.setAttribute("aria-label", "Emotion correctness feedback");

  const status = document.createElement("span");
  status.className = "feedback-status";

  const buttons = [likeButton, dislikeButton, regenerateButton, emotionButton];
  likeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "like", controls, status)
  ));
  dislikeButton.addEventListener("click", () => (
    submitFeedback(metadata.id, "dislike", controls, status)
  ));
  regenerateButton.addEventListener("click", () => (
    renderRegenerationReasons(wrapper, metadata, controls, status, buttons)
  ));
  emotionButton.addEventListener("click", () => (
    renderEmotionFeedbackChoices(metadata, controls, status)
  ));

  controls.appendChild(likeButton);
  controls.appendChild(dislikeButton);
  controls.appendChild(regenerateButton);
  controls.appendChild(emotionButton);
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

async function submitEmotionFeedback(metadata, feedback, status) {
  status.textContent = "";
  const emotionState = metadata.emotion_state || null;
  const predictedEmotion = metadata.predicted_emotion
    || (emotionState && emotionState.primary_emotion)
    || (currentEmotionState && currentEmotionState.primary_emotion)
    || "";
  try {
    const body = {
      message_id: metadata.id || "",
      feedback,
      predicted_emotion: predictedEmotion,
    };
    if (metadata.turn_count) {
      body.turn_count = metadata.turn_count;
    }
    const response = await fetch("/api/emotion/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error("Emotion feedback failed.");
    }
    status.textContent = "情绪反馈已保存";
  } catch (error) {
    status.textContent = "情绪反馈保存失败";
  }
}

function renderEmotionFeedbackChoices(metadata, controls, status) {
  let choices = findChildByClass(controls, "emotion-feedback-choices");
  if (choices) {
    choices.remove();
    return;
  }

  choices = document.createElement("div");
  choices.className = "emotion-feedback-choices";
  emotionFeedbackChoices.forEach(([label, feedback]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "feedback-button emotion-feedback-choice";
    button.textContent = label;
    button.addEventListener("click", () => (
      submitEmotionFeedback(metadata, feedback, status)
    ));
    choices.appendChild(button);
  });
  controls.insertBefore(choices, status);
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
    currentEmotionState = {primary_emotion: emotion.emotion};
    emotionStatusEl.textContent = `情感状态：${emotion.emotion}`;
    clearSafetyStatus();
    return;
  }
  currentEmotionState = null;
  emotionStatusEl.textContent = "情感状态：暂无";
  clearSafetyStatus();
}

function renderEmotionState(state) {
  if (!state || !state.primary_emotion) {
    currentEmotionState = null;
    emotionStatusEl.textContent = "情感状态：暂无";
    clearSafetyStatus();
    return;
  }
  currentEmotionState = state;
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

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadProfile() {
  const payload = await fetchJson("/api/profile");
  profileState.profile = payload.profile || {};
  return payload;
}

async function saveProfile(profile) {
  const payload = await fetchJson("/api/profile", {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({profile}),
  });
  profileState.profile = payload.profile || {};
  return payload;
}

async function loadProfileQuestions() {
  const payload = await fetchJson("/api/profile/onboarding/questions");
  profileState.questions = payload.questions || [];
  return profileState.questions;
}

async function requestProfileDraft() {
  const payload = await fetchJson("/api/profile/onboarding/draft", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({answers: profileState.answers}),
  });
  return payload.draft || {};
}

function setProfilePromptSkipped() {
  try {
    sessionStorage.setItem("profileOnboardingSkipped", "true");
  } catch (error) {
    // Session storage can be unavailable in strict browser modes.
  }
}

function hasProfilePromptBeenSkipped() {
  try {
    return sessionStorage.getItem("profileOnboardingSkipped") === "true";
  } catch (error) {
    return false;
  }
}

function hideProfilePrompt() {
  if (!profilePromptEl) {
    return;
  }
  profilePromptEl.hidden = true;
}

async function maybeShowFirstRunProfilePrompt() {
  if (!profilePromptEl) {
    return;
  }
  if (hasProfilePromptBeenSkipped()) {
    return;
  }

  try {
    const payload = await loadProfile();
    if (payload.is_empty) {
      profilePromptEl.hidden = false;
    }
  } catch (error) {
    hideProfilePrompt();
  }
}

function openProfilePanel() {
  if (!profilePanelEl || !profileBackdropEl || !profilePanelBodyEl) {
    return;
  }
  profilePanelEl.hidden = false;
  profileBackdropEl.hidden = false;
  renderProfileLoading();
  loadProfile()
    .then((payload) => {
      if (payload.is_empty) {
        renderProfileEmpty();
      } else {
        renderProfileForm(profileState.profile, {
          title: "编辑我的画像",
          description: "这些内容会用于后续聊天中的称呼、陪伴方式和回应边界。",
        });
      }
    })
    .catch(() => {
      renderProfileError("画像加载失败，请稍后重试。", openProfilePanel);
    });
}

function closeProfilePanel() {
  if (!profilePanelEl || !profileBackdropEl) {
    return;
  }
  profilePanelEl.hidden = true;
  profileBackdropEl.hidden = true;
}

function clearProfileBody() {
  profilePanelBodyEl.replaceChildren();
}

function createProfileStatus(message = "", isError = false) {
  const status = document.createElement("p");
  status.className = isError ? "profile-status error" : "profile-status";
  status.textContent = message;
  return status;
}

function renderProfileLoading() {
  clearProfileBody();
  const status = createProfileStatus("正在加载画像…");
  profilePanelBodyEl.appendChild(status);
}

function renderProfileError(message, retryHandler) {
  clearProfileBody();
  const title = document.createElement("h3");
  title.textContent = "暂时无法处理";
  const status = createProfileStatus(message, true);
  const actions = document.createElement("div");
  actions.className = "profile-actions";
  const retryButton = document.createElement("button");
  retryButton.type = "button";
  retryButton.textContent = "重试";
  retryButton.addEventListener("click", retryHandler);
  actions.appendChild(retryButton);
  profilePanelBodyEl.appendChild(title);
  profilePanelBodyEl.appendChild(status);
  profilePanelBodyEl.appendChild(actions);
}

function renderProfileEmpty() {
  clearProfileBody();
  const title = document.createElement("h3");
  title.textContent = "还没有画像";
  const description = document.createElement("p");
  description.textContent = "可以回答几个问题生成草稿，也可以直接手动填写。";
  const actions = document.createElement("div");
  actions.className = "profile-actions";

  const startButton = document.createElement("button");
  startButton.type = "button";
  startButton.textContent = "开始问答";
  startButton.addEventListener("click", startProfileOnboarding);

  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.className = "secondary-button";
  editButton.textContent = "手动填写";
  editButton.addEventListener("click", () => {
    renderProfileForm({}, {
      title: "手动填写画像",
      description: "留空的字段不会保存。",
    });
  });

  actions.appendChild(startButton);
  actions.appendChild(editButton);
  profilePanelBodyEl.appendChild(title);
  profilePanelBodyEl.appendChild(description);
  profilePanelBodyEl.appendChild(actions);
}

function renderProfileForm(profile, options = {}) {
  clearProfileBody();
  const title = document.createElement("h3");
  title.textContent = options.title || "编辑我的画像";
  const description = document.createElement("p");
  description.textContent = options.description || "修改后点击保存，关闭面板不会自动保存。";

  const form = document.createElement("form");
  form.className = "profile-form";

  profileFields.forEach(([key, label]) => {
    const field = document.createElement("div");
    field.className = "profile-field";

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", `profile-${key}`);
    labelEl.textContent = label;

    const input = document.createElement("textarea");
    input.id = `profile-${key}`;
    input.name = key;
    input.rows = key === "preferred_name" ? 2 : 3;
    input.value = profile[key] || "";

    field.appendChild(labelEl);
    field.appendChild(input);
    form.appendChild(field);
  });

  const actions = document.createElement("div");
  actions.className = "profile-actions";
  const saveButton = document.createElement("button");
  saveButton.type = "submit";
  saveButton.textContent = "保存";
  const cancelButton = document.createElement("button");
  cancelButton.type = "button";
  cancelButton.className = "secondary-button";
  cancelButton.textContent = "关闭";
  cancelButton.addEventListener("click", closeProfilePanel);
  actions.appendChild(saveButton);
  actions.appendChild(cancelButton);

  const status = createProfileStatus();
  form.appendChild(actions);
  form.appendChild(status);
  form.addEventListener("submit", (event) => (
    submitProfileForm(event, form, status, saveButton)
  ));

  profilePanelBodyEl.appendChild(title);
  profilePanelBodyEl.appendChild(description);
  profilePanelBodyEl.appendChild(form);
}

async function startProfileOnboarding() {
  hideProfilePrompt();
  profilePanelEl.hidden = false;
  profileBackdropEl.hidden = false;
  profileState.answers = [];
  profileState.questionIndex = 0;
  renderProfileLoading();

  try {
    await loadProfileQuestions();
    renderProfileQuestion();
  } catch (error) {
    renderProfileError("问答加载失败，请稍后重试。", startProfileOnboarding);
  }
}

function renderProfileQuestion() {
  clearProfileBody();
  const question = profileState.questions[profileState.questionIndex];
  if (!question) {
    buildProfileDraft();
    return;
  }

  const title = document.createElement("h3");
  title.textContent = `问题 ${profileState.questionIndex + 1} / ${profileState.questions.length}`;
  const prompt = document.createElement("p");
  prompt.textContent = question.question;

  const field = document.createElement("div");
  field.className = "profile-field";
  const label = document.createElement("label");
  label.setAttribute("for", "profile-question-answer");
  label.textContent = "你的回答";
  const answerInput = document.createElement("textarea");
  answerInput.id = "profile-question-answer";
  answerInput.rows = 4;
  field.appendChild(label);
  field.appendChild(answerInput);

  const actions = document.createElement("div");
  actions.className = "profile-actions";
  const nextButton = document.createElement("button");
  nextButton.type = "button";
  nextButton.textContent = "下一题";
  nextButton.addEventListener("click", () => {
    answerProfileQuestion(answerInput.value);
  });
  const skipButton = document.createElement("button");
  skipButton.type = "button";
  skipButton.className = "secondary-button";
  skipButton.textContent = "跳过";
  skipButton.addEventListener("click", () => {
    answerProfileQuestion("");
  });
  actions.appendChild(nextButton);
  actions.appendChild(skipButton);

  profilePanelBodyEl.appendChild(title);
  profilePanelBodyEl.appendChild(prompt);
  profilePanelBodyEl.appendChild(field);
  profilePanelBodyEl.appendChild(actions);
  answerInput.focus();
}

function answerProfileQuestion(answer) {
  const question = profileState.questions[profileState.questionIndex];
  if (!question) {
    return;
  }
  profileState.answers.push({
    key: question.key,
    answer: answer.trim(),
  });
  profileState.questionIndex += 1;
  renderProfileQuestion();
}

async function buildProfileDraft() {
  renderProfileLoading();
  try {
    const draft = await requestProfileDraft();
    renderProfileForm(draft, {
      title: "确认画像草稿",
      description: "这是根据你的回答生成的草稿，可以先修改再保存。",
    });
  } catch (error) {
    renderProfileError("画像草稿生成失败，请稍后重试。", buildProfileDraft);
  }
}

async function submitProfileForm(event, form, status, saveButton) {
  event.preventDefault();
  const profile = {};
  profileFields.forEach(([key]) => {
    const field = form.elements[key];
    if (field && field.value.trim()) {
      profile[key] = field.value.trim();
    }
  });

  saveButton.disabled = true;
  status.className = "profile-status";
  status.textContent = "正在保存…";
  try {
    const payload = await saveProfile(profile);
    renderProfileForm(payload.profile || {}, {
      title: "编辑我的画像",
      description: "画像已保存。后续聊天会使用这些偏好。",
    });
  } catch (error) {
    saveButton.disabled = false;
    status.className = "profile-status error";
    status.textContent = "保存失败，当前编辑已保留，请稍后重试。";
  }
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
    maybeShowFirstRunProfilePrompt();
  } finally {
    setLocked(false);
  }
}

async function streamMessage(message) {
  setLocked(true);
  let aiMessage = null;
  let source = null;

  try {
    const response = await fetch("/api/chat/streams", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message}),
    });
    if (!response.ok) {
      throw new Error("Message request failed.");
    }
    const payload = await response.json();
    source = new EventSource(`/api/chat/streams/${encodeURIComponent(payload.stream_id)}`);
  } catch (error) {
    aiMessage = addMessage("ai", "发送失败，请稍后重试");
    setLocked(false);
    return;
  }

  source.addEventListener("user_message", (event) => {
    const payload = JSON.parse(event.data);
    clearSafetyStatus();
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
      renderFeedbackControls(aiMessage.wrapper, {
        id: payload.message_id,
        feedback: null,
        turn_count: payload.turn_count || null,
        emotion_state: payload.emotion_state || null,
        predicted_emotion: payload.predicted_emotion || "",
      });
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

if (profileButtonEl) {
  profileButtonEl.addEventListener("click", openProfilePanel);
}
if (profileCloseEl) {
  profileCloseEl.addEventListener("click", closeProfilePanel);
}
if (profileBackdropEl) {
  profileBackdropEl.addEventListener("click", closeProfilePanel);
}
if (profilePromptStartEl) {
  profilePromptStartEl.addEventListener("click", startProfileOnboarding);
}
if (profilePromptSkipEl) {
  profilePromptSkipEl.addEventListener("click", () => {
    setProfilePromptSkipped();
    hideProfilePrompt();
  });
}

initialize();
