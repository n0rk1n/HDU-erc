(() => {
  "use strict";

  const stepCopy = {
    received: ["收到你的消息", "本轮消息已保存。"],
    decision: ["判断是否需要情绪识别", "结合本轮对话与最近识别情况，判断是否需要重新分析。"],
    emotion: ["分析对话情绪", "正在结合对话内容识别情绪。"],
    response: ["组织并生成回复", "结合当前可用的对话信息，生成回应。"],
  };
  const stateCopy = { completed: "已完成", running: "进行中", pending: "待处理", failed: "未完成", skipped: "已跳过" };
  const symbols = { completed: "✓", running: "", pending: "·", failed: "!", skipped: "−" };

  function element(tag, className, text = "") {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
  }

  function percent(emotion) {
    return Number.isFinite(emotion?.confidence) ? `${Math.round(emotion.confidence * 100)}%` : "—";
  }

  function timestamp(value) {
    const date = new Date(value);
    return value && !Number.isNaN(date.getTime())
      ? date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })
      : "";
  }

  function updateBadge(emotion) {
    const badge = document.querySelector("#emotion-badge");
    const label = document.querySelector("#emotion-label");
    const meta = document.querySelector("#emotion-meta");
    label.textContent = emotion?.display_label || "尚未识别";
    meta.textContent = emotion ? `最近识别 · ${timestamp(emotion.analyzed_at)}` : "随对话按需分析";
    badge.dataset.state = emotion ? "recognized" : "empty";
    badge.title = emotion
      ? `最近一次识别：${emotion.display_label}；模型置信度 ${percent(emotion)}。本轮未识别时保留此结果。`
      : "调用情绪识别后，这里会显示最近一次成功识别的结果。";
  }

  function headline(processing, status) {
    if (status === "failed") return "本轮处理未完成";
    const active = processing.steps.find((step) => step.status === "running");
    if (active) return {
      decision: "正在理解你的这轮表达", emotion: "正在分析对话情绪", response: "正在生成回复",
    }[active.id] || "正在处理";
    return processing.emotion_status === "completed" ? "情绪识别完成" : "本轮处理完成";
  }

  function summaryText(processing) {
    if (processing.emotion) return `${processing.emotion.display_label} · 模型置信度 ${percent(processing.emotion)}`;
    if (processing.emotion_status === "failed") return "本次识别未完成";
    if (processing.emotion_status === "skipped") return "本轮未调用情绪识别";
    if (processing.emotion_status === "running") return "结合对话内容识别情绪，结果即将显示";
    return "根据实际处理进度更新";
  }

  function render(item, processing, status = item.dataset.state) {
    let card = item.querySelector("details");
    if (!processing) {
      if (card) card.parentNode.removeChild(card);
      return;
    }
    const terminal = status === "completed" || status === "failed";
    const wasTerminal = card?.dataset.terminal === "true";
    if (!card) {
      card = element("details", "process-card");
      card.open = !terminal;
      const article = item.querySelector("article");
      article.insertBefore(card, item.querySelector("p"));
    } else if (terminal && !wasTerminal) {
      card.open = false;
    }
    card.dataset.terminal = String(terminal);
    card.dataset.state = terminal ? status : "running";
    const summary = element("summary", "process-summary");
    const icon = element("span", "process-symbol", terminal ? (status === "failed" ? "!" : "✓") : "✧");
    icon.setAttribute("aria-hidden", "true");
    const heading = element("div", "process-heading");
    heading.append(element("strong", "process-title", headline(processing, status)),
      element("span", "process-subtitle", summaryText(processing)));
    const elapsed = Number.isFinite(processing.elapsed_ms) ? `本轮 ${(processing.elapsed_ms / 1000).toFixed(1)} 秒` : "处理中";
    summary.append(icon, heading, element("span", "process-duration", elapsed), element("span", "process-chevron", "⌄"));
    const content = element("div", "process-content");
    content.append(element("div", "process-section-label", "处理过程"));
    const steps = element("ol", "process-steps");
    for (const step of processing.steps) {
      if (!stepCopy[step.id] || !stateCopy[step.status]) continue;
      const row = element("li", "process-step");
      row.dataset.state = step.status;
      const mark = element("span", "step-mark", symbols[step.status]);
      mark.setAttribute("aria-hidden", "true");
      const copy = element("div", "step-copy");
      let [title, description] = stepCopy[step.id];
      if (step.id === "emotion") {
        if (step.status === "completed") {
          title = "已完成情绪识别";
          description = "本轮识别结果已保存。";
        } else if (step.status === "failed") {
          title = "本次识别未完成";
          description = processing.emotion_invoked ? "已尝试识别，但未获得有效结果。" : "识别准备未完成，未调用识别模型。";
        } else if (step.status === "skipped") {
          title = "本轮未调用情绪识别";
          description = "本轮未执行新的情绪识别；已有结果时，右上角保留最近记录。";
        }
      }
      if (step.id === "response" && step.status === "failed") description = "回复未能完整生成，已保留可用内容。";
      if (step.id === "response" && step.status === "completed") description = "本轮回复已生成并保存。";
      if (step.id === "decision" && step.status === "failed") description = "本轮判断未能完成。";
      const titleRow = element("div", "step-title-row");
      titleRow.append(element("strong", "step-title", title), element("span", "step-state", stateCopy[step.status]));
      copy.append(titleRow, element("div", "step-description", description));
      row.append(mark, copy);
      steps.append(row);
    }
    content.append(steps);
    if (processing.emotion) {
      const result = processing.emotion;
      const panel = element("div", "emotion-result");
      panel.append(element("div", "process-section-label", "本轮情绪识别结果"));
      const resultRow = element("div", "emotion-result-row");
      resultRow.append(element("strong", "emotion-chip", result.display_label),
        element("span", "emotion-confidence", `模型置信度 ${percent(result)}`));
      panel.append(resultRow, element("div", "emotion-evidence", result.evidence),
        element("div", "emotion-footnote", "基于本轮及相关对话的模型判断，供你参考。"));
      content.append(panel);
    }
    card.replaceChildren(summary, content);
  }

  window.ChatPresentation = { render, updateBadge, headline };
})();
