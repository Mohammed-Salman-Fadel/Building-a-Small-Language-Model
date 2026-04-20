const elements = {
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#emptyState"),
  typing: document.querySelector("#typingIndicator"),
  input: document.querySelector("#inputBox"),
  send: document.querySelector("#sendBtn"),
  stop: document.querySelector("#stopBtn"),
  charCount: document.querySelector("#charCount"),
  topbarTitle: document.querySelector("#topbarTitle"),
  modelBadge: document.querySelector("#modelBadge"),
  modelName: document.querySelector("#modelName"),
  runtimeDot: document.querySelector("#runtimeDot"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  connectionStatus: document.querySelector("#connectionStatus"),
  historyList: document.querySelector("#historyList"),
  newChat: document.querySelector("#newChatBtn"),
  clearChat: document.querySelector("#clearChatBtn"),
  clearAll: document.querySelector("#clearAllBtn"),
  reload: document.querySelector("#reloadBtn"),
  copyAll: document.querySelector("#copyAllBtn"),
  export: document.querySelector("#exportBtn"),
  lowerTemp: document.querySelector("#lowerTempBtn"),
  balancedTemp: document.querySelector("#balancedTempBtn"),
};

const state = {
  history: [],
  savedChats: [],
  isGenerating: false,
  abortController: null,
  temperature: 0.8,
  topK: 40,
  maxNewTokens: 128,
};

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatText(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/```(\w*)\n?([\s\S]*?)```/g, "<pre><code>$2</code></pre>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function nowTime() {
  const now = new Date();
  return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
}

function setRuntimeStatus(kind, text) {
  elements.runtimeDot.classList.toggle("error", kind === "error");
  elements.runtimeStatus.textContent = text;
  elements.connectionStatus.lastChild.textContent = ` Connection: ${kind === "error" ? "offline" : "local"}`;
}

function setBusy(value) {
  state.isGenerating = value;
  elements.input.disabled = value;
  elements.stop.classList.toggle("visible", value);
  elements.typing.classList.toggle("visible", value);
  updateSendState();
}

function updateSendState() {
  elements.send.disabled = state.isGenerating || elements.input.value.trim().length === 0;
}

function updateInputHeight() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
}

function updateCharCount() {
  const length = elements.input.value.length;
  elements.charCount.textContent = `${length} / 4000`;
  elements.charCount.classList.toggle("warn", length > 3500);
}

function handleInput() {
  updateInputHeight();
  updateCharCount();
  updateSendState();
}

function appendMessage(role, text) {
  elements.emptyState.style.display = "none";

  const message = document.createElement("div");
  message.className = "message";

  const isUser = role === "user";
  const isError = role === "error";
  const avatarClass = isUser ? "avatar-user" : isError ? "avatar-error" : "avatar-bot";
  const avatarText = isUser ? "YOU" : isError ? "ERR" : "AI";
  const sender = isUser ? "YOU" : isError ? "ERROR" : "LOCALMIND";

  message.innerHTML = `
    <div class="msg-inner">
      <div class="avatar ${avatarClass}">${avatarText}</div>
      <div class="msg-body">
        <div class="msg-sender">
          ${sender}
          <span class="msg-time">${nowTime()}</span>
        </div>
        <div class="msg-text">${formatText(text)}</div>
        ${!isUser && !isError ? `
        <div class="msg-actions">
          <button class="msg-action-btn" type="button" data-copy-message>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            COPY
          </button>
        </div>` : ""}
      </div>
    </div>
  `;

  elements.messages.appendChild(message);
  scrollToBottom();
  return message;
}

function scrollToBottom() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    const runtime = data.runtime;
    elements.modelName.textContent = `${runtime.mode}-${runtime.device}`;
    setRuntimeStatus("ready", `CONNECTED - ${runtime.mode} - ${runtime.device} - context ${runtime.context_length}`);
  } catch (error) {
    elements.modelName.textContent = "runtime-error";
    setRuntimeStatus("error", `ERROR - ${error.message}`);
  }
}

async function reloadModel() {
  setRuntimeStatus("ready", "RELOADING - local model runtime");
  try {
    const response = await fetch("/api/reload", { method: "POST" });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    await loadStatus();
  } catch (error) {
    setRuntimeStatus("error", `ERROR - ${error.message}`);
  }
}

async function sendMessage() {
  const text = elements.input.value.trim();
  if (!text || state.isGenerating) {
    return;
  }

  appendMessage("user", text);
  elements.topbarTitle.textContent = `// ${text.slice(0, 54)}${text.length > 54 ? "..." : ""}`;
  elements.input.value = "";
  handleInput();
  setBusy(true);

  state.abortController = new AbortController();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortController.signal,
      body: JSON.stringify({
        message: text,
        history: state.history,
        temperature: state.temperature,
        top_k: state.topK,
        max_new_tokens: state.maxNewTokens,
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const data = await response.json();
    state.history = data.history || [];
    appendMessage("bot", data.response);
    saveCurrentConversation();
  } catch (error) {
    if (error.name !== "AbortError") {
      appendMessage("error", error.message);
    }
  } finally {
    state.abortController = null;
    setBusy(false);
    elements.input.focus();
  }
}

function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
  }
  setBusy(false);
}

function clearVisibleMessages() {
  [...elements.messages.children].forEach((child) => {
    if (child !== elements.emptyState) {
      child.remove();
    }
  });
  elements.emptyState.style.display = "";
}

function newChat() {
  state.history = [];
  clearVisibleMessages();
  elements.topbarTitle.textContent = "// new conversation";
  elements.input.focus();
}

function clearAllHistory() {
  state.savedChats = [];
  renderHistoryList();
  newChat();
}

function saveCurrentConversation() {
  if (state.history.length === 0) {
    return;
  }

  const firstPrompt = state.history[0].user;
  const existingIndex = state.savedChats.findIndex((chat) => chat.title === firstPrompt);
  const chat = {
    title: firstPrompt,
    date: "TODAY",
    history: [...state.history],
  };

  if (existingIndex >= 0) {
    state.savedChats[existingIndex] = chat;
  } else {
    state.savedChats.unshift(chat);
  }
  state.savedChats = state.savedChats.slice(0, 8);
  renderHistoryList();
}

function renderHistoryList() {
  elements.historyList.innerHTML = "";
  if (state.savedChats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "No saved conversations yet.";
    elements.historyList.appendChild(empty);
    return;
  }

  state.savedChats.forEach((chat) => {
    const item = document.createElement("button");
    item.className = "history-item";
    item.type = "button";
    item.innerHTML = `
      <div class="hi-title">${escapeHtml(chat.title)}</div>
      <div class="hi-date">${chat.date}</div>
    `;
    item.addEventListener("click", () => loadConversation(chat));
    elements.historyList.appendChild(item);
  });
}

function loadConversation(chat) {
  state.history = [...chat.history];
  clearVisibleMessages();
  for (const turn of state.history) {
    appendMessage("user", turn.user);
    appendMessage("bot", turn.assistant);
  }
  elements.topbarTitle.textContent = `// ${chat.title.slice(0, 54)}${chat.title.length > 54 ? "..." : ""}`;
}

function conversationText() {
  return state.history
    .map((turn) => `You: ${turn.user}\nLocalMind: ${turn.assistant}`)
    .join("\n\n");
}

async function copyAll() {
  await navigator.clipboard.writeText(conversationText());
}

function exportChat() {
  const blob = new Blob([conversationText()], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "localmind-chat.txt";
  link.click();
  URL.revokeObjectURL(link.href);
}

function setTemperature(value) {
  state.temperature = value;
  elements.lowerTemp.classList.toggle("active", value === 0.4);
  elements.balancedTemp.classList.toggle("active", value === 0.8);
}

elements.input.addEventListener("input", handleInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

elements.send.addEventListener("click", sendMessage);
elements.stop.addEventListener("click", stopGeneration);
elements.newChat.addEventListener("click", newChat);
elements.clearChat.addEventListener("click", newChat);
elements.clearAll.addEventListener("click", clearAllHistory);
elements.reload.addEventListener("click", reloadModel);
elements.modelBadge.addEventListener("click", reloadModel);
elements.copyAll.addEventListener("click", copyAll);
elements.export.addEventListener("click", exportChat);
elements.lowerTemp.addEventListener("click", () => setTemperature(0.4));
elements.balancedTemp.addEventListener("click", () => setTemperature(0.8));

elements.messages.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-message]");
  if (!button) {
    return;
  }
  const text = button.closest(".msg-body").querySelector(".msg-text").innerText;
  await navigator.clipboard.writeText(text);
  button.textContent = "COPIED";
  setTimeout(() => {
    button.textContent = "COPY";
  }, 1200);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt;
    handleInput();
    elements.input.focus();
  });
});

setTemperature(0.8);
handleInput();
renderHistoryList();
loadStatus();
