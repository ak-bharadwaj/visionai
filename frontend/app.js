// app.js — main application logic
// Requires: overlay (from overlay.js), camera (from camera.js), audio (from audio.js)

// ─── Toast ────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), duration);
}
window.showToast = showToast;

// ─── State ────────────────────────────────────────────────────────
let ws = null;
let overlayActive          = false;
let detectivePanelActive   = false;
let demoPresentationActive = false;
window.currentMode = 'NAVIGATE';

// Track last input source for output routing ('voice' or 'chat')
let lastInputSource = 'chat';

// Severity icons matching sev0–sev3 banner states
// sev1 = danger (red), sev2 = warning (orange), sev3 = success (green)
const SEV_ICONS = ['', '🚨', '⚠️', '✅'];

// ─── WebSocket ────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/guidance`);

  ws.onopen = () => {
    document.getElementById('ws-dot').className = 'ws-dot connected';
    document.getElementById('ws-label').textContent = 'Live';
    showToast('Connected to VisionTalk');
  };

  ws.onclose = () => {
    // Null out handlers on the closing socket to prevent ghost fires
    // after a new WebSocket is already created.
    ws.onerror = null;
    ws.onclose = null;
    document.getElementById('ws-dot').className = 'ws-dot disconnected';
    document.getElementById('ws-label').textContent = 'Reconnecting...';
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => {
    document.getElementById('ws-label').textContent = 'Error';
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'narration') handleNarration(data);
      else if (data.type === 'answer')       handleAnswer(data);
      else if (data.type === 'reading')      handleReading(data);
      else if (data.type === 'system')       handleSystem(data);
      else if (data.type === 'init')         handleInit(data);
      else if (data.type === 'found_object') handleFoundObject(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };
}
connectWS();

// ─── sendCommand ──────────────────────────────────────────────────
function sendCommand(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}
window.sendCommand = sendCommand;

// ─── WS handlers ─────────────────────────────────────────────────
function handleInit(data) {
  if (data.mode) applyModeState(data.mode);
}

function handleNarration(data) {
  if (data.text) setBanner(data.text, data.severity || 0);
  if (data.fps)  updateFps(data.fps);
  updateDetectivePanel(data.detections, data.fps);
  if ((overlayActive || demoPresentationActive) && data.detections?.length) {
    overlay.update(data.detections);
  } else if (!overlayActive && !demoPresentationActive) {
    overlay.clear();
  }
}

function handleAnswer(data) {
  const answer = data.answer || 'No answer received.';
  const src    = data.input_source || lastInputSource;
  lastInputSource = src;  // keep in sync for subsequent btn-diff clicks

  // Show answer in guidance banner (green = success)
  setBanner(answer, 3);
  if (data.fps) updateFps(data.fps);

  // Add to conversation panel (always — voice or chat)
  if (data.question && data.answer) {
    addConversationTurn(data.question, data.answer, src);
  }

  // Keep overlay clear during answer display
  overlay.clear();

  // Stay in ASK for multi-turn — never auto-return to NAVIGATE
  if (window.currentMode === 'ASK' && src !== 'voice') {
    setTimeout(() => document.getElementById('ask-input')?.focus(), 200);
  }
}

function handleReading(data) {
  if (data.text) setBanner(data.text, 0);
  if (data.fps)  updateFps(data.fps);
  if (data.detections) updateDetectivePanel(data.detections, data.fps);

  // If OCR found text, persist it in the conversation panel
  if (data.text && data.text !== 'No text found. Move closer or adjust angle.' &&
      data.text.startsWith('Reading:')) {
    addReadingToConversation(data.text);
  }
}

function handleSystem(data) {
  if (data.text) {
    setBanner(data.text, 0);
    showToast(data.text);
  }
}

function handleFoundObject(data) {
  // Hide the find banner — target has been located
  document.getElementById('find-banner')?.classList.add('hidden');
  // Play success chime (Web Audio API, offline, no assets needed)
  window.playSuccess?.();
  // Show a success toast and update the guidance banner
  if (data.text) {
    showToast(data.text, 3500);
    setBanner(data.text, 3);  // severity 3 = green / success
  }
}

// ─── UI helpers ───────────────────────────────────────────────────
function setBanner(text, severity) {
  document.getElementById('guidance-text').textContent = text;
  const banner = document.getElementById('guidance-banner');
  const sev    = Math.min(severity, 3);
  banner.className = `guidance-banner sev${sev}`;
  // Update severity icon
  const icon = document.getElementById('guidance-sev-icon');
  if (icon) icon.textContent = SEV_ICONS[sev] || '';
}

function updateFps(fps) {
  if (fps) document.getElementById('fps-display').textContent = `${fps} fps`;
}

function applyModeState(mode) {
  const current = (typeof mode === 'string') ? mode : (mode.current_mode || 'NAVIGATE');
  window.currentMode = current;
  document.getElementById('mode-badge').textContent = current;

  // Update button active + aria-pressed states
  document.querySelectorAll('.mode-btn').forEach(btn => {
    const isActive = btn.dataset.mode === current;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });

  // Show ask-row only in ASK mode
  document.getElementById('ask-row').classList.toggle('hidden', current !== 'ASK');

  // Show conversation panel if switching to ASK and it has content
  if (current === 'ASK') {
    const msgs = document.getElementById('convo-messages');
    if (msgs && msgs.children.length > 0) {
      showPanel(document.getElementById('conversation-panel'));
    }
    // Auto-focus the text input
    setTimeout(() => document.getElementById('ask-input')?.focus(), 120);
  }
}
window.applyModeState = applyModeState;

// ─── Mode buttons ─────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    sendCommand({ type: 'command', action: 'set_mode', mode });
    applyModeState({ current_mode: mode });
    if (mode === 'ASK') {
      showToast('ASK mode — speak or type your question');
    } else if (mode === 'NAVIGATE') {
      showToast('NAVIGATE — spatial narration active');
    } else if (mode === 'READ') {
      showToast('READ — point camera at text');
    }
  });
});

// ─── ASK input (chat path) ────────────────────────────────────────
function submitQuestion() {
  const input = document.getElementById('ask-input');
  const q = input.value.trim();
  if (!q) return;

  lastInputSource = 'chat';

  if (window.currentMode !== 'ASK') {
    applyModeState({ current_mode: 'ASK' });
  }

  sendCommand({ type: 'command', action: 'ask', question: q, input_source: 'chat' });
  input.value = '';

  // Show conversation panel immediately so user sees the Q appear when answer arrives
  showPanel(document.getElementById('conversation-panel'));

  showToast(`Asking: "${q}"`);
}
document.getElementById('btn-send').addEventListener('click', submitQuestion);
document.getElementById('ask-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitQuestion();
});

// ─── Secondary controls ────────────────────────────────────────────
document.getElementById('btn-overlay').addEventListener('click', () => {
  overlayActive = !overlayActive;
  document.getElementById('btn-overlay').classList.toggle('active', overlayActive);
  sendCommand({ type: 'command', action: 'toggle_overlay' });
  if (!overlayActive) overlay.clear();
  showToast(overlayActive ? 'Bounding boxes ON' : 'Bounding boxes OFF');
});

document.getElementById('btn-snapshot').addEventListener('click', () => {
  sendCommand({ type: 'command', action: 'snapshot' });
  showToast('Scene snapshot saved!');
});

document.getElementById('btn-diff').addEventListener('click', () => {
  sendCommand({ type: 'command', action: 'scene_diff', input_source: lastInputSource });
  showToast('Comparing with snapshot...');
});

// ─── Settings panel ────────────────────────────────────────────────
const settingsPanel = document.getElementById('settings-panel');

function showPanel(el) {
  el.classList.remove('hidden');
  el.classList.add('panel-entering');
  el.addEventListener('animationend', () => el.classList.remove('panel-entering'), { once: true });
}
function hidePanel(el) {
  el.classList.add('hidden');
}

document.getElementById('btn-settings').addEventListener('click', () => {
  if (settingsPanel.classList.contains('hidden')) showPanel(settingsPanel);
  else hidePanel(settingsPanel);
});

// Close button inside panel
document.getElementById('btn-settings-close').addEventListener('click', () => {
  hidePanel(settingsPanel);
});

// Close when clicking outside the panel
document.addEventListener('click', (e) => {
  if (!settingsPanel.classList.contains('hidden') &&
      !settingsPanel.contains(e.target) &&
      e.target.id !== 'btn-settings') {
    hidePanel(settingsPanel);
  }
});

// ─── Demo Presentation Mode ────────────────────────────────────────
document.getElementById('btn-demo-mode').addEventListener('click', () => {
  demoPresentationActive = !demoPresentationActive;
  const btn = document.getElementById('btn-demo-mode');
  btn.classList.toggle('active', demoPresentationActive);
  btn.querySelector('.demo-btn-label').textContent =
    demoPresentationActive ? 'Exit Demo' : 'Demo Mode';
  document.body.classList.toggle('demo-presentation', demoPresentationActive);

  if (demoPresentationActive) {
    // Force overlay on (guard: only if not already active)
    if (!overlayActive) {
      overlayActive = true;
      document.getElementById('btn-overlay').classList.add('active');
      sendCommand({ type: 'command', action: 'toggle_overlay' });
    }
    // Open detective panel
    detectivePanelActive = true;
    showPanel(document.getElementById('detective-panel'));
    document.getElementById('btn-detective').classList.add('active');
    document.getElementById('btn-detective').querySelector('.demo-btn-label').textContent = 'Hide Data';
    showToast('Demo Mode ON — Judges can see live AI');
  } else {
    document.getElementById('btn-detective').querySelector('.demo-btn-label').textContent = 'Show Data';
    showToast('Demo Mode OFF');
  }
});

// ─── Detective Panel ──────────────────────────────────────────────
document.getElementById('btn-detective').addEventListener('click', () => {
  detectivePanelActive = !detectivePanelActive;
  const panel = document.getElementById('detective-panel');
  if (detectivePanelActive) showPanel(panel); else hidePanel(panel);
  document.getElementById('btn-detective').classList.toggle('active', detectivePanelActive);
  document.getElementById('btn-detective').querySelector('.demo-btn-label').textContent =
    detectivePanelActive ? 'Hide Data' : 'Show Data';
  showToast(detectivePanelActive ? 'Detective Panel ON' : 'Detective Panel OFF');
});

function updateDetectivePanel(detections, fps) {
  if (!detectivePanelActive && !demoPresentationActive) return;
  if (fps) document.getElementById('det-fps').textContent = `FPS: ${fps}`;
  const container = document.getElementById('detective-table');
  if (!detections || detections.length === 0) {
    container.innerHTML = '<p class="detective-empty">No objects detected this frame.</p>';
    return;
  }
  const rows = detections.map(d => `
    <tr>
      <td><strong>${d.class_name}</strong></td>
      <td class="det-lv${d.distance_level}">${d.distance}</td>
      <td>${d.direction}</td>
      <td>${Math.round(d.confidence * 100)}%</td>
      <td style="font-size:9px;font-family:monospace">[${d.x1},${d.y1}]→[${d.x2},${d.y2}]</td>
    </tr>
  `).join('');
  container.innerHTML = `
    <table>
      <thead><tr>
        <th>Object</th><th>Distance</th><th>Direction</th>
        <th>Confidence</th><th>BBox</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── Conversation Panel ────────────────────────────────────────────
const MAX_CONVO_TURNS = 4;

function addConversationTurn(question, answer, inputSource) {
  const panel = document.getElementById('conversation-panel');
  const msgs  = document.getElementById('convo-messages');
  showPanel(panel);

  const qEl = document.createElement('div');
  qEl.className   = inputSource === 'voice' ? 'bubble-q bubble-q-voice' : 'bubble-q';
  qEl.textContent = question;

  const aEl = document.createElement('div');
  aEl.className   = 'bubble-a';
  aEl.textContent = answer;

  msgs.appendChild(qEl);
  msgs.appendChild(aEl);

  // Keep max 4 turns (8 bubbles = 4 Q + 4 A)
  while (msgs.children.length > MAX_CONVO_TURNS * 2) {
    msgs.removeChild(msgs.firstChild);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

// Reading mode — show OCR results in conversation panel so they persist
const _readHistory = new Set();
function addReadingToConversation(text) {
  if (_readHistory.has(text)) return;
  _readHistory.add(text);

  const panel = document.getElementById('conversation-panel');
  const msgs  = document.getElementById('convo-messages');

  if (window.currentMode === 'READ') {
    showPanel(panel);
  }

  const el = document.createElement('div');
  el.className   = 'bubble-read';
  el.textContent = text;
  msgs.appendChild(el);

  while (msgs.children.length > MAX_CONVO_TURNS * 2 + 4) {
    msgs.removeChild(msgs.firstChild);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

document.getElementById('btn-clear-convo').addEventListener('click', () => {
  document.getElementById('convo-messages').innerHTML = '';
  hidePanel(document.getElementById('conversation-panel'));
  _readHistory.clear();
  // Also clear backend brain + OCR history so next answer is fresh
  sendCommand({ type: 'command', action: 'clear_history' });
});
