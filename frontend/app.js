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

// FIND mode capture state (mirrors backend state machine)
// idle → confirming → captured → awaiting_question
let findCaptureState = 'idle';

// Severity icons matching sev0–sev3 banner states
// sev1 = danger (red), sev2 = warning (orange), sev3 = success (green)
const SEV_ICONS = ['', '🚨', '⚠️', '✅'];

// ─── Mode-driven capture timers ───────────────────────────────────
let captureTimer = null;

function startNavigateCapture() {
  stopCapture();
  captureTimer = setInterval(() => {
    if (window.currentMode !== 'NAVIGATE') return;
    window.sendFrameToBackend?.('NAVIGATE');
  }, 2000);
}

function startReadCapture() {
  stopCapture();
  captureTimer = setInterval(() => {
    if (window.currentMode !== 'READ') return;
    window.sendFrameToBackend?.('READ');
  }, 3000);
}

function stopCapture() {
  if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
}

// Called whenever the active mode changes (from applyModeState or pickMode)
function onModeChange(newMode) {
  stopCapture();
  if (newMode === 'NAVIGATE') {
    window.startCamera?.();
    startNavigateCapture();
  } else if (newMode === 'READ') {
    window.startCamera?.();
    startReadCapture();
  } else if (newMode === 'FIND') {
    window.startCamera?.();
    // No auto-send in FIND — user triggers manually
  } else if (newMode === 'ASK') {
    window.startCamera?.();
    // ASK: camera on but no auto-send; frame sent when user submits question
  } else {
    // Welcome / STOPPED / unknown — camera off
    window.stopCamera?.();
  }
}
window.onModeChange = onModeChange;

// ─── Welcome Screen ───────────────────────────────────────────────
(function initWelcome() {
  const overlay = document.getElementById('welcome-overlay');
  if (!overlay) return;

  function pickMode(mode) {
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.4s ease';
    setTimeout(() => overlay.remove(), 400);
    // OCR card maps to READ on backend
    const backendMode = mode === 'OCR' ? 'READ' : mode;
    sendCommand({ type: 'command', action: 'set_mode', mode: backendMode });
    applyModeState({ current_mode: backendMode });
    onModeChange(backendMode);
    showToast(`${backendMode} mode activated`);
  }
  window._welcomePickMode = pickMode;

  overlay.querySelectorAll('.welcome-card').forEach(card => {
    card.addEventListener('click', () => pickMode(card.dataset.mode));
  });

  // Voice on welcome screen — listen for mode names
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SR) {
    const rec = new SR();
    rec.lang = 'en-US'; rec.continuous = false; rec.interimResults = false;
    rec.onresult = (e) => {
      const t = e.results[0][0].transcript.toLowerCase().trim();
      if (t.includes('navigate')) pickMode('NAVIGATE');
      else if (t.includes('find'))                   pickMode('FIND');
      else if (t.includes('read') || t.includes('ocr')) pickMode('OCR');
      else { try { rec.start(); } catch(_) {} } // unrecognised — keep listening
    };
    rec.onerror = () => {};
    rec.onend   = () => { if (document.getElementById('welcome-overlay')) { try { rec.start(); } catch(_) {} } };
    try { rec.start(); } catch(_) {}
  }
})();

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
      if (data.type === 'speak')         handleSpeak(data);
      else if (data.type === 'narration')       handleNarration(data);
      else if (data.type === 'answer')       handleAnswer(data);
      else if (data.type === 'reading')      handleReading(data);
      else if (data.type === 'system')       handleSystem(data);
      else if (data.type === 'init')         handleInit(data);
      else if (data.type === 'found_object') handleFoundObject(data);
      else if (data.type === 'find_prompt')  handleFindPrompt(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };
}
connectWS();

// Conversation panel is always visible — answers accumulate as chat bubbles
showPanel(document.getElementById('conversation-panel'));

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
    overlay.update(data.detections, data.frame_w || 640, data.frame_h || 480);
  } else if (!overlayActive && !demoPresentationActive) {
    overlay.clear();
  }
}

function handleAnswer(data) {
  const answer = data.answer || 'No answer received.';
  const src    = data.input_source || lastInputSource;
  lastInputSource = src;

  setBanner(answer, 3);
  if (data.fps) updateFps(data.fps);

  if (data.question && data.answer) {
    addConversationTurn(data.question, data.answer, src);
  }

  overlay.clear();

  if (window.currentMode === 'ASK' && src !== 'voice') {
    setTimeout(() => document.getElementById('ask-input')?.focus(), 200);
  }

  // FIND mode: after answer, reset capture state to idle for next capture
  if (window.currentMode === 'FIND') {
    findCaptureState = 'idle';
    setFindCaptureBanner('idle');
  }
}

function handleReading(data) {
  if (data.text) setBanner(data.text, 0);
  if (data.fps)  updateFps(data.fps);
  if (data.detections) updateDetectivePanel(data.detections, data.fps);

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

// Browser TTS — called when server relays {"type":"speak","text":"..."}
function handleSpeak(data) {
  if (!data.text) return;
  try {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(data.text);
    window.speechSynthesis.speak(utt);
  } catch (e) {
    console.warn('speechSynthesis error:', e);
  }
}

function handleFoundObject(data) {
  document.getElementById('find-banner')?.classList.add('hidden');
  window.playSuccess?.();
  if (data.text) {
    showToast(data.text, 3500);
    setBanner(data.text, 3);
  }
}

// FIND capture state machine — driven by backend prompts
function handleFindPrompt(data) {
  const state = data.state;  // 'confirming' | 'captured' | 'answered'
  findCaptureState = state;
  setFindCaptureBanner(state, data.text);
}

// ─── FIND capture banner helper ───────────────────────────────────
function setFindCaptureBanner(state, text) {
  const banner = document.getElementById('find-capture-banner');
  const label  = document.getElementById('find-capture-label');
  if (!banner) return;
  if (state === 'idle') {
    banner.classList.add('hidden');
  } else {
    banner.classList.remove('hidden');
    if (label) label.textContent = text || (
      state === 'confirming'       ? 'Shall I capture what\'s in front of me?' :
      state === 'captured'         ? 'What would you like to know?' :
      ''
    );
  }
}

// ─── UI helpers ───────────────────────────────────────────────────
function setBanner(text, severity) {
  document.getElementById('guidance-text').textContent = text;
  const banner = document.getElementById('guidance-banner');
  const sev    = Math.min(severity, 3);
  banner.className = `guidance-banner sev${sev}`;
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

  document.querySelectorAll('.mode-btn').forEach(btn => {
    const isActive = btn.dataset.mode === current;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });

  // Show ask-row only in ASK mode
  document.getElementById('ask-row').classList.toggle('hidden', current !== 'ASK');
  // Show find-row only in FIND mode
  document.getElementById('find-row')?.classList.toggle('hidden', current !== 'FIND');

  if (current === 'ASK') {
    setTimeout(() => document.getElementById('ask-input')?.focus(), 120);
  }

  if (current === 'FIND') {
    // Reset capture state when entering FIND
    findCaptureState = 'idle';
    setFindCaptureBanner('idle');
    showToast('FIND — tap 📸 or say "capture" to capture scene');
  }

  // Start/stop camera + capture timers based on new mode
  onModeChange(current);
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
      showToast('NAVIGATE — danger alerts only');
    } else if (mode === 'READ') {
      showToast('READ — point camera at text');
    } else if (mode === 'FIND') {
      showToast('FIND — capture scene then ask');
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

  showPanel(document.getElementById('conversation-panel'));
  showToast(`Asking: "${q}"`);
}
document.getElementById('btn-send').addEventListener('click', submitQuestion);
document.getElementById('ask-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitQuestion();
});

// ─── FIND capture flow (UI) ───────────────────────────────────────
document.getElementById('btn-find-capture')?.addEventListener('click', () => {
  sendCommand({ type: 'command', action: 'find_start_capture' });
  showToast('Preparing capture...');
});

document.getElementById('btn-find-capture-yes')?.addEventListener('click', () => {
  if (window.triggerFindCapture) {
    window.triggerFindCapture();
  } else {
    window.sendCommand?.({ type: 'command', action: 'find_capture' });
  }
  showToast('Capturing...');
});

document.getElementById('btn-find-send')?.addEventListener('click', () => {
  const input = document.getElementById('find-question-input');
  const q = input?.value.trim();
  if (!q) return;
  sendCommand({ type: 'command', action: 'find_question', question: q });
  input.value = '';
  showToast(`Asking: "${q}"`);
});

document.getElementById('find-question-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-find-send')?.click();
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

document.getElementById('btn-settings-close').addEventListener('click', () => {
  hidePanel(settingsPanel);
});

document.addEventListener('click', (e) => {
  if (!settingsPanel.classList.contains('hidden') &&
      !settingsPanel.contains(e.target) &&
      e.target.id !== 'btn-settings') {
    hidePanel(settingsPanel);
  }
});

// Camera facing-mode selector
document.getElementById('btn-apply-camera')?.addEventListener('click', () => {
  const facing = document.getElementById('camera-facing')?.value || 'environment';
  // Restart camera with new facing mode
  window.stopCamera?.();
  // Override facingMode by passing via a temporary global then re-starting
  window._preferredFacingMode = facing;
  window.startCamera?.();
  hidePanel(settingsPanel);
  showToast(`Camera switched to ${facing === 'environment' ? 'back' : 'front'}`);
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
    if (!overlayActive) {
      overlayActive = true;
      document.getElementById('btn-overlay').classList.add('active');
      sendCommand({ type: 'command', action: 'toggle_overlay' });
    }
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

  while (msgs.children.length > MAX_CONVO_TURNS * 2) {
    msgs.removeChild(msgs.firstChild);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

const _readHistory = new Set();
function addReadingToConversation(text) {
  if (_readHistory.has(text)) return;
  _readHistory.add(text);

  const panel = document.getElementById('conversation-panel');
  const msgs  = document.getElementById('convo-messages');

  if (window.currentMode === 'READ') showPanel(panel);

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
  sendCommand({ type: 'command', action: 'clear_history' });
});

