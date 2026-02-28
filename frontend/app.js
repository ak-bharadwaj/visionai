// app.js — main application logic
// Requires: overlay (from overlay.js), camera (from camera.js), audio (from audio.js)

// ─── Panel helpers (defined first — used before DOMContentLoaded) ─
function showPanel(el) {
  if (!el) return;
  el.classList.remove('hidden');
  el.classList.add('panel-entering');
  el.addEventListener('animationend', () => el.classList.remove('panel-entering'), { once: true });
}
function hidePanel(el) {
  if (!el) return;
  el.classList.add('hidden');
}

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
// Google-level: Enable overlay by default so bounding boxes always show
let overlayActive = true;  // Changed from false to true - boxes show by default
let detectivePanelActive = false;
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

// Adaptive NAVIGATE rate: burst when danger present, relax when clear
const NAV_INTERVAL_DANGER = 400;   // ms — fast burst when hazard detected
const NAV_INTERVAL_NORMAL = 800;   // ms — standard scan rate
const NAV_INTERVAL_CLEAR = 1500;  // ms — relax when path clear for 3+ frames
let _navClearCount = 0;          // consecutive clear frames

function _navInterval() {
  const mult = _batterySaver ? 2 : 1;
  if (_navClearCount >= 3) return NAV_INTERVAL_CLEAR * mult;
  if (_navClearCount === 0) return NAV_INTERVAL_DANGER * mult;
  return NAV_INTERVAL_NORMAL * mult;
}

function startNavigateCapture() {
  stopCapture();
  function tick() {
    if (window.currentMode !== 'NAVIGATE') return;
    window.sendFrameToBackend?.('NAVIGATE');
    captureTimer = setTimeout(tick, _navInterval());
  }
  captureTimer = setTimeout(tick, NAV_INTERVAL_NORMAL);
}

function startReadCapture() {
  stopCapture();
  captureTimer = setInterval(() => {
    if (window.currentMode !== 'READ') return;
    window.sendFrameToBackend?.('READ');
  }, 1500);
}

function startScanCapture() {
  stopCapture();
  captureTimer = setInterval(() => {
    if (window.currentMode !== 'SCAN') return;
    window.sendFrameToBackend?.('SCAN');
  }, 800);
}

function stopCapture() {
  if (captureTimer) {
    clearInterval(captureTimer);
    clearTimeout(captureTimer);
    captureTimer = null;
  }
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
  } else if (newMode === 'SCAN') {
    window.startCamera?.();
    startScanCapture();
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

// ─── Battery-aware capture rate ───────────────────────────────────
// When battery < 20%, double all capture intervals to conserve power.
let _batterySaver = false;
if (navigator.getBattery) {
  navigator.getBattery().then(bat => {
    function _checkBat() {
      const low = bat.level < 0.20 && !bat.charging;
      if (low !== _batterySaver) {
        _batterySaver = low;
        if (low) {
          showToast('Battery low — reducing scan rate to save power');
        }
        // Restart current capture timer with updated rate
        if (window.currentMode === 'NAVIGATE') startNavigateCapture();
        else if (window.currentMode === 'READ') startReadCapture();
        else if (window.currentMode === 'SCAN') startScanCapture();
      }
    }
    bat.addEventListener('levelchange', _checkBat);
    bat.addEventListener('chargingchange', _checkBat);
    _checkBat();
  }).catch(() => { });
}

// ─── Ollama offline detection ─────────────────────────────────────
// If a 'system' message mentions Ollama not running, surface it clearly.
function _checkOllamaOffline(text) {
  if (!text) return;
  const lower = text.toLowerCase();
  if (lower.includes('ollama') && (lower.includes('not running') || lower.includes('offline'))) {
    const banner = document.getElementById('guidance-banner');
    if (banner) {
      banner.className = 'guidance-banner sev1';
      document.getElementById('guidance-text').textContent =
        'AI offline — run: ollama serve (then: ollama pull phi3:mini)';
    }
    showToast('Ollama not running — AI answers unavailable', 6000);
  }
}
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
      else if (t.includes('find')) pickMode('FIND');
      else if (t.includes('read') || t.includes('ocr')) pickMode('OCR');
      else if (t.includes('scan')) pickMode('SCAN');
      else { try { rec.start(); } catch (_) { } } // unrecognised — keep listening
    };
    rec.onerror = () => { };
    rec.onend = () => { if (document.getElementById('welcome-overlay')) { try { rec.start(); } catch (_) { } } };
    try { rec.start(); } catch (_) { }
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
      if (data.type === 'speak') handleSpeak(data);
      else if (data.type === 'narration') handleNarration(data);
      else if (data.type === 'answer') handleAnswer(data);
      else if (data.type === 'reading') handleReading(data);
      else if (data.type === 'system') handleSystem(data);
      else if (data.type === 'init') handleInit(data);
      else if (data.type === 'found_object') handleFoundObject(data);
      else if (data.type === 'find_prompt') handleFindPrompt(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };
}
connectWS();

// Conversation panel is hidden by default until a turn occurs

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
  // Google-level: Update nav state for destination input
  if (data.nav_state) window._navState = data.nav_state;
  if (data.nav_destination !== undefined) window._navDestination = data.nav_destination || '';
}

// Google-level: Enhanced narration handler with better bounding box support
function handleNarration(data) {
  if (data.text) setBanner(data.text, data.severity || 0);
  if (data.fps) updateFps(data.fps);
  updateDetectivePanel(data.detections, data.fps);
  
  // Google-level: Ensure frame dimensions are always provided
  const frame_w = data.frame_w || 640;
  const frame_h = data.frame_h || 480;
  
  // Google-level: Validate and fix bounding boxes before rendering
  if (data.detections && Array.isArray(data.detections)) {
    data.detections = data.detections.map(det => {
      // Ensure all required fields exist
      if (!det.x1 && det.x1 !== 0) det.x1 = 0;
      if (!det.y1 && det.y1 !== 0) det.y1 = 0;
      if (!det.x2 && det.x2 !== 0) det.x2 = frame_w;
      if (!det.y2 && det.y2 !== 0) det.y2 = frame_h;
      
      // Ensure bbox is valid
      det.x1 = Math.max(0, Math.min(frame_w - 1, det.x1));
      det.y1 = Math.max(0, Math.min(frame_h - 1, det.y1));
      det.x2 = Math.max(det.x1 + 1, Math.min(frame_w, det.x2));
      det.y2 = Math.max(det.y1 + 1, Math.min(frame_h, det.y2));
      
      // Ensure distance_level exists
      if (!det.distance_level) det.distance_level = 3;
      if (!det.confidence) det.confidence = 0.5;
      if (!det.distance) det.distance = "unknown";
      
      return det;
    });
  }
  
  // Google-level: Always show bounding boxes when detections exist (unless explicitly disabled)
  if (data.detections && data.detections.length > 0) {
    // Always update overlay if we have detections (overlayActive is true by default)
    if (overlayActive || demoPresentationActive) {
      overlay.update(data.detections, frame_w, frame_h);
    } else {
      // Even if overlay is off, show boxes in NAVIGATE mode for safety
      if (window.currentMode === 'NAVIGATE' || window.currentMode === 'FIND') {
        overlay.update(data.detections, frame_w, frame_h);
      } else {
        overlay.clear();
      }
    }
  } else {
    // Only clear if overlay is explicitly off
    if (!overlayActive && !demoPresentationActive) {
      overlay.clear();
    }
  }
  // Always draw nav arrow in NAVIGATE mode when dangerous objects are present
  if (window.currentMode === 'NAVIGATE' && data.detections?.length) {
    const dangerous = data.detections.filter(d => d.distance_level <= 2);
    if (dangerous.length) {
      overlay.drawNavArrow(dangerous, data.frame_w || 640, data.frame_h || 480);
      // Burst mode — reset clear counter
      _navClearCount = 0;
      // Haptic: short pulse on warning, long on danger
      const maxSev = Math.min(...dangerous.map(d => d.distance_level));
      if (navigator.vibrate) {
        navigator.vibrate(maxSev === 1 ? [200, 50, 200] : [80]);
      }
    } else {
      _navClearCount = Math.min(_navClearCount + 1, 5);
    }
  } else if (window.currentMode === 'NAVIGATE') {
    _navClearCount = Math.min(_navClearCount + 1, 5);
  }
}

function handleAnswer(data) {
  const answer = data.answer || 'No answer received.';
  const src = data.input_source || lastInputSource;
  lastInputSource = src;

  setBanner(answer, 3);
  if (data.fps) updateFps(data.fps);
  _checkOllamaOffline(answer);

  if (data.question && data.answer) {
    // For voice input: question was not shown optimistically, so add full turn
    // For chat input: question bubble already shown — only append the answer
    if (src === 'voice') {
      addConversationTurn(data.question, data.answer, src);
    } else {
      addAnswerBubble(data.answer);
    }
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
  if (data.fps) updateFps(data.fps);
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
    _checkOllamaOffline(data.text);
  }
}

// Browser TTS — called when server relays {"type":"speak","text":"..."}
// Google-level: Enhanced TTS with better reliability and error handling
function handleSpeak(data) {
  if (!data.text || !data.text.trim()) return;
  
  try {
    // Cancel any ongoing speech
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    
    // Wait a brief moment to ensure cancellation completes
    setTimeout(() => {
      try {
        if (!window.speechSynthesis) {
          console.error('[TTS] speechSynthesis not available');
          return;
        }
        
        const utt = new SpeechSynthesisUtterance(data.text.trim());
        
        // Google-level: Enhanced TTS settings for better quality
        utt.rate = 1.0;  // Normal speed
        utt.pitch = 1.0; // Normal pitch
        utt.volume = 1.0; // Full volume
        
        // Error handling
        utt.onerror = (e) => {
          console.error('[TTS] Speech error:', e);
          // Fallback: show text in console
          console.log('[TTS Fallback]', data.text);
        };
        
        utt.onstart = () => {
          console.debug('[TTS] Speaking:', data.text.substring(0, 50));
        };
        
        utt.onend = () => {
          console.debug('[TTS] Finished speaking');
        };
        
        // Speak with error recovery
        try {
          window.speechSynthesis.speak(utt);
        } catch (speakError) {
          console.error('[TTS] Speak error:', speakError);
          // Fallback: try again after short delay
          setTimeout(() => {
            try {
              window.speechSynthesis.speak(utt);
            } catch (retryError) {
              console.error('[TTS] Retry failed:', retryError);
            }
          }, 100);
        }
      } catch (e) {
        console.error('[TTS] Setup error:', e);
      }
    }, 50);
  } catch (e) {
    console.error('[TTS] Fatal error:', e);
  }
}

// Initialize TTS on page load (required for some browsers)
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    // Warm up speechSynthesis
    if (window.speechSynthesis) {
      const warmup = new SpeechSynthesisUtterance('');
      warmup.volume = 0;
      try {
        window.speechSynthesis.speak(warmup);
        window.speechSynthesis.cancel();
        console.debug('[TTS] Initialized and warmed up');
      } catch (e) {
        console.warn('[TTS] Warmup failed:', e);
      }
    }
  });
}

function handleFoundObject(data) {
  document.getElementById('find-banner')?.classList.add('hidden');
  window.playSuccess?.();
  if (data.text) {
    showToast(data.text, 3500);
    setBanner(data.text, 3);
  }
  // Draw animated target ring on the detected object
  if (data.detection) {
    overlay.drawFindTarget(data.detection, data.frame_w || 640, data.frame_h || 480);
    // Auto-clear the ring after 4 seconds
    setTimeout(() => overlay.clear(), 4000);
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
  const label = document.getElementById('find-capture-label');
  if (!banner) return;
  if (state === 'idle') {
    banner.classList.add('hidden');
  } else {
    banner.classList.remove('hidden');
    if (label) label.textContent = text || (
      state === 'confirming' ? 'Shall I capture what\'s in front of me?' :
        state === 'captured' ? 'What would you like to know?' :
          ''
    );
  }
}

// ─── UI helpers ───────────────────────────────────────────────────
function setBanner(text, severity) {
  document.getElementById('guidance-text').textContent = text;
  const banner = document.getElementById('guidance-banner');
  const sev = Math.min(severity, 3);
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
    } else if (mode === 'SCAN') {
      showToast('SCAN — point at QR code or barcode');
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

  // Force capture so the backend LLM has an image to process
  window.sendFrameToBackend?.('ASK');

  // Show the question bubble immediately (optimistic) and a "thinking" placeholder
  addQuestionBubble(q, 'chat');

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

  // Send current frame just in case user didn't capture explicitly
  window.sendFrameToBackend?.('FIND');

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

/** Show just the question bubble immediately (optimistic display). */
function addQuestionBubble(question, inputSource) {
  const panel = document.getElementById('conversation-panel');
  const msgs = document.getElementById('convo-messages');
  showPanel(panel);

  const qEl = document.createElement('div');
  qEl.className = inputSource === 'voice' ? 'bubble-q bubble-q-voice' : 'bubble-q';
  qEl.textContent = question;
  msgs.appendChild(qEl);

  // Add a transient "thinking" placeholder
  const thinkEl = document.createElement('div');
  thinkEl.className = 'bubble-a bubble-thinking';
  thinkEl.textContent = '…';
  thinkEl.setAttribute('data-thinking', '1');
  msgs.appendChild(thinkEl);

  while (msgs.children.length > MAX_CONVO_TURNS * 2 + 2) {
    msgs.removeChild(msgs.firstChild);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

/** Replace the thinking placeholder with the real answer bubble. */
function addAnswerBubble(answer) {
  const panel = document.getElementById('conversation-panel');
  const msgs = document.getElementById('convo-messages');
  showPanel(panel);

  // Remove existing thinking placeholder(s)
  msgs.querySelectorAll('[data-thinking]').forEach(el => el.remove());

  const aEl = document.createElement('div');
  aEl.className = 'bubble-a';
  aEl.textContent = answer;
  msgs.appendChild(aEl);

  while (msgs.children.length > MAX_CONVO_TURNS * 2) {
    msgs.removeChild(msgs.firstChild);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

/** Full turn (question + answer together) — used for voice answers. */
function addConversationTurn(question, answer, inputSource) {
  const panel = document.getElementById('conversation-panel');
  const msgs = document.getElementById('convo-messages');
  showPanel(panel);

  const qEl = document.createElement('div');
  qEl.className = inputSource === 'voice' ? 'bubble-q bubble-q-voice' : 'bubble-q';
  qEl.textContent = question;

  const aEl = document.createElement('div');
  aEl.className = 'bubble-a';
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
  const msgs = document.getElementById('convo-messages');

  if (window.currentMode === 'READ') showPanel(panel);

  const el = document.createElement('div');
  el.className = 'bubble-read';
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

