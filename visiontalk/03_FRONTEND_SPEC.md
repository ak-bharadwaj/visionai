# VisionTalk — Complete Professional Frontend Specification
## All 9 files. Exact code. Professional UI. All features.

---

## UI LAYOUT (visual wireframe)

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER                                                      │
│  👁 VisionTalk     ● LIVE  [NAVIGATE badge]   ⚙            │
├─────────────────────────────────────────────────────────────┤
│ CAMERA + CANVAS OVERLAY                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  📷 Live Camera Feed                                  │  │
│  │  Canvas bounding boxes drawn on top                   │  │
│  │  Zone lines: LEFT | AHEAD | RIGHT                     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│ GUIDANCE BANNER                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  📢 "Chair nearby, to your left"                      │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│ 3 MODE BUTTONS                                              │
│  [🧭 NAVIGATE]  [🧠 ASK]  [📖 READ]                       │
├─────────────────────────────────────────────────────────────┤
│ ASK ROW (hidden unless ASK mode)                            │
│  [🎙 Hold to Speak]  [...type question...]  [➤ Send]       │
├─────────────────────────────────────────────────────────────┤
│ SECONDARY CONTROLS                                          │
│  [👁 Overlay]  [📸 Remember]  [❓ What Changed?]           │
├─────────────────────────────────────────────────────────────┤
│ DEMO CONTROLS (for judges)                                  │
│  [🎬 Demo Mode]  [🔬 Show Data]                            │
├─────────────────────────────────────────────────────────────┤
│ DETECTIVE PANEL (hidden, toggled by 🔬)                     │
│  Object | Distance | Dir | Conf | BBox                     │
│  chair  | nearby   | left| 89%  | [142,241]→[318,479]      │
├─────────────────────────────────────────────────────────────┤
│ CONVERSATION PANEL (auto-shows in ASK mode)                 │
│                         You: "What is in front?"   [bubble]│
│  🧠 "I can see a chair to your left."  [bubble]            │
├─────────────────────────────────────────────────────────────┤
│ SETTINGS PANEL (hidden, toggled by ⚙)                       │
│  Phone IP: [192.168.1.x:8080  ] [Apply]                    │
│  [QR Code image]                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## FILE 1: `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <meta name="description" content="VisionTalk — offline AI spatial assistant for the visually impaired">
  <title>VisionTalk</title>
  <link rel="manifest" href="/static/manifest.json">
  <meta name="theme-color" content="#0f0f13">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>

  <!-- ─── HEADER ─────────────────────────────────────────────── -->
  <header class="app-header">
    <div class="header-brand">
      <span class="brand-icon">👁</span>
      <span class="brand-name">VisionTalk</span>
    </div>
    <div class="header-status">
      <span id="ws-dot" class="ws-dot disconnected"></span>
      <span id="ws-label" class="ws-label">Connecting...</span>
      <span id="mode-badge" class="mode-badge">NAVIGATE</span>
      <span id="fps-display" class="fps-display">-- fps</span>
    </div>
    <button id="btn-settings" class="icon-btn" aria-label="Settings">⚙</button>
  </header>

  <!-- ─── SETTINGS PANEL ────────────────────────────────────── -->
  <div id="settings-panel" class="settings-panel hidden">
    <div class="settings-inner">
      <h3 class="settings-title">📡 Camera Setup</h3>
      <p class="settings-hint">Install <strong>IP Webcam</strong> app on Android. Enter IP below.</p>
      <div class="ip-row">
        <input id="ip-input" class="ip-input" type="text"
               placeholder="192.168.1.x:8080"
               value="">
        <button id="btn-apply-ip" class="apply-btn">Apply</button>
      </div>
      <div class="qr-section">
        <p class="settings-hint">Scan to open VisionTalk on phone:</p>
        <img id="qr-img" src="/qr" alt="QR Code" class="qr-img" loading="lazy">
      </div>
    </div>
  </div>

  <!-- ─── MAIN ──────────────────────────────────────────────── -->
  <main class="app-main">

    <!-- Camera + Canvas Overlay -->
    <div class="camera-wrap" id="camera-wrap">
      <img id="camera-feed" class="camera-feed" alt="Camera feed"
           src="" draggable="false">
      <canvas id="overlay-canvas" class="overlay-canvas"></canvas>
      <div id="camera-placeholder" class="camera-placeholder">
        <div class="placeholder-icon">📷</div>
        <p class="placeholder-text">Waiting for camera...</p>
        <p class="placeholder-hint">Set phone IP in ⚙ settings</p>
      </div>
    </div>

    <!-- Guidance Banner -->
    <div id="guidance-banner" class="guidance-banner sev0">
      <span id="guidance-text" class="guidance-text">
        Initializing VisionTalk...
      </span>
    </div>

    <!-- Mode Buttons -->
    <section class="mode-controls" aria-label="Mode selection">
      <button id="btn-navigate" class="mode-btn active" data-mode="NAVIGATE">
        <span class="mode-icon">🧭</span>
        <span class="mode-label">NAVIGATE</span>
        <span class="mode-desc">Spatial narrator</span>
      </button>
      <button id="btn-ask" class="mode-btn" data-mode="ASK">
        <span class="mode-icon">🧠</span>
        <span class="mode-label">ASK</span>
        <span class="mode-desc">Voice Q&amp;A</span>
      </button>
      <button id="btn-read" class="mode-btn" data-mode="READ">
        <span class="mode-icon">📖</span>
        <span class="mode-label">READ</span>
        <span class="mode-desc">OCR reader</span>
      </button>
    </section>

    <!-- ASK Row (voice input) — hidden by default -->
    <div id="ask-row" class="ask-row hidden">
      <button id="btn-mic" class="mic-btn" aria-label="Hold to speak">
        🎙 Hold to Speak
      </button>
      <input id="ask-input" class="ask-input" type="text"
             placeholder="Or type your question..."
             autocomplete="off" autocorrect="off">
      <button id="btn-send" class="send-btn" aria-label="Send question">➤</button>
    </div>

    <!-- Secondary Controls -->
    <section class="secondary-controls" aria-label="Extra controls">
      <button id="btn-overlay" class="secondary-btn">
        <span>👁</span> Overlay
      </button>
      <button id="btn-snapshot" class="secondary-btn">
        <span>📸</span> Remember
      </button>
      <button id="btn-diff" class="secondary-btn">
        <span>❓</span> What Changed?
      </button>
    </section>

    <!-- Demo Controls -->
    <section class="demo-controls" aria-label="Demo controls for judges">
      <button id="btn-demo-mode" class="demo-btn">
        <span class="demo-btn-icon">🎬</span>
        <span class="demo-btn-label">Demo Mode</span>
        <span class="demo-btn-hint">For judges — forces overlay + big text</span>
      </button>
      <button id="btn-detective" class="demo-btn">
        <span class="demo-btn-icon">🔬</span>
        <span class="demo-btn-label">Show Data</span>
        <span class="demo-btn-hint">Live AI detection proof</span>
      </button>
    </section>

    <!-- Detective Panel -->
    <div id="detective-panel" class="detective-panel hidden">
      <div class="detective-header">
        <span>🔬 Live AI Detection Data</span>
        <span id="det-fps" class="det-fps">FPS: --</span>
      </div>
      <div id="detective-table">
        <p class="detective-empty">No detections yet...</p>
      </div>
    </div>

    <!-- Conversation Panel -->
    <div id="conversation-panel" class="conversation-panel hidden">
      <div class="convo-header">
        <span>💬 AI Conversation</span>
        <button id="btn-clear-convo" class="convo-clear">✕ Clear</button>
      </div>
      <div id="convo-messages" class="convo-messages"></div>
    </div>

  </main>

  <!-- Toast -->
  <div id="toast" class="toast hidden"></div>

  <!-- Scripts -->
  <script src="/static/camera.js"></script>
  <script src="/static/overlay.js"></script>
  <script src="/static/audio.js"></script>
  <script src="/static/app.js"></script>
  <script src="/static/voice.js"></script>
</body>
</html>
```

---

## FILE 2: `frontend/style.css`

```css
/* ── Variables ──────────────────────────────────────── */
:root {
  --bg:          #0f0f13;
  --bg-surface:  #16161d;
  --bg-card:     #1c1c26;
  --border:      rgba(255,255,255,0.08);
  --purple:      #6c63ff;
  --purple-glow: rgba(108,99,255,0.35);
  --blue:        #0ea5e9;
  --green:       #10b981;
  --orange:      #f97316;
  --red:         #ef4444;
  --yellow:      #f59e0b;
  --text:        #e8e8f0;
  --text-muted:  #6b7280;
  --radius:      12px;
  --radius-sm:   8px;
  --transition:  0.2s ease;
  --shadow:      0 4px 24px rgba(0,0,0,0.4);
}

/* ── Reset ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; -webkit-tap-highlight-color: transparent; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100dvh;
  display: flex; flex-direction: column;
  overflow-x: hidden;
}
button {
  cursor: pointer; border: none; background: none;
  font-family: inherit; font-size: inherit;
  transition: all var(--transition);
}
input {
  font-family: inherit; font-size: inherit;
  border: none; outline: none; background: none; color: inherit;
}

/* ── Header ─────────────────────────────────────────── */
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px;
  background: rgba(22,22,29,0.85);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 100;
}
.header-brand { display: flex; align-items: center; gap: 8px; }
.brand-icon   { font-size: 22px; }
.brand-name   { font-size: 18px; font-weight: 800; letter-spacing: -0.5px;
                background: linear-gradient(135deg, var(--purple), var(--blue));
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header-status { display: flex; align-items: center; gap: 8px; flex: 1; justify-content: center; }

.ws-dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--red); flex-shrink: 0;
  transition: background var(--transition);
}
.ws-dot.connected    { background: var(--green); box-shadow: 0 0 8px var(--green); }
.ws-dot.disconnected { background: var(--red); }

.ws-label     { font-size: 11px; color: var(--text-muted); }
.mode-badge   {
  font-size: 11px; font-weight: 700; letter-spacing: 1px;
  padding: 3px 8px; border-radius: 20px;
  background: rgba(108,99,255,0.2);
  border: 1px solid rgba(108,99,255,0.4);
  color: var(--purple);
}
.fps-display  { font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.icon-btn     {
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--bg-card); border: 1px solid var(--border);
  font-size: 16px; display: flex; align-items: center; justify-content: center;
}
.icon-btn:hover { background: var(--bg-surface); border-color: var(--purple); }

/* ── Settings Panel ─────────────────────────────────── */
.settings-panel {
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
  padding: 16px;
}
.settings-panel.hidden { display: none; }
.settings-inner  { max-width: 480px; margin: 0 auto; }
.settings-title  { font-size: 14px; font-weight: 700; margin-bottom: 8px; color: var(--purple); }
.settings-hint   { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
.ip-row          { display: flex; gap: 8px; margin-bottom: 16px; }
.ip-input        {
  flex: 1; background: var(--bg-card);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 8px 12px; font-size: 14px; color: var(--text);
}
.ip-input:focus  { border-color: var(--purple); }
.apply-btn       {
  background: var(--purple); color: #fff;
  padding: 8px 16px; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600;
}
.apply-btn:hover { opacity: 0.9; }
.qr-section      { text-align: center; }
.qr-img          { width: 140px; height: 140px; border-radius: 10px; margin-top: 8px; }

/* ── Main ────────────────────────────────────────────── */
.app-main {
  flex: 1; display: flex; flex-direction: column; gap: 10px;
  padding: 12px; max-width: 600px; width: 100%; margin: 0 auto;
}

/* ── Camera ─────────────────────────────────────────── */
.camera-wrap {
  position: relative; width: 100%; border-radius: var(--radius);
  overflow: hidden; background: #000;
  aspect-ratio: 4/3;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.camera-feed {
  width: 100%; height: 100%; object-fit: cover; display: block;
}
.overlay-canvas {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
.camera-placeholder {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 8px;
  background: var(--bg-card);
}
.camera-placeholder.hidden { display: none; }
.placeholder-icon { font-size: 40px; opacity: 0.4; }
.placeholder-text { font-size: 14px; color: var(--text-muted); }
.placeholder-hint { font-size: 11px; color: var(--text-muted); opacity: 0.6; }

/* ── Guidance Banner ────────────────────────────────── */
.guidance-banner {
  border-radius: var(--radius);
  padding: 14px 18px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  min-height: 58px;
  display: flex; align-items: center;
  transition: border-color var(--transition), background var(--transition);
}
.guidance-text {
  font-size: 16px; font-weight: 600; letter-spacing: -0.2px; line-height: 1.4;
}
.guidance-banner.sev0 { border-color: var(--border); }
.guidance-banner.sev1 { border-color: var(--red);    background: rgba(239,68,68,0.1); }
.guidance-banner.sev2 { border-color: var(--orange); background: rgba(249,115,22,0.08); }
.guidance-banner.sev3 { border-color: var(--green);  background: rgba(16,185,129,0.08); }

/* Demo presentation mode: big text */
body.demo-presentation .guidance-text  { font-size: 22px; }
body.demo-presentation .guidance-banner { min-height: 80px; }

/* ── Mode Controls ──────────────────────────────────── */
.mode-controls {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
}
.mode-btn {
  display: flex; flex-direction: column; align-items: center;
  gap: 2px; padding: 12px 6px;
  background: var(--bg-card);
  border: 2px solid var(--border);
  border-radius: var(--radius);
  transition: all var(--transition);
}
.mode-btn:active { transform: scale(0.96); }
.mode-icon  { font-size: 22px; }
.mode-label { font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.mode-desc  { font-size: 9px; color: var(--text-muted); }

.mode-btn[data-mode="NAVIGATE"].active {
  border-color: var(--purple); background: rgba(108,99,255,0.12);
  box-shadow: 0 0 20px var(--purple-glow);
}
.mode-btn[data-mode="ASK"].active {
  border-color: var(--blue); background: rgba(14,165,233,0.1);
  box-shadow: 0 0 16px rgba(14,165,233,0.25);
}
.mode-btn[data-mode="READ"].active {
  border-color: var(--green); background: rgba(16,185,129,0.1);
  box-shadow: 0 0 16px rgba(16,185,129,0.25);
}

/* ── ASK Row ────────────────────────────────────────── */
.ask-row {
  display: flex; gap: 8px; align-items: center;
}
.ask-row.hidden { display: none; }
.mic-btn {
  flex-shrink: 0; padding: 10px 14px;
  background: var(--bg-card);
  border: 2px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600;
  white-space: nowrap;
}
.mic-btn.listening {
  border-color: var(--red); background: rgba(239,68,68,0.1);
  color: var(--red); animation: pulse 1.5s infinite;
}
.ask-input {
  flex: 1; background: var(--bg-card);
  border: 2px solid var(--border); border-radius: var(--radius-sm);
  padding: 10px 12px; font-size: 14px;
}
.ask-input:focus { border-color: var(--blue); }
.send-btn {
  width: 42px; height: 42px; border-radius: var(--radius-sm);
  background: var(--blue); color: #fff;
  font-size: 16px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
}
.send-btn:hover { opacity: 0.9; }

/* ── Secondary Controls ─────────────────────────────── */
.secondary-controls {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
}
.secondary-btn {
  display: flex; align-items: center; justify-content: center; gap: 5px;
  padding: 10px 6px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 12px; font-weight: 500; color: var(--text-muted);
}
.secondary-btn:hover { border-color: var(--purple); color: var(--text); }
.secondary-btn.active {
  border-color: var(--purple); color: var(--purple);
  background: rgba(108,99,255,0.1);
}

/* ── Demo Controls ──────────────────────────────────── */
.demo-controls {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.demo-btn {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  padding: 10px 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  transition: all var(--transition);
}
.demo-btn-icon  { font-size: 18px; }
.demo-btn-label { font-size: 12px; font-weight: 600; }
.demo-btn-hint  { font-size: 9px; color: var(--text-muted); text-align: center; }
.demo-btn.active {
  border-color: var(--orange); color: var(--orange);
  background: rgba(249,115,22,0.1);
  box-shadow: 0 0 12px rgba(249,115,22,0.2);
}

/* ── Detective Panel ────────────────────────────────── */
.detective-panel {
  background: rgba(0,0,0,0.6);
  border: 1px solid rgba(108,99,255,0.3);
  border-radius: var(--radius-sm);
  padding: 10px;
  font-family: 'Courier New', monospace;
}
.detective-panel.hidden { display: none; }
.detective-header {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 12px; font-weight: 700; color: var(--purple);
  margin-bottom: 8px;
}
.det-fps { color: var(--green); }
#detective-table        { overflow-x: auto; }
#detective-table table  { width: 100%; border-collapse: collapse; font-size: 10px; }
#detective-table th     {
  color: var(--purple); font-weight: 700;
  padding: 4px 6px; text-align: left;
  border-bottom: 1px solid var(--border);
}
#detective-table td     { padding: 3px 6px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.det-lv1                { color: var(--red)    !important; font-weight: 700; }
.det-lv2                { color: var(--orange) !important; }
.det-lv3, .det-lv4      { color: var(--green)  !important; }
.detective-empty        { font-size: 11px; color: var(--text-muted); padding: 6px; }

/* ── Conversation Panel ─────────────────────────────── */
.conversation-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden; max-height: 270px;
  display: flex; flex-direction: column;
}
.conversation-panel.hidden { display: none; }
.convo-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 14px;
  font-size: 12px; font-weight: 700; color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  background: var(--bg-surface); flex-shrink: 0;
}
.convo-clear {
  font-size: 11px; color: var(--text-muted); padding: 2px 6px;
  border-radius: 4px; border: 1px solid var(--border);
}
.convo-messages {
  flex: 1; overflow-y: auto; padding: 10px;
  display: flex; flex-direction: column; gap: 8px;
  scroll-behavior: smooth;
}
.bubble-q {
  align-self: flex-end; max-width: 85%;
  background: var(--blue); color: #fff;
  padding: 8px 12px; border-radius: 14px 14px 2px 14px;
  font-size: 13px; animation: fadeSlideIn 0.3s ease;
}
.bubble-q::before { content: "You  "; font-size: 10px; opacity: 0.75; }
.bubble-a {
  align-self: flex-start; max-width: 85%;
  background: rgba(108,99,255,0.18);
  border: 1px solid rgba(108,99,255,0.3); color: var(--text);
  padding: 8px 12px; border-radius: 14px 14px 14px 2px;
  font-size: 13px; animation: fadeSlideIn 0.3s ease;
}
.bubble-a::before { content: "🧠 AI  "; font-size: 10px; color: var(--purple); }

/* ── Toast ──────────────────────────────────────────── */
.toast {
  position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
  background: rgba(28,28,38,0.95); color: var(--text);
  padding: 10px 20px; border-radius: 20px;
  font-size: 13px; font-weight: 600;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  z-index: 999; white-space: nowrap;
  animation: toastIn 0.25s ease;
}
.toast.hidden { display: none; }

/* ── Animations ─────────────────────────────────────── */
@keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:0.5} }
@keyframes fadeSlideIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
@keyframes toastIn     { from{opacity:0;transform:translateX(-50%) translateY(8px)} to{opacity:1;transform:translateX(-50%)} }

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar              { width: 4px; }
::-webkit-scrollbar-track        { background: transparent; }
::-webkit-scrollbar-thumb        { background: var(--border); border-radius: 2px; }
```

---

## FILE 3: `frontend/camera.js`

```javascript
// camera.js — MJPEG camera feed from Android IP Webcam app
// Rules: Never getUserMedia(). Camera is always <img src="http://IP/video">

const cameraFeed        = document.getElementById('camera-feed');
const cameraPlaceholder = document.getElementById('camera-placeholder');

const STORAGE_KEY = 'visiontalk_phone_ip';

function applyCameraSource(ip) {
  if (!ip) return;
  localStorage.setItem(STORAGE_KEY, ip);
  const url = ip.startsWith('http') ? ip : `http://${ip}`;
  cameraFeed.src = `${url}/video`;
  cameraPlaceholder.classList.add('hidden');
}

// Restore saved IP on load
(function restoreIp() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    document.getElementById('ip-input').value = saved;
    applyCameraSource(saved);
  }
})();

// Apply button
document.getElementById('btn-apply-ip').addEventListener('click', () => {
  const ip = document.getElementById('ip-input').value.trim();
  if (ip) { applyCameraSource(ip); showToast('📷 Camera source applied'); }
});
document.getElementById('ip-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-apply-ip').click();
});

// Camera error fallback
cameraFeed.addEventListener('error', () => {
  cameraPlaceholder.classList.remove('hidden');
});
cameraFeed.addEventListener('load', () => {
  cameraPlaceholder.classList.add('hidden');
});
```

---

## FILE 4: `frontend/overlay.js`

```javascript
// overlay.js — canvas bounding box renderer
// Canvas must be: position:absolute; inset:0; width:100%; height:100%; pointer-events:none

const canvas  = document.getElementById('overlay-canvas');
const ctx     = canvas.getContext('2d');
const COLORS  = { 1: '#ef4444', 2: '#f97316', 3: '#10b981', 4: '#6b7280' };
const LABELS  = { 1: 'VERY CLOSE', 2: 'NEARBY', 3: 'AHEAD', 4: 'FAR' };

function resizeCanvas() {
  const wrap = document.getElementById('camera-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Scale factor: YOLO inference at 320px, display size varies
function scaleCoord(val, model_dim, display_dim) {
  return (val / model_dim) * display_dim;
}
const MODEL_W = 640, MODEL_H = 480;   // typical camera frame size

const overlay = {
  update(detections) {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this._drawZones();
    detections.forEach(d => this._drawBox(d));
  },
  clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  },
  _drawZones() {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    const w = canvas.width;
    const h = canvas.height;
    // Left | Center | Right zone dividers
    [0.33, 0.67].forEach(ratio => {
      ctx.beginPath();
      ctx.moveTo(w * ratio, 0);
      ctx.lineTo(w * ratio, h);
      ctx.stroke();
    });
    // Zone labels
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('LEFT', w * 0.167, 14);
    ctx.fillText('AHEAD', w * 0.5, 14);
    ctx.fillText('RIGHT', w * 0.833, 14);
  },
  _drawBox(d) {
    const dw = canvas.width, dh = canvas.height;
    const x1 = scaleCoord(d.x1, MODEL_W, dw);
    const y1 = scaleCoord(d.y1, MODEL_H, dh);
    const x2 = scaleCoord(d.x2, MODEL_W, dw);
    const y2 = scaleCoord(d.y2, MODEL_H, dh);
    const color = COLORS[d.distance_level] || '#ffffff';
    const conf  = Math.round(d.confidence * 100);
    const label = `${d.class_name} · ${d.distance} · ${conf}%`;

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Filled label background
    ctx.font = 'bold 11px Inter, sans-serif';
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 20, textW + 10, 20);

    // Label text
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 5, y1 - 6);
  }
};

window.overlay = overlay;
```

---

## FILE 5: `frontend/voice.js`

```javascript
// voice.js — push-to-talk speech recognition
// Requires: window.sendCommand and window.showToast from app.js (loaded first)

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById('btn-mic');

if (!SpeechRecognition) {
  micBtn.textContent = '🎙 (Not supported)';
  micBtn.disabled = true;
} else {
  const rec = new SpeechRecognition();
  rec.lang = 'en-US';
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  let listening = false;

  micBtn.addEventListener('mousedown', startListening);
  micBtn.addEventListener('touchstart', e => { e.preventDefault(); startListening(); });
  micBtn.addEventListener('mouseup', stopListening);
  micBtn.addEventListener('touchend', stopListening);

  function startListening() {
    if (listening) return;
    listening = true;
    micBtn.classList.add('listening');
    micBtn.textContent = '🔴 Listening...';
    rec.start();
  }
  function stopListening() {
    if (!listening) return;
    listening = false;
    micBtn.classList.remove('listening');
    micBtn.textContent = '🎙 Hold to Speak';
    try { rec.stop(); } catch (_) {}
  }

  rec.onresult = (e) => {
    const transcript = e.results[0][0].transcript.trim();
    const lower = transcript.toLowerCase();
    window.showToast?.(`💬 Heard: "${transcript}"`);

    // Mode commands
    if (lower === 'navigate' || lower.includes('navigate mode')) {
      window.sendCommand?.({ type:'command', action:'set_mode', mode:'NAVIGATE' });
      document.getElementById('btn-navigate').click();
    } else if (lower === 'read' || lower.includes('read mode')) {
      window.sendCommand?.({ type:'command', action:'set_mode', mode:'READ' });
      document.getElementById('btn-read').click();
    }
    // Scene memory commands
    else if (lower.includes('remember') || lower === 'remember this') {
      window.sendCommand?.({ type:'command', action:'snapshot' });
      window.showToast?.('📸 Scene snapshot saved');
    } else if (lower.includes('what changed') || lower.includes('what is different')) {
      window.sendCommand?.({ type:'command', action:'scene_diff' });
    }
    // Any other spoken text → sent as ASK question
    else {
      window.sendCommand?.({ type:'command', action:'ask', question: transcript });
      document.getElementById('btn-ask').click();
    }
  };

  rec.onerror = (e) => {
    stopListening();
    if (e.error !== 'no-speech') window.showToast?.(`🎙 Error: ${e.error}`);
  };
  rec.onend = stopListening;
}
```

---

## FILE 6: `frontend/audio.js`

```javascript
// audio.js — stub only. TTS is handled server-side via pyttsx3.
// No browser audio needed for this architecture.
console.debug('[audio.js] TTS is server-side. No browser audio required.');
```

---

## FILE 7: `frontend/app.js`

```javascript
// app.js — main application logic
// Requires: overlay (from overlay.js), showToast (defined here)

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
let overlayActive       = false;
let detectivePanelActive = false;
let demoPresentationActive = false;
window.currentMode = 'NAVIGATE';

// ─── WebSocket ────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/guidance`);

  ws.onopen = () => {
    document.getElementById('ws-dot').className = 'ws-dot connected';
    document.getElementById('ws-label').textContent = 'Live';
    showToast('✅ Connected to VisionTalk');
  };

  ws.onclose = () => {
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
      else if (data.type === 'answer')   handleAnswer(data);
      else if (data.type === 'reading')  handleReading(data);
      else if (data.type === 'system')   handleSystem(data);
      else if (data.type === 'init')     handleInit(data);
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
  if (data.text)     setBanner(data.text, data.severity || 0);
  if (data.fps)      updateFps(data.fps);
  updateDetectivePanel(data.detections, data.fps);
  if ((overlayActive || demoPresentationActive) && data.detections?.length) {
    overlay.update(data.detections);
  } else if (!overlayActive && !demoPresentationActive) {
    overlay.clear();
  }
}

function handleAnswer(data) {
  const answer = data.answer || 'No answer received.';
  setBanner(answer, 3);
  updateFps(data.fps || 0);
  if (data.question && data.answer) {
    addConversationTurn(data.question, data.answer);
  }
  // Return to NAVIGATE after answer
  applyModeState({ current_mode: 'NAVIGATE' });
  sendCommand({ type:'command', action:'set_mode', mode:'NAVIGATE' });
}

function handleReading(data) {
  if (data.text)     setBanner(data.text, 0);
  if (data.fps)      updateFps(data.fps);
  if (data.detections) updateDetectivePanel(data.detections, data.fps);
}

function handleSystem(data) {
  if (data.text) {
    setBanner(data.text, 0);
    showToast(data.text);
  }
}

// ─── UI helpers ───────────────────────────────────────────────────
function setBanner(text, severity) {
  document.getElementById('guidance-text').textContent = text;
  const banner = document.getElementById('guidance-banner');
  banner.className = `guidance-banner sev${Math.min(severity, 3)}`;
}

function updateFps(fps) {
  if (fps) document.getElementById('fps-display').textContent = `${fps} fps`;
}

function applyModeState(mode) {
  const current = mode.current_mode || 'NAVIGATE';
  window.currentMode = current;
  document.getElementById('mode-badge').textContent = current;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === current);
  });
  document.getElementById('ask-row').classList.toggle('hidden', current !== 'ASK');
}

// ─── Mode buttons ─────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    sendCommand({ type:'command', action:'set_mode', mode });
    applyModeState({ current_mode: mode });
    if (mode === 'ASK') {
      showToast('🧠 ASK mode — speak or type your question');
      setTimeout(() => document.getElementById('ask-input').focus(), 100);
    } else if (mode === 'NAVIGATE') {
      showToast('🧭 NAVIGATE — spatial narration active');
    } else if (mode === 'READ') {
      showToast('📖 READ — point camera at text');
    }
  });
});

// ─── ASK input ────────────────────────────────────────────────────
function submitQuestion() {
  const input = document.getElementById('ask-input');
  const q = input.value.trim();
  if (!q) return;
  sendCommand({ type:'command', action:'ask', question: q });
  input.value = '';
  showToast(`❓ Asking: "${q}"`);
}
document.getElementById('btn-send').addEventListener('click', submitQuestion);
document.getElementById('ask-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitQuestion();
});

// ─── Secondary controls ────────────────────────────────────────────
document.getElementById('btn-overlay').addEventListener('click', () => {
  overlayActive = !overlayActive;
  document.getElementById('btn-overlay').classList.toggle('active', overlayActive);
  sendCommand({ type:'command', action:'toggle_overlay' });
  if (!overlayActive) overlay.clear();
  showToast(overlayActive ? '👁 Bounding boxes ON' : '👁 Bounding boxes OFF');
});

document.getElementById('btn-snapshot').addEventListener('click', () => {
  sendCommand({ type:'command', action:'snapshot' });
  showToast('📸 Scene snapshot saved!');
});

document.getElementById('btn-diff').addEventListener('click', () => {
  sendCommand({ type:'command', action:'scene_diff' });
  showToast('❓ Comparing with snapshot...');
});

// ─── Settings ─────────────────────────────────────────────────────
document.getElementById('btn-settings').addEventListener('click', () => {
  document.getElementById('settings-panel').classList.toggle('hidden');
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
    // Force overlay on
    overlayActive = true;
    document.getElementById('btn-overlay').classList.add('active');
    sendCommand({ type:'command', action:'toggle_overlay' });
    // Open detective
    detectivePanelActive = true;
    document.getElementById('detective-panel').classList.remove('hidden');
    document.getElementById('btn-detective').classList.add('active');
    showToast('🎬 Demo Mode ON — Judges can see live AI');
  } else {
    showToast('🎬 Demo Mode OFF');
  }
});

// ─── Detective Panel ──────────────────────────────────────────────
document.getElementById('btn-detective').addEventListener('click', () => {
  detectivePanelActive = !detectivePanelActive;
  document.getElementById('detective-panel').classList.toggle('hidden', !detectivePanelActive);
  document.getElementById('btn-detective').classList.toggle('active', detectivePanelActive);
  showToast(detectivePanelActive ? '🔬 Detective Panel ON' : '🔬 Detective Panel OFF');
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
        <th>Confidence</th><th>BBox Coords</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── Conversation Panel ────────────────────────────────────────────
function addConversationTurn(question, answer) {
  const panel = document.getElementById('conversation-panel');
  const msgs  = document.getElementById('convo-messages');
  panel.classList.remove('hidden');

  const qEl = document.createElement('div');
  qEl.className   = 'bubble-q';
  qEl.textContent = question;

  const aEl = document.createElement('div');
  aEl.className   = 'bubble-a';
  aEl.textContent = answer;

  msgs.appendChild(qEl);
  msgs.appendChild(aEl);

  // Keep last 6 bubbles (3 Q&A turns)
  while (msgs.children.length > 6) msgs.removeChild(msgs.firstChild);
  msgs.scrollTop = msgs.scrollHeight;
}

document.getElementById('btn-clear-convo').addEventListener('click', () => {
  document.getElementById('convo-messages').innerHTML = '';
  document.getElementById('conversation-panel').classList.add('hidden');
});
```

---

## FILE 8: `frontend/manifest.json`

```json
{
  "name": "VisionTalk",
  "short_name": "VisionTalk",
  "description": "Offline AI spatial assistant for the visually impaired",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f0f13",
  "theme_color": "#0f0f13",
  "orientation": "portrait",
  "icons": [
    { "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>👁</text></svg>",
      "sizes": "any", "type": "image/svg+xml" }
  ]
}
```

---

## FILE 9: `frontend/sw.js`

```javascript
// Service Worker — caches app shell only.
// CRITICAL: NEVER intercept /ws/, /api/, /qr, or port 8080 (camera).

const CACHE = 'visiontalk-v1';
const SHELL = ['/', '/static/style.css', '/static/app.js',
               '/static/camera.js', '/static/overlay.js',
               '/static/voice.js', '/static/audio.js'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = e.request.url;
  // NEVER intercept WebSocket, API, QR, or camera port
  if (url.includes('/ws/')   ||
      url.includes('/api/')  ||
      url.includes('/qr')    ||
      url.includes(':8080')  ||
      e.request.method !== 'GET') return;

  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});

// Register in index.html — ADD before </body>:
// <script>if('serviceWorker' in navigator) navigator.serviceWorker.register('/static/sw.js');</script>
```

---

## IMPORTANT NOTES FOR CODING AGENT

1. Service Worker registration: add `<script>if('serviceWorker' in navigator) navigator.serviceWorker.register('/static/sw.js');</script>` before `</body>` in `index.html`

2. The `overlay.js` uses `MODEL_W=640, MODEL_H=480` as reference dimensions. These match the camera frame size. YOLO bounding box coords are original frame coordinates and must be scaled to canvas display size.

3. `app.js` must be loaded AFTER `overlay.js` and `camera.js` (script order in HTML matters).

4. `voice.js` must be loaded AFTER `app.js` (needs `window.sendCommand`).

5. All WebSocket messages match the protocol defined in `FINAL_SPEC.md` exactly.
