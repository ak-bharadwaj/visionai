# VisionTalk — Demo UI Additions
## ⚠️ CORRECT UNDERSTANDING: No CSS black overlay. Real demo uses real user + real AI.

---

## WHAT "BLIND DEMO" ACTUALLY MEANS

```
The USER physically closes their eyes.
They speak to the system by voice.
They hear guidance through the laptop speaker.
Judges watch the laptop/projector showing LIVE CAMERA + BOUNDING BOXES.
The bounding boxes PROVE the AI is thinking — not hardcoded.
```

There is NO "black screen overlay" feature. The screen is always visible to JUDGES.
The USER has closed eyes. Judges see everything the AI sees.

---

## THE 3 DEMO UI FEATURES TO ADD

### Feature 1: 🎬 DEMO PRESENTATION MODE
When "🎬 Demo" button is pressed:
- **Overlay goes PERMANENTLY ON** (bounding boxes always visible)
- **Banner goes BIG** — large font so judges can read from far away
- **Detective panel opens** — live BBox coords + confidence showing
- **FPS counter visible** — proves it's running at ~10fps, not faked
- Detection labels on canvas show: `"chair · left · nearby · 89%"`

This is what judges see on the projector:
```
┌─────────────────────────────────────────────────────────┐
│ Live camera with bounding boxes                        │
│ [Chair ─ 89% ─ nearby, left]   [Table ─ 91% ─ ahead]  │
│ ████████████                    xxxxxxxxxxxxxxxxxx     │
│                                                        │
│    ↑ ZONE: FAR LEFT | LEFT | AHEAD | RIGHT | FAR RIGHT │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  📢 "Chair nearby, to your left"                       │ ← BIG TEXT
└─────────────────────────────────────────────────────────┘
```

### Feature 2: 🔬 DETECTIVE PANEL
Live table visible during demo:
```
Object   Distance  Direction  Conf   BBox coords
chair    nearby    left       89%    [142,241]→[318,479]
table    ahead     ahead      91%    [200,150]→[500,450]
person   far       right      76%    [580,80]→[640,400]
```
Updates live every frame. Proves real inference.

### Feature 3: 💬 CONVERSATION PANEL
Chat-style Q&A history from ASK mode:
```
You: "What is in front of me?"
🧠 VisionTalk: "I can see a table directly ahead with a bottle."

You: "Guide me to the door"
🧠 VisionTalk: "I can see a doorway on your right, approximately 4 steps away."
```

---

## HTML — ADD INSIDE `<main>` AFTER `.secondary-controls`

```html
<!-- Demo Presentation Mode Button -->
<section class="demo-controls">
  <button id="btn-demo-mode" class="demo-btn">
    🎬 Demo Mode
    <span class="mode-hint">For judges — bounding boxes + big text</span>
  </button>
  <button id="btn-detective" class="demo-btn">
    🔬 Show Data
    <span class="mode-hint">Live AI detection proof</span>
  </button>
</section>

<!-- Detective Panel — live detection table for judges -->
<div id="detective-panel" class="detective-panel hidden">
  <div class="detective-header">
    <span>🔬 Live AI Detection Data</span>
    <span id="det-fps" class="det-fps">FPS: --</span>
  </div>
  <div id="detective-table">
    <p class="detective-empty">Waiting for detections...</p>
  </div>
</div>

<!-- Conversation Panel — ASK mode Q&A history -->
<div id="conversation-panel" class="conversation-panel hidden">
  <div class="convo-header">💬 AI Conversation Log</div>
  <div id="convo-messages" class="convo-messages"></div>
</div>
```

---

## CSS — APPEND at end of `style.css`

```css
/* ===== DEMO CONTROLS ===== */
.demo-controls {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.demo-btn {
  display: flex; flex-direction: column;
  align-items: center; gap: 2px;
  background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: var(--radius);
  padding: 10px 8px;
  color: var(--text-muted);
  font-size: 13px; font-weight: 500;
  transition: all var(--transition);
}
.demo-btn .mode-hint { font-size: 9px; opacity: 0.65; }
.demo-btn:active { transform: scale(0.95); }
.demo-btn.active {
  border-color: var(--orange); color: var(--orange);
  background: rgba(249,115,22,0.12);
  box-shadow: 0 0 15px rgba(249,115,22,0.2);
}

/* DEMO MODE: big banner for projector */
body.demo-presentation .guidance-text {
  font-size: 24px !important;
  padding: 20px !important;
  letter-spacing: -0.5px;
}
body.demo-presentation .guidance-banner {
  min-height: 90px;
}

/* ===== DETECTIVE PANEL ===== */
.detective-panel {
  background: rgba(0,0,0,0.7);
  border: 1px solid rgba(108,99,255,0.3);
  border-radius: var(--radius);
  padding: 10px;
  font-family: 'Courier New', monospace;
}
.detective-panel.hidden { display: none; }
.detective-header {
  display: flex; justify-content: space-between;
  font-size: 12px; font-weight: 600; color: var(--purple);
  margin-bottom: 8px;
}
.det-fps { color: var(--green); }
#detective-table { overflow-x: auto; }
#detective-table table { width: 100%; border-collapse: collapse; font-size: 10px; }
#detective-table th {
  color: var(--purple); font-weight: 600;
  padding: 4px 6px; text-align: left;
  border-bottom: 1px solid var(--border);
}
#detective-table td {
  padding: 3px 6px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--text);
}
.det-lv1 { color: var(--red)    !important; font-weight: 700; }
.det-lv2 { color: var(--orange) !important; }
.det-lv3, .det-lv4 { color: var(--green) !important; }
.detective-empty { color: var(--text-muted); font-size: 11px; padding: 8px; }

/* ===== CONVERSATION PANEL ===== */
.conversation-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden; max-height: 260px;
  display: flex; flex-direction: column;
}
.conversation-panel.hidden { display: none; }
.convo-header {
  padding: 8px 14px; font-size: 12px; font-weight: 600;
  color: var(--text-muted); border-bottom: 1px solid var(--border);
  background: var(--bg-surface); flex-shrink: 0;
}
.convo-messages {
  flex: 1; overflow-y: auto; padding: 10px;
  display: flex; flex-direction: column; gap: 8px;
}
.bubble-q {
  align-self: flex-end;
  background: var(--blue); color: #fff;
  padding: 8px 12px; border-radius: 12px 12px 2px 12px;
  font-size: 13px; max-width: 85%;
  animation: fadeSlideIn 0.3s ease;
}
.bubble-q::before { content: "You  "; font-size: 10px; opacity: 0.8; }
.bubble-a {
  align-self: flex-start;
  background: rgba(108,99,255,0.18);
  border: 1px solid rgba(108,99,255,0.3);
  color: var(--text);
  padding: 8px 12px; border-radius: 12px 12px 12px 2px;
  font-size: 13px; max-width: 85%;
  animation: fadeSlideIn 0.3s ease;
}
.bubble-a::before { content: "🧠 AI  "; font-size: 10px; color: var(--purple); }
```

---

## JAVASCRIPT — ADD TO `app.js`

### Section 1: Demo Presentation Mode
```javascript
// ─── Demo Presentation Mode ───────────────────────────────────────
let demoPresentationActive = false;

document.getElementById('btn-demo-mode').addEventListener('click', () => {
  demoPresentationActive = !demoPresentationActive;
  const btn = document.getElementById('btn-demo-mode');
  btn.classList.toggle('active', demoPresentationActive);
  btn.childNodes[0].textContent = demoPresentationActive
    ? '🎬 Exit Demo' : '🎬 Demo Mode';

  document.body.classList.toggle('demo-presentation', demoPresentationActive);

  if (demoPresentationActive) {
    // Force overlay ON
    overlayActive = true;
    document.getElementById('btn-overlay').classList.add('active');
    sendCommand({ type: 'command', action: 'toggle_overlay' });
    // Open detective panel
    detectivePanelActive = true;
    document.getElementById('detective-panel').classList.remove('hidden');
    document.getElementById('btn-detective').classList.add('active');
    showToast('🎬 Demo Mode ON — Judges can see live AI on screen');
  } else {
    showToast('🎬 Demo Mode OFF');
  }
});
```

### Section 2: Detective Panel
```javascript
// ─── Detective Panel ──────────────────────────────────────────────
let detectivePanelActive = false;

document.getElementById('btn-detective').addEventListener('click', () => {
  detectivePanelActive = !detectivePanelActive;
  const btn = document.getElementById('btn-detective');
  btn.classList.toggle('active', detectivePanelActive);
  btn.childNodes[0].textContent = detectivePanelActive
    ? '🔬 Hide Data' : '🔬 Show Data';
  document.getElementById('detective-panel').classList.toggle('hidden', !detectivePanelActive);
});

function updateDetectivePanel(detections, fps) {
  if (!detectivePanelActive && !demoPresentationActive) return;
  document.getElementById('det-fps').textContent = `FPS: ${fps}`;
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
      <td style="font-size:9px">[${d.x1},${d.y1}]→[${d.x2},${d.y2}]</td>
    </tr>
  `).join('');
  container.innerHTML = `
    <table>
      <thead><tr>
        <th>Object</th><th>Distance</th><th>Direction</th>
        <th>AI Confidence</th><th>Bounding Box</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}
```

### Section 3: Conversation Panel
```javascript
// ─── Conversation Panel ───────────────────────────────────────────
const convoPanel    = document.getElementById('conversation-panel');
const convoMessages = document.getElementById('convo-messages');

function addConversationTurn(question, answer) {
  convoPanel.classList.remove('hidden');
  const qEl = document.createElement('div');
  qEl.className = 'bubble-q';
  qEl.textContent = question;
  const aEl = document.createElement('div');
  aEl.className = 'bubble-a';
  aEl.textContent = answer;
  convoMessages.appendChild(qEl);
  convoMessages.appendChild(aEl);
  // Keep last 6 bubbles (3 Q&A turns)
  while (convoMessages.children.length > 6) {
    convoMessages.removeChild(convoMessages.firstChild);
  }
  convoMessages.scrollTop = convoMessages.scrollHeight;
}
```

### Section 4: REPLACE handleNarration and handleAnswer

```javascript
// ─── REPLACE handleNarration (was in base app.js) ─────────────────
function handleNarration(data) {
  if (data.text) setBanner(data.text, data.severity || 0);
  updateFps(data.fps);
  // Always update detective when demo mode active
  updateDetectivePanel(data.detections, data.fps);
  if (overlayActive && data.detections && data.detections.length) {
    overlay.update(data.detections);
  } else if (!overlayActive) {
    overlay.clear();
  }
}

// ─── REPLACE handleAnswer (was in base app.js) ────────────────────
function handleAnswer(data) {
  setBanner(data.answer || 'No answer.', 3);
  updateFps(data.fps);
  if (data.question && data.answer) {
    addConversationTurn(data.question, data.answer);
  }
  overlay.clear();
}

// ─── handle system messages ───────────────────────────────────────
// ADD to ws.onmessage data routing:
// if (data.type === 'system') { setBanner(data.text, 0); showToast(data.text); }
```

---

## VOICE COMMANDS FOR DEMO ACTOR

These go in `voice.js` `rec.onresult` handler (ADD these to existing checks):

```javascript
// Scene snapshot commands
if (lower.includes('remember') || lower === 'remember this') {
  window.sendCommand?.({ type:'command', action:'snapshot' });
  window.showToast?.('📸 Scene remembered');
}
else if (lower.includes('what changed') || lower.includes('what is different')) {
  window.sendCommand?.({ type:'command', action:'scene_diff' });
}
// These already handled: navigate, read, and other speech → sent as ASK question
```

---

## DEMO SCRIPT FOR ACTOR (3 minutes)

```
[BEFORE DEMO]
Press 🎬 Demo Mode → bounding boxes appear + detective panel opens

[0:00] ACTOR closes eyes. Narrator: "Ready. Navigate mode active."
[0:15] ACTOR walks toward chair
       SCREEN: bounding box tracks chair with "89% confidence"
       AUDIO: "Chair nearby, to your left"
[0:30] ACTOR getting closer
       AUDIO: "Chair very close, directly ahead. Stop."
[0:40] ACTOR stops. Asks by voice: "What is in front of me?"
       LLM answers: "I can see a chair directly ahead and a table to your right."
       SCREEN: Chat bubble appears in conversation panel
[1:00] ACTOR asks: "What color is the chair?"
       INSTANT answer: "The chair appears to be dark brown."
       Judge sees: no delay (color_sense = pure OpenCV, no LLM)
[1:15] ACTOR asks: "Guide me to the door"
       System: "I can see a doorway on your right, about 4 steps away."
[1:30] ACTOR says "Remember this" → snapshot saved
[1:40] ACTOR moves to new position
       System: approaching alerts fire if person walks toward camera
[1:50] ACTOR says "What changed?" 
       System: "The bottle is gone. A new person is present."
[2:00] ACTOR says "Read" → points camera at paper
       System reads: "Paracetamol 500mg. Take 2 tablets with water."
[2:20] ACTOR opens eyes, turns to judges
       "100% offline. No internet. No cloud. Runs on your laptop."
[2:30] DEMO COMPLETE
```
