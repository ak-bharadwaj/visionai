// voice.js — push-to-talk (header mic) + tap-to-activate (big button) speech recognition
// Requires: window.sendCommand and window.showToast from app.js (loaded first)
//
// DESIGN:
//  - Header mic button (btn-mic): always visible, push-to-talk — hold to speak
//  - Big voice button (btn-voice-activate): tap to toggle continuous listening on/off
//  - All voice-originated questions are tagged input_source:'voice'
//  - All chat-originated questions are tagged input_source:'chat'
//  - Voice recognition works in ANY mode — always routes correctly
//  - ALL interactive UI elements have at least one voice command
//  - Short 1-2 word commands preferred for speed ("data", "overlay", "front", "back")

// ── Debounce helper ──────────────────────────────────────────────────────────
// Prevents duplicate commands when browser fires recognition result twice quickly.
function _debounce(fn, wait) {
  let timer = null;
  return function (...args) {
    if (timer !== null) return; // drop duplicate within window
    fn.apply(this, args);
    timer = setTimeout(() => { timer = null; }, wait);
  };
}

// ── Camera switch helper ──────────────────────────────────────────────────────
// Selects a camera facing value and applies it — covers btn-apply-camera and
// the camera-facing select element.
function _switchCamera(facing) {
  const sel = document.getElementById('camera-facing');
  if (sel) sel.value = facing;
  // Try the Apply button first (triggers camera.js handler)
  const applyBtn = document.getElementById('btn-apply-camera');
  if (applyBtn) {
    applyBtn.click();
  } else {
    // Fallback: send set_camera command directly
    window.sendCommand?.({ type: 'command', action: 'set_camera', source: facing });
  }
  window.showToast?.(`Switched to ${facing === 'environment' ? 'back' : 'front'} camera`);
}

// ── Shared command router ────────────────────────────────────────────────────
// Routes a transcript to the correct feature action. Returns true if a command
// was matched, false if it was treated as a free-form question.
function _routeVoiceCommandRaw(transcript) {
  const lower = transcript.toLowerCase().trim();
  window.showToast?.(`Heard: "${transcript}"`);

  // ── FIND mode capture flow — intercept BEFORE all other handlers ──────────
  if (window.currentMode === 'FIND') {
    // "capture" / "find more" → start capture (ask for confirmation)
    if (lower === 'capture' || lower === 'find more' || lower === 'take photo' ||
      lower === 'take a photo' || lower === 'take a picture' || lower === 'take picture' ||
      lower === 'photograph' || lower === 'photo' || lower === 'picture' ||
      lower === 'snap' || lower === 'snap it' || lower === 'shoot' ||
      lower === 'look at this' || lower === 'what is this' || lower === "what's this" ||
      lower === 'analyze this' || lower === 'analyse this' || lower === 'analyze' ||
      lower === 'analyse' || lower === 'check this' || lower === 'show me' ||
      lower === 'image' || lower === 'get image' || lower === 'capture this') {
      window.sendCommand?.({ type: 'command', action: 'find_start_capture' });
      window.showToast?.('Starting capture...');
      return true;
    }
    // Confirmation — user said yes to "Shall I capture?"
    if (lower === 'yes' || lower === 'ok' || lower === 'sure' || lower === 'go ahead' ||
      lower === 'yeah' || lower === 'yep' || lower === 'do it' || lower === 'confirm') {
      // Click the visible "Yes" button so all handlers fire correctly
      const yesBtn = document.getElementById('btn-find-capture-yes');
      if (yesBtn && !yesBtn.closest('.hidden')) {
        yesBtn.click();
      } else if (window.triggerFindCapture) {
        window.triggerFindCapture();
      } else {
        window.sendCommand?.({ type: 'command', action: 'find_capture' });
      }
      window.showToast?.('Capturing...');
      return true;
    }
    // Any other utterance in FIND mode → treat as a question about the capture
    window.sendFrameToBackend?.('FIND');
    window.sendCommand?.({
      type: 'command', action: 'find_question',
      question: transcript, input_source: 'voice'
    });
    window.showToast?.(`FIND question: "${transcript}"`);
    return true;
  }

  // ── Emergency SOS ─────────────────────────────────────────────────────────
  if (lower === 'help' || lower === 'emergency' || lower === 'sos' ||
    lower === 'call for help' || lower === 'help me') {
    const sosOverlay = document.getElementById('sos-overlay');
    if (sosOverlay) sosOverlay.classList.remove('hidden');
    const banner = document.getElementById('guidance-banner');
    if (banner) {
      banner.classList.add('sos-active');
      setTimeout(() => banner.classList.remove('sos-active'), 8000);
    }
    window.sendCommand?.({
      type: 'command', action: 'speak',
      text: 'SOS activated. Stay calm. Alert others around you.'
    });
    window.showToast?.('SOS ACTIVATED');
    return true;
  }

  // ── Dismiss SOS overlay ───────────────────────────────────────────────────
  if (lower === 'dismiss' || lower === 'dismiss sos' || lower === 'close sos' ||
    lower === 'cancel sos' || lower === 'clear sos') {
    document.getElementById('sos-overlay')?.classList.add('hidden');
    window.showToast?.('SOS dismissed');
    return true;
  }

  // ── Contextual help — list all available voice commands ───────────────────
  if (lower === 'what can i say' || lower === 'commands' || lower === 'help commands' ||
    lower === 'voice commands' || lower === 'what commands') {
    const helpText = (
      'Say: navigate, ask, read, find, scan, back camera, front camera, ' +
      'overlay, remember, what changed, data, demo, settings, close settings, ' +
      'stop camera, clear, repeat, sos, dismiss, or any question.'
    );
    window.showToast?.('Commands listed');
    const gt = document.getElementById('guidance-text');
    if (gt) gt.textContent = helpText;
    window.sendCommand?.({ type: 'command', action: 'speak', text: helpText });
    return true;
  }

  // ── Stop camera (btn-stop-camera) ─────────────────────────────────────────
  if (lower === 'stop camera' || lower === 'camera off' || lower === 'pause camera' ||
    lower === 'turn off camera' || lower === 'disable camera') {
    document.getElementById('btn-stop-camera')?.click();
    window.showToast?.('Camera stopped');
    return true;
  }

  // ── Camera switch — front / back (camera-facing select + btn-apply-camera) ─
  if (lower === 'front camera' || lower === 'switch to front' || lower === 'front' ||
    lower === 'selfie camera' || lower === 'user camera' || lower === 'face camera') {
    _switchCamera('user');
    return true;
  }
  if (lower === 'back camera' || lower === 'rear camera' || lower === 'switch to back' ||
    lower === 'back' || lower === 'environment camera' || lower === 'main camera') {
    _switchCamera('environment');
    return true;
  }

  // ── Settings panel (btn-settings / btn-settings-close) ───────────────────
  if (lower === 'settings' || lower === 'open settings' || lower === 'camera settings') {
    document.getElementById('btn-settings')?.click();
    window.showToast?.('Settings opened');
    return true;
  }
  if (lower === 'close settings' || lower === 'hide settings' || lower === 'dismiss settings' ||
    lower === 'settings close' || lower === 'exit settings') {
    document.getElementById('btn-settings-close')?.click();
    window.showToast?.('Settings closed');
    return true;
  }

  // ── Mode switches ─────────────────────────────────────────────────────────
  if (lower === 'navigate' || lower === 'navigation' || lower === 'navigate mode' ||
    lower === 'go to navigate' || lower === 'hazard detection') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'NAVIGATE' });
    window.applyModeState?.({ current_mode: 'NAVIGATE' });
    window.showToast?.('NAVIGATE — spatial narration active');
    return true;
  }

  // Navigate-to destination — "navigate to the kitchen", "go to exit", "take me to door"
  const navToMatch = lower.match(
    /^(?:navigate to|go to|take me to|head to|walk to|i want to go to|destination)\s+(?:the\s+|a\s+)?(.+)$/
  );
  if (navToMatch) {
    const dest = navToMatch[1].trim();
    if (window.currentMode !== 'NAVIGATE') {
      window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'NAVIGATE' });
      window.applyModeState?.({ current_mode: 'NAVIGATE' });
    }
    window.sendCommand?.({ type: 'command', action: 'nav_destination', destination: dest });
    window.showToast?.(`Navigating to: ${dest}`);
    return true;
  }

  if (lower === 'read' || lower === 'reading' || lower === 'read mode' ||
    lower === 'go to read' || lower === 'read text') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'READ' });
    window.applyModeState?.({ current_mode: 'READ' });
    window.showToast?.('READ — point camera at text');
    return true;
  }
  if (lower === 'ask' || lower === 'question' || lower === 'ask mode' ||
    lower === 'go to ask' || lower === 'ask a question') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'ASK' });
    window.applyModeState?.({ current_mode: 'ASK' });
    window.showToast?.('ASK mode — speak or type your question');
    return true;
  }
  if (lower === 'find mode' || lower === 'go to find' || lower === 'find something') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'FIND' });
    window.applyModeState?.({ current_mode: 'FIND' });
    window.showToast?.('FIND mode — say "capture" to begin');
    return true;
  }
  if (lower === 'scan' || lower === 'scan mode' || lower === 'scan barcode' ||
    lower === 'scan qr' || lower === 'read barcode' || lower === 'read qr' ||
    lower === 'barcode' || lower === 'qr code') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'SCAN' });
    window.applyModeState?.({ current_mode: 'SCAN' });
    window.showToast?.('SCAN mode — point at QR code or barcode');
    return true;
  }

  // ── Overlay toggle (btn-overlay) ─────────────────────────────────────────
  if (lower === 'overlay' || lower === 'show overlay' || lower === 'hide overlay' ||
    lower === 'toggle overlay' || lower === 'boxes' || lower === 'bounding boxes') {
    document.getElementById('btn-overlay')?.click();
    return true;
  }

  // ── Scene memory (btn-snapshot / btn-diff) ────────────────────────────────
  if (lower === 'remember' || lower === 'remember this' || lower === 'snapshot' ||
    lower === 'save scene' || lower === 'save snapshot' || lower === 'take snapshot') {
    window.sendCommand?.({ type: 'command', action: 'snapshot' });
    window.showToast?.('Scene snapshot saved');
    return true;
  }
  if (lower === 'what changed' || lower === 'compare' || lower === 'scene diff' ||
    lower === 'diff' || lower === 'differences' || lower === 'show diff' ||
    lower.includes('what is different') || lower.includes('what has changed')) {
    document.getElementById('btn-diff')?.click();
    return true;
  }

  // ── Demo mode (btn-demo-mode) ────────────────────────────────────────────
  if (lower === 'demo' || lower === 'demo mode' || lower === 'presentation' ||
    lower === 'presentation mode' || lower === 'toggle demo') {
    document.getElementById('btn-demo-mode')?.click();
    return true;
  }

  // ── Detective / data panel (btn-detective) ────────────────────────────────
  if (lower === 'data' || lower === 'show data' || lower === 'show detections' ||
    lower === 'show ai data' || lower === 'detective' || lower === 'live data' ||
    lower === 'detection data' || lower === 'open data') {
    document.getElementById('btn-detective')?.click();
    return true;
  }
  if (lower === 'hide data' || lower === 'close data' || lower === 'hide detections' ||
    lower === 'close detective' || lower === 'close panel') {
    // Click again to toggle off (btn-detective is a toggle)
    const detPanel = document.getElementById('detective-panel');
    if (detPanel && !detPanel.classList.contains('hidden')) {
      document.getElementById('btn-detective')?.click();
    }
    return true;
  }

  // ── Clear conversation (btn-clear-convo) ─────────────────────────────────
  if (lower === 'clear' || lower === 'clear conversation' || lower === 'clear chat' ||
    lower === 'clear history' || lower === 'new conversation' || lower === 'reset chat') {
    document.getElementById('btn-clear-convo')?.click();
    window.showToast?.('Conversation cleared');
    return true;
  }

  // ── Input box and send interactions (ASK mode) ───────────────────────────
  if (lower === 'send' || lower === 'submit' || lower === 'ask it' || lower === 'send question') {
    document.getElementById('btn-send')?.click();
    return true;
  }
  if (lower === 'clear input' || lower === 'delete text' || lower === 'erase text' || lower === 'clear text') {
    const input = document.getElementById('ask-input');
    if (input) {
      input.value = '';
      window.showToast?.('Input cleared');
    }
    return true;
  }

  // ── Panel scrolling ──────────────────────────────────────────────────────
  if (lower === 'scroll down' || lower === 'page down' || lower === 'go down') {
    const msgs = document.getElementById('convo-messages');
    if (msgs) msgs.scrollBy({ top: 300, behavior: 'smooth' });
    return true;
  }
  if (lower === 'scroll up' || lower === 'page up' || lower === 'go up') {
    const msgs = document.getElementById('convo-messages');
    if (msgs) msgs.scrollBy({ top: -300, behavior: 'smooth' });
    return true;
  }

  // ── Hide/Show camera feed ───────────────────────────────────────────────
  if (lower === 'hide camera' || lower === 'hide feed' || lower === 'hide video') {
    const feed = document.querySelector('.camera-box');
    if (feed) feed.style.opacity = '0';
    window.showToast?.('Camera feed hidden');
    return true;
  }
  if (lower === 'show camera' || lower === 'show feed' || lower === 'show video') {
    const feed = document.querySelector('.camera-box');
    if (feed) feed.style.opacity = '1';
    window.showToast?.('Camera feed visible');
    return true;
  }

  // ── Cancel / Stop (general) ───────────────────────────────────────────────
  if (lower === 'cancel find' || lower === 'stop searching' || lower === 'cancel search') {
    window.sendCommand?.({ type: 'command', action: 'find_cancel' });
    window.showToast?.('Search cancelled');
    document.getElementById('find-banner')?.classList.add('hidden');
    return true;
  }
  if (lower === 'stop' || lower === 'stop it' || lower === 'cancel') {
    window.sendCommand?.({ type: 'command', action: 'find_cancel' });
    document.getElementById('find-banner')?.classList.add('hidden');
    window.showToast?.('Cancelled');
    return true;
  }

  // ── Repeat last message ──────────────────────────────────────────────────
  if (lower === 'repeat' || lower === 'say again' || lower === 'again' ||
    lower === 'repeat that' || lower === 'what did you say') {
    const bannerText = document.getElementById('guidance-text')?.textContent?.trim();
    if (bannerText) {
      window.sendCommand?.({ type: 'command', action: 'speak', text: bannerText });
      window.showToast?.('Repeating last message');
    }
    return true;
  }

  // ── "Where is X?" → ASK ─────────────────────────────────────────────────
  const whereMatch = lower.match(/^(?:where is|where's)\s+(?:my\s+|the\s+)?(.+)$/);
  if (whereMatch) {
    const target = whereMatch[1].trim();
    if (window.currentMode !== 'ASK') window.applyModeState?.({ current_mode: 'ASK' });
    window.sendFrameToBackend?.('ASK');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: `Where is the ${target}?`, input_source: 'voice'
    });
    window.showToast?.(`Asking: where is ${target}?`);
    return true;
  }

  // ── Find object — "find my keys", "find bottle", "look for person" ────────
  // Must come AFTER "find mode" match above (which is exact, not regex)
  const findMatch = lower.match(/^(?:find|look for|search for)\s+(?:my\s+)?(.+)$/);
  if (findMatch) {
    const target = findMatch[1].trim();
    window.sendCommand?.({ type: 'command', action: 'find_object', target });
    window.showToast?.(`Searching for: ${target}`);
    const findBanner = document.getElementById('find-banner');
    const findLabel = document.getElementById('find-label');
    if (findBanner) findBanner.classList.remove('hidden');
    if (findLabel) findLabel.textContent = `Searching: ${target}`;
    return true;
  }

  // ── Scene description shortcuts ───────────────────────────────────────────
  if (lower.includes('what am i looking at') || lower.includes('what do you see') ||
    lower.includes('describe everything') || lower.includes('describe the scene') ||
    lower.includes('describe my surroundings') || lower.includes('what is around me') ||
    lower.includes('what is in front') || lower === 'describe' || lower === 'look around') {
    window.sendFrameToBackend?.('ASK');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: 'What can you see around me?', input_source: 'voice'
    });
    window.showToast?.('Describing scene...');
    return true;
  }

  // ── Safety shortcuts ──────────────────────────────────────────────────────
  if (lower === 'is it safe' || lower === 'can i walk' || lower === 'is the path clear' ||
    lower === 'is it clear' || lower === 'safe to go' || lower === 'clear path' ||
    lower === 'path clear') {
    window.sendFrameToBackend?.('ASK');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: 'Is it safe to walk forward?', input_source: 'voice'
    });
    return true;
  }

  // ── People / direction shortcuts ──────────────────────────────────────────
  if (lower === 'how many people' || lower === 'anyone here' || lower === 'is anyone there' ||
    lower === 'people around' || lower === 'people nearby') {
    window.sendFrameToBackend?.('ASK');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: 'How many people can you see?', input_source: 'voice'
    });
    return true;
  }
  if (lower === 'which way' || lower === 'which direction' || lower === 'where should i go' ||
    lower === 'guide me' || lower === 'navigate me' || lower === 'which way to go') {
    window.sendFrameToBackend?.('ASK');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: 'Which way should I go?', input_source: 'voice'
    });
    return true;
  }

  // ── Free-form utterance in NAVIGATE mode ─────────────────────────────────
  if (window.currentMode === 'NAVIGATE') {
    const navState = window._navState || 'IDLE';
    if (navState === 'WAIT_DEST') {
      window.sendCommand?.({ type: 'command', action: 'nav_destination', destination: transcript });
      window.showToast?.(`Destination set: "${transcript}"`);
      return true;
    }
    window.sendFrameToBackend?.('NAVIGATE');
    window.sendCommand?.({
      type: 'command', action: 'ask',
      question: transcript, input_source: 'voice'
    });
    window.showToast?.(`Asking: "${transcript}"`);
    return true;
  }

  // ── Free-form ASK fallback ────────────────────────────────────────────────
  if (window.currentMode !== 'ASK') window.applyModeState?.({ current_mode: 'ASK' });
  window.sendFrameToBackend?.('ASK');
  window.sendCommand?.({ type: 'command', action: 'ask', question: transcript, input_source: 'voice' });
  return false;
}

// Debounced public version — drops duplicate recognition events within 600 ms
// (reduced from 800ms to make commands feel snappier)
const routeVoiceCommand = _debounce(_routeVoiceCommandRaw, 600);

// ────────────────────────────────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById('btn-mic');           // header mic (PTT)
const voiceBtn = document.getElementById('btn-voice-activate'); // big tap button

// ── Always show the PTT mic button ──────────────────────────────────────────
// Previously hidden by default; now always visible so users can hold it
// for instant push-to-talk from anywhere in the app.
if (micBtn) micBtn.classList.remove('hidden');

if (!SpeechRecognition) {
  // Degrade gracefully — show a disabled state without breaking the page
  if (micBtn) {
    micBtn.title = 'Voice recognition not supported in this browser';
    micBtn.disabled = true;
    micBtn.style.opacity = '0.4';
  }
  if (voiceBtn) {
    voiceBtn.title = 'Voice recognition not supported in this browser';
    voiceBtn.disabled = true;
    voiceBtn.style.opacity = '0.4';
    const lbl = voiceBtn.parentElement?.querySelector('.voice-activate-label');
    if (lbl) lbl.textContent = 'Voice not supported';
  }
} else {
  // ════════════════════════════════════════════════════════════════
  // 1. PUSH-TO-TALK — header mic button (hold to speak)
  //    Optimised for speed: recognition starts immediately on press,
  //    result fires as soon as speech ends (no waiting for timeout).
  // ════════════════════════════════════════════════════════════════
  const recPTT = new SpeechRecognition();
  recPTT.lang = 'en-US';
  recPTT.interimResults = false;
  recPTT.maxAlternatives = 1;
  recPTT.continuous = false;

  let pttListening = false;
  let pttHoldActive = false;
  let listeningBadge = null;

  if (micBtn) {
    micBtn.addEventListener('mousedown', startPTT);
    micBtn.addEventListener('touchstart', e => { e.preventDefault(); startPTT(); }, { passive: false });
    micBtn.addEventListener('mouseup', stopPTT);
    micBtn.addEventListener('mouseleave', stopPTT);  // safety: release if cursor leaves
    micBtn.addEventListener('touchend', stopPTT);
    micBtn.addEventListener('touchcancel', stopPTT);
  }

  function startPTT() {
    if (pttListening) return;
    pttListening = true;
    pttHoldActive = true;
    micBtn.classList.add('listening');
    showListeningBadge();
    window.sendCommand?.({ type: 'command', action: 'stt_unmute' });
    try { recPTT.start(); } catch (_) { }
  }

  function stopPTT() {
    if (!pttHoldActive) return;
    pttHoldActive = false;
    try { recPTT.stop(); } catch (_) { }
    window.sendCommand?.({ type: 'command', action: 'stt_mute' });
  }

  function clearPTTState() {
    pttListening = false;
    pttHoldActive = false;
    micBtn?.classList.remove('listening');
    hideListeningBadge();
  }

  function showListeningBadge() {
    if (listeningBadge) return;
    listeningBadge = document.createElement('div');
    listeningBadge.className = 'mic-listening-badge';
    listeningBadge.textContent = 'Listening...';
    document.body.appendChild(listeningBadge);
  }

  function hideListeningBadge() {
    if (listeningBadge) { listeningBadge.remove(); listeningBadge = null; }
  }

  recPTT.onresult = (event) => {
    const transcript = event.results[0][0].transcript.trim();
    routeVoiceCommand(transcript);
  };

  recPTT.onerror = (e) => {
    clearPTTState();
    if (e.error !== 'no-speech') window.showToast?.(`Mic error: ${e.error}`);
  };

  recPTT.onend = () => {
    if (!pttHoldActive) clearPTTState();
  };

  // ════════════════════════════════════════════════════════════════
  // 2. TAP-TO-ACTIVATE — big voice button (tap toggles listening)
  //    Restarts immediately after each result for hands-free use.
  // ════════════════════════════════════════════════════════════════
  if (voiceBtn) {
    const recTA = new SpeechRecognition();
    recTA.lang = 'en-US';
    recTA.interimResults = false;
    recTA.maxAlternatives = 1;
    recTA.continuous = false;

    let taActive = false;
    let noSpeechCount = 0;

    const voiceLabel = voiceBtn.parentElement?.querySelector('.voice-activate-label');

    function setTAListening(on) {
      taActive = on;
      voiceBtn.classList.toggle('listening', on);
      voiceBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      if (voiceLabel) {
        voiceLabel.textContent = on ? 'Listening…' : 'Tap to speak';
        voiceLabel.classList.toggle('listening', on);
      }
      if (!on) noSpeechCount = 0;
    }

    function startTA() {
      if (taActive) return;
      setTAListening(true);
      try { recTA.start(); } catch (_) { }
    }

    function stopTA() {
      setTAListening(false);
      try { recTA.stop(); } catch (_) { }
    }

    voiceBtn.addEventListener('click', () => {
      if (taActive) stopTA(); else startTA();
    });

    recTA.onresult = (event) => {
      noSpeechCount = 0;
      const transcript = event.results[0][0].transcript.trim();
      routeVoiceCommand(transcript);
    };

    recTA.onerror = (e) => {
      if (e.error === 'no-speech') {
        noSpeechCount++;
        if (noSpeechCount >= 5) {
          setTAListening(false);
          window.showToast?.('Mic not hearing audio — tap to try again');
          noSpeechCount = 0;
        }
      } else {
        window.showToast?.(`Mic error: ${e.error}`);
        setTAListening(false);
      }
    };

    recTA.onend = () => {
      // Restart immediately for continuous listening when toggled on
      if (taActive) {
        try { recTA.start(); } catch (_) { }
      }
    };
  }
}
