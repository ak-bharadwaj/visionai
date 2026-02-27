// voice.js — push-to-talk (header mic) + tap-to-activate (big button) speech recognition
// Requires: window.sendCommand and window.showToast from app.js (loaded first)
//
// DESIGN:
//  - Header mic button (btn-mic): push-to-talk — hold to speak
//  - Big voice button (btn-voice-activate): tap to toggle listening on/off
//  - All voice-originated questions are tagged input_source:'voice'
//  - All chat-originated questions are tagged input_source:'chat'
//  - Voice recognition works in ANY mode — always routes correctly
//  - Mode commands ('navigate', 'read', 'ask', 'find') switch mode + show toast
//  - In FIND mode: capture/yes/ok triggers capture flow; other utterances → find_question

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

// ── Shared command router ────────────────────────────────────────────────────
// Routes a transcript to the correct feature action. Returns true if a command
// was matched, false if it was treated as a free-form question.
function _routeVoiceCommandRaw(transcript) {
  const lower = transcript.toLowerCase();
  window.showToast?.(`Heard: "${transcript}"`);

  // ── FIND mode capture flow — intercept BEFORE all other handlers ──────────
  if (window.currentMode === 'FIND') {
    // "capture" / "find more" → start capture (ask for confirmation)
    if (lower === 'capture' || lower === 'find more' || lower === 'take photo' || lower === 'scan' ||
        lower === 'take a photo' || lower === 'take a picture' || lower === 'take picture' ||
        lower === 'photograph' || lower === 'photo' || lower === 'picture' ||
        lower === 'snap' || lower === 'snap it' || lower === 'shoot' ||
        lower === 'look at this' || lower === 'what is this' || lower === 'what\'s this' ||
        lower === 'analyze this' || lower === 'analyse this' || lower === 'analyze' ||
        lower === 'analyse' || lower === 'check this' || lower === 'show me' ||
        lower === 'image' || lower === 'get image' || lower === 'capture this') {
      window.sendCommand?.({ type: 'command', action: 'find_start_capture' });
      window.showToast?.('Starting capture...');
      return true;
    }
    // Confirmation — user said yes to "Shall I capture?"
    if (lower === 'yes' || lower === 'ok' || lower === 'sure' || lower === 'go ahead' ||
        lower === 'yeah' || lower === 'yep' || lower === 'do it') {
      // triggerFindCapture() sends one frame then the find_capture action
      if (window.triggerFindCapture) {
        window.triggerFindCapture();
      } else {
        window.sendCommand?.({ type: 'command', action: 'find_capture' });
      }
      window.showToast?.('Capturing...');
      return true;
    }
    // Any other utterance in FIND mode → treat as a question about the capture
    window.sendCommand?.({ type: 'command', action: 'find_question',
      question: transcript, input_source: 'voice' });
    window.showToast?.(`FIND question: "${transcript}"`);
    return true;
  }

  // Emergency SOS
  if (lower === 'help' || lower === 'emergency' || lower === 'sos' ||
      lower === 'call for help' || lower === 'help me') {
    // Show SOS overlay
    const overlay = document.getElementById('sos-overlay');
    if (overlay) overlay.classList.remove('hidden');
    // Flash guidance banner red
    const banner = document.getElementById('guidance-banner');
    if (banner) {
      banner.classList.add('sos-active');
      setTimeout(() => banner.classList.remove('sos-active'), 8000);
    }
    // TTS via server — speak directly, no LLM involved
    window.sendCommand?.({
      type: 'command', action: 'speak',
      text: 'SOS activated. Stay calm. Alert others around you.'
    });
    window.showToast?.('SOS ACTIVATED');
    return true;
  }

  // Contextual help — list all available voice commands
  if (lower === 'what can i say' || lower === 'commands' ||
      lower === 'voice commands') {
    const helpText = (
      'Available commands: navigate, ask, read, find object name, ' +
      'remember this, what changed, show data, clear, stop, settings, ' +
      'and any question in ask mode.'
    );
    window.showToast?.('Commands displayed');
    const gt = document.getElementById('guidance-text');
    if (gt) gt.textContent = helpText;
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: 'List available voice commands briefly', input_source: 'voice' });
    return true;
  }

  // Mode switch
  if (lower === 'navigate' || lower === 'navigation' || lower.includes('navigate mode') || lower === 'go to navigate') {
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
    // Ensure mode switches to NAVIGATE first, then set destination
    if (window.currentMode !== 'NAVIGATE') {
      window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'NAVIGATE' });
      window.applyModeState?.({ current_mode: 'NAVIGATE' });
    }
    window.sendCommand?.({ type: 'command', action: 'nav_destination', destination: dest });
    window.showToast?.(`Navigating to: ${dest}`);
    return true;
  }
  if (lower === 'read' || lower === 'reading' || lower.includes('read mode') || lower === 'go to read') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'READ' });
    window.applyModeState?.({ current_mode: 'READ' });
    window.showToast?.('READ — point camera at text');
    return true;
  }
  if (lower === 'ask' || lower === 'question' || lower.includes('ask mode') || lower === 'go to ask') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'ASK' });
    window.applyModeState?.({ current_mode: 'ASK' });
    window.showToast?.('ASK mode — speak or type your question');
    return true;
  }
  if (lower === 'find' || lower.includes('find mode') || lower === 'go to find') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'FIND' });
    window.applyModeState?.({ current_mode: 'FIND' });
    window.showToast?.('FIND mode — say "capture" to begin');
    return true;
  }

  // Overlay toggle
  if (lower === 'overlay' || lower === 'show overlay' || lower === 'hide overlay') {
    document.getElementById('btn-overlay')?.click();
    return true;
  }

  // Scene memory
  if (lower === 'remember' || lower === 'remember this' || lower === 'snapshot') {
    window.sendCommand?.({ type: 'command', action: 'snapshot' });
    window.showToast?.('Scene snapshot saved');
    return true;
  }
  if (lower === 'what changed' || lower === 'compare' || lower === 'scene diff' ||
      lower.includes('what is different') || lower.includes('what has changed')) {
    window.sendCommand?.({ type: 'command', action: 'scene_diff', input_source: 'voice' });
    return true;
  }

  // Clear conversation
  if (lower === 'clear' || lower === 'clear conversation' || lower === 'clear chat') {
    document.getElementById('btn-clear-convo')?.click();
    window.showToast?.('Conversation cleared');
    return true;
  }

  // Demo mode
  if (lower === 'demo' || lower === 'demo mode') {
    document.getElementById('btn-demo-mode')?.click();
    return true;
  }

  // "Where is X?" — route as ASK question
  const whereMatch = lower.match(/^(?:where is|where's)\s+(?:my\s+|the\s+)?(.+)$/);
  if (whereMatch) {
    const target = whereMatch[1].trim();
    if (window.currentMode !== 'ASK') {
      window.applyModeState?.({ current_mode: 'ASK' });
    }
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: `Where is the ${target}?`, input_source: 'voice' });
    window.showToast?.(`Asking: where is ${target}?`);
    return true;
  }

  // Find Mode — "find my keys", "find bottle", "look for person", etc.
  const findMatch = lower.match(/^(?:find|look for|search for)\s+(?:my\s+)?(.+)$/);
  if (findMatch) {
    const target = findMatch[1].trim();
    window.sendCommand?.({ type: 'command', action: 'find_object', target });
    window.showToast?.(`Searching for: ${target}`);
    // Show find indicator if present
    const findBanner = document.getElementById('find-banner');
    const findLabel  = document.getElementById('find-label');
    if (findBanner) findBanner.classList.remove('hidden');
    if (findLabel)  findLabel.textContent = `Searching: ${target}`;
    return true;
  }

  // Cancel find
  if (lower === 'cancel find' || lower === 'stop searching' || lower === 'cancel search') {
    window.sendCommand?.({ type: 'command', action: 'find_cancel' });
    window.showToast?.('Search cancelled');
    document.getElementById('find-banner')?.classList.add('hidden');
    return true;
  }

  // Stop / Cancel (general)
  if (lower === 'stop' || lower === 'stop it' || lower === 'cancel') {
    window.sendCommand?.({ type: 'command', action: 'find_cancel' });
    document.getElementById('find-banner')?.classList.add('hidden');
    window.showToast?.('Cancelled');
    return true;
  }

  // Repeat last message
  if (lower === 'repeat' || lower === 'say again' || lower === 'again') {
    const bannerText = document.getElementById('guidance-text')?.textContent?.trim();
    if (bannerText) {
      window.sendCommand?.({ type: 'command', action: 'speak', text: bannerText });
      window.showToast?.('Repeating last message');
    }
    return true;
  }

  // Settings
  if (lower === 'settings' || lower === 'open settings') {
    document.getElementById('btn-settings')?.click();
    window.showToast?.('Settings opened');
    return true;
  }

  // Clear history
  if (lower === 'clear history' || lower === 'new conversation' || lower === 'reset') {
    document.getElementById('btn-clear-convo')?.click();
    window.showToast?.('History cleared');
    return true;
  }

  // Show data / detective panel
  if (lower === 'show data' || lower === 'show detections' || lower === 'show ai data') {
    document.getElementById('btn-detective')?.click();
    return true;
  }

  // Scan / barcode mode
  if (lower === 'scan' || lower === 'scan barcode' || lower === 'scan qr' ||
      lower === 'read barcode' || lower === 'read qr' || lower === 'barcode' || lower === 'qr code') {
    window.sendCommand?.({ type: 'command', action: 'set_mode', mode: 'SCAN' });
    window.applyModeState?.({ current_mode: 'SCAN' });
    window.showToast?.('SCAN mode — point at QR code or barcode');
    return true;
  }

  // Natural "describe the scene" / "what am I looking at" → instant NAVIGATE answer
  if (lower.includes('what am i looking at') || lower.includes('what do you see') ||
      lower.includes('describe everything') || lower.includes('describe the scene') ||
      lower.includes('describe my surroundings') || lower.includes('what is around me') ||
      lower.includes('what is in front') || lower === 'describe' || lower === 'look around') {
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: 'What can you see around me?', input_source: 'voice' });
    window.showToast?.('Describing scene...');
    return true;
  }

  // "Is it safe?" shortcuts
  if (lower === 'is it safe' || lower === 'can i walk' || lower === 'is the path clear' ||
      lower === 'is it clear' || lower === 'safe to go' || lower === 'clear path') {
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: 'Is it safe to walk forward?', input_source: 'voice' });
    return true;
  }

  // "How many people?" shortcut
  if (lower === 'how many people' || lower === 'anyone here' || lower === 'is anyone there' ||
      lower === 'people around' || lower === 'people nearby') {
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: 'How many people can you see?', input_source: 'voice' });
    return true;
  }

  // "Which way?" shortcuts
  if (lower === 'which way' || lower === 'which direction' || lower === 'where should i go' ||
      lower === 'guide me' || lower === 'navigate me') {
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: 'Which way should I go?', input_source: 'voice' });
    return true;
  }

  // Free-form utterance in NAVIGATE mode — treat as destination if system is
  // waiting for one (WAIT_DEST state), otherwise route as a navigation question.
  if (window.currentMode === 'NAVIGATE') {
    const navState = window._navState || 'IDLE';
    if (navState === 'WAIT_DEST') {
      // System just asked "Where would you like to go?" — user's reply IS the destination
      window.sendCommand?.({ type: 'command', action: 'nav_destination', destination: transcript });
      window.showToast?.(`Destination set: "${transcript}"`);
      return true;
    }
    // ACTIVE navigate mode — treat as a question about the current scene/path
    window.sendCommand?.({ type: 'command', action: 'ask',
      question: transcript, input_source: 'voice' });
    window.showToast?.(`Asking: "${transcript}"`);
    return true;
  }

  // Free-form ASK question — auto-sent immediately, no button press needed
  if (window.currentMode !== 'ASK') {
    window.applyModeState?.({ current_mode: 'ASK' });
  }
  window.sendCommand?.({ type: 'command', action: 'ask', question: transcript, input_source: 'voice' });
  return false;
}

// Debounced public version — drops duplicate recognition events within 800 ms
const routeVoiceCommand = _debounce(_routeVoiceCommandRaw, 800);

// ────────────────────────────────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn        = document.getElementById('btn-mic');           // header mic (PTT)
const voiceBtn      = document.getElementById('btn-voice-activate'); // big tap button

if (!SpeechRecognition) {
  // Degrade gracefully — show a disabled state without breaking the page
  if (micBtn) {
    micBtn.title    = 'Voice recognition not supported in this browser';
    micBtn.disabled = true;
    micBtn.style.opacity = '0.4';
  }
  if (voiceBtn) {
    voiceBtn.title    = 'Voice recognition not supported in this browser';
    voiceBtn.disabled = true;
    voiceBtn.style.opacity = '0.4';
    // Update label so the user knows
    const lbl = voiceBtn.parentElement?.querySelector('.voice-activate-label');
    if (lbl) lbl.textContent = 'Voice not supported';
  }
} else {
  // ════════════════════════════════════════════════════════════════
  // 1. PUSH-TO-TALK — header mic button (hold to speak)
  // ════════════════════════════════════════════════════════════════
  const recPTT = new SpeechRecognition();
  recPTT.lang            = 'en-US';
  recPTT.interimResults  = false;
  recPTT.maxAlternatives = 1;
  recPTT.continuous      = false;

  let pttListening   = false;
  let pttHoldActive  = false;
  let listeningBadge = null;

  if (micBtn) {
    micBtn.addEventListener('mousedown',  startPTT);
    micBtn.addEventListener('touchstart', e => { e.preventDefault(); startPTT(); }, { passive: false });
    micBtn.addEventListener('mouseup',    stopPTT);
    micBtn.addEventListener('touchend',   stopPTT);
  }

  function startPTT() {
    if (pttListening) return;
    pttListening  = true;
    pttHoldActive = true;
    micBtn.classList.add('listening');
    showListeningBadge();
    // BUG FIX: unmute server STT so it processes audio while PTT is active
    window.sendCommand?.({ type: 'command', action: 'stt_unmute' });
    try { recPTT.start(); } catch (_) {}
  }

  function stopPTT() {
    if (!pttHoldActive) return;
    pttHoldActive = false;
    try { recPTT.stop(); } catch (_) {}
    // BUG FIX: mute server STT again once PTT ends
    window.sendCommand?.({ type: 'command', action: 'stt_mute' });
  }

  function clearPTTState() {
    pttListening  = false;
    pttHoldActive = false;
    micBtn?.classList.remove('listening');
    hideListeningBadge();
  }

  function showListeningBadge() {
    if (listeningBadge) return;
    listeningBadge = document.createElement('div');
    listeningBadge.className   = 'mic-listening-badge';
    listeningBadge.textContent = '🔴 Listening...';
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
  // ════════════════════════════════════════════════════════════════
  if (voiceBtn) {
    const recTA = new SpeechRecognition();
    recTA.lang            = 'en-US';
    recTA.interimResults  = false;
    recTA.maxAlternatives = 1;
    recTA.continuous      = false;

    let taActive     = false;  // whether tap-mode is currently listening
    let noSpeechCount = 0;     // consecutive no-speech errors — escape hatch

    // BUG FIX: .voice-activate-label is now a sibling of #btn-voice-activate
    // inside .voice-activate-section — querySelector works correctly
    const voiceLabel = voiceBtn.parentElement?.querySelector('.voice-activate-label');

    function setTAListening(on) {
      taActive = on;
      voiceBtn.classList.toggle('listening', on);
      voiceBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      if (voiceLabel) {
        voiceLabel.textContent = on ? 'Listening…' : 'Tap to speak';
        voiceLabel.classList.toggle('listening', on);
      }
      if (!on) noSpeechCount = 0; // reset stuck counter when manually stopped
    }

    function startTA() {
      if (taActive) return;
      setTAListening(true);
      try { recTA.start(); } catch (_) {}
    }

    function stopTA() {
      setTAListening(false);
      try { recTA.stop(); } catch (_) {}
    }

    // Tap: click on desktop, touchend on mobile
    voiceBtn.addEventListener('click', () => {
      if (taActive) {
        stopTA();
      } else {
        startTA();
      }
    });

    recTA.onresult = (event) => {
      noSpeechCount = 0; // successful result resets the stuck counter
      const transcript = event.results[0][0].transcript.trim();
      routeVoiceCommand(transcript);
    };

    recTA.onerror = (e) => {
      if (e.error === 'no-speech') {
        noSpeechCount++;
        // BUG FIX: after 5 consecutive no-speech events, escape the stuck state
        // so the button doesn't show "Listening" forever with a broken mic
        if (noSpeechCount >= 5) {
          setTAListening(false);
          window.showToast?.('Mic not hearing audio — tap to try again');
          noSpeechCount = 0;
        }
        // else: onend will restart as normal — don't call setTAListening(false)
      } else {
        window.showToast?.(`Mic error: ${e.error}`);
        setTAListening(false);
      }
    };

    recTA.onend = () => {
      // If still toggled on (and not stuck), immediately restart for continuous listening
      if (taActive) {
        try { recTA.start(); } catch (_) {}
      }
    };
  }
}
