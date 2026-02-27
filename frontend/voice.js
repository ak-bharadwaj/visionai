// voice.js — push-to-talk (header mic) + tap-to-activate (big button) speech recognition
// Requires: window.sendCommand and window.showToast from app.js (loaded first)
//
// DESIGN:
//  - Header mic button (btn-mic): push-to-talk — hold to speak
//  - Big voice button (btn-voice-activate): tap to toggle listening on/off
//  - All voice-originated questions are tagged input_source:'voice'
//  - All chat-originated questions are tagged input_source:'chat'
//  - Voice recognition works in ANY mode — always routes correctly
//  - Mode commands ('navigate', 'read', 'ask') switch mode + show toast
//  - Any other utterance → sent as ASK question (switches to ASK mode)

// ── Shared command router ────────────────────────────────────────────────────
// Routes a transcript to the correct feature action. Returns true if a command
// was matched, false if it was treated as a free-form ASK question.
function routeVoiceCommand(transcript) {
  const lower = transcript.toLowerCase();
  window.showToast?.(`Heard: "${transcript}"`);

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

  // "Where is X?" — route as ASK question (same as chat path, so both voice+chat use brain.answer)
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

  // Repeat last message — re-speaks the current guidance banner text via TTS
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

  // Free-form ASK question
  if (window.currentMode !== 'ASK') {
    window.applyModeState?.({ current_mode: 'ASK' });
  }
  window.sendCommand?.({ type: 'command', action: 'ask', question: transcript, input_source: 'voice' });

  // Echo in the text input briefly for visual feedback
  const askInput = document.getElementById('ask-input');
  if (askInput) {
    askInput.value = transcript;
    setTimeout(() => { if (askInput.value === transcript) askInput.value = ''; }, 3000);
  }
  return false;
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn        = document.getElementById('btn-mic');           // header mic button
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

  let pttListening    = false;
  let pttHoldActive   = false;
  let listeningBadge  = null;

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
    try { recPTT.start(); } catch (_) {}
  }

  function stopPTT() {
    if (!pttHoldActive) return;
    pttHoldActive = false;
    try { recPTT.stop(); } catch (_) {}
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

    let taActive      = false;   // whether tap-mode is currently listening
    let taNoSpeechTimer = null;  // auto-stop timeout

    const voiceLabel = voiceBtn.parentElement?.querySelector('.voice-activate-label');

    function setTAListening(on) {
      taActive = on;
      voiceBtn.classList.toggle('listening', on);
      voiceBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      if (voiceLabel) {
        voiceLabel.textContent = on ? 'Listening...' : 'Tap to speak a command';
        voiceLabel.classList.toggle('listening', on);
      }
    }

    function startTA() {
      if (taActive) return;
      setTAListening(true);
      // Auto-stop after 5 s of no speech
      taNoSpeechTimer = setTimeout(() => {
        if (taActive) {
          try { recTA.stop(); } catch (_) {}
        }
      }, 5000);
      try { recTA.start(); } catch (_) {}
    }

    function stopTA() {
      clearTimeout(taNoSpeechTimer);
      setTAListening(false);
    }

    // Tap: click on desktop, touchend on mobile (prevent ghost click with preventDefault)
    voiceBtn.addEventListener('click', () => {
      if (taActive) {
        try { recTA.stop(); } catch (_) {}
        stopTA();
      } else {
        startTA();
      }
    });

    recTA.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      clearTimeout(taNoSpeechTimer);
      setTAListening(false);
      routeVoiceCommand(transcript);
    };

    recTA.onerror = (e) => {
      clearTimeout(taNoSpeechTimer);
      setTAListening(false);
      if (e.error !== 'no-speech') window.showToast?.(`Mic error: ${e.error}`);
    };

    recTA.onend = () => {
      clearTimeout(taNoSpeechTimer);
      setTAListening(false);
    };
  }
}
