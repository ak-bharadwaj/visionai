// audio.js — browser-side audio utilities
// TTS speech is handled server-side via pyttsx3.
// This file provides short UI sound effects using the Web Audio API (offline, no assets).

(function () {
  let _ctx = null;

  function _getCtx() {
    if (!_ctx) {
      try {
        _ctx = new (window.AudioContext || window.webkitAudioContext)();
      } catch (e) {
        console.warn('[audio.js] Web Audio API not supported:', e);
      }
    }
    return _ctx;
  }

  /**
   * playSuccess() — plays a short high-pitched "ding" to signal a found object.
   * Generated entirely via Web Audio API — no network requests, works offline.
   * Call: window.playSuccess()
   */
  function playSuccess() {
    const ctx = _getCtx();
    if (!ctx) return;

    // Resume in case browser suspended the context (autoplay policy)
    if (ctx.state === 'suspended') ctx.resume();

    const now = ctx.currentTime;

    // Main tone: high sine wave at 1046 Hz (C6), quick attack, fast decay
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(1046.5, now);        // C6
    osc1.frequency.exponentialRampToValueAtTime(1318.5, now + 0.04); // E6 shimmer
    gain1.gain.setValueAtTime(0, now);
    gain1.gain.linearRampToValueAtTime(0.28, now + 0.02);   // fast attack
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.32); // decay
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.35);

    // Harmonic: triangle wave an octave up for brightness
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(2093, now);           // C7
    gain2.gain.setValueAtTime(0, now);
    gain2.gain.linearRampToValueAtTime(0.12, now + 0.02);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now);
    osc2.stop(now + 0.25);
  }

  window.playSuccess = playSuccess;
  console.debug('[audio.js] Web Audio success ding ready.');
})();
