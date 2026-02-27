// camera.js — getUserMedia browser camera (replaces IP Webcam MJPEG)
// Exposes: startCamera(), stopCamera(), captureFrame(), sendFrameToBackend(mode), triggerFindCapture()

(function () {
  'use strict';

  const videoEl       = document.getElementById('camera-feed');
  const placeholderEl = document.getElementById('camera-placeholder');
  const stopBtn       = document.getElementById('btn-stop-camera');

  // Hidden canvas used for JPEG snapshot
  const snapCanvas = document.createElement('canvas');
  const snapCtx    = snapCanvas.getContext('2d');

  let _stream = null;   // active MediaStream

  // ── startCamera ──────────────────────────────────────────────────
  async function startCamera() {
    if (_stream) return;   // already running

    try {
      const facing = window._preferredFacingMode || 'environment';
      const constraints = {
        video: {
          facingMode: { ideal: facing },
          width:  { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };
      _stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = _stream;
      await videoEl.play();

      if (placeholderEl) placeholderEl.classList.add('hidden');
      if (stopBtn)       stopBtn.classList.remove('hidden');

      window.showToast?.('Camera started');
    } catch (err) {
      console.error('[VisionTalk] Camera error:', err);
      window.showToast?.(`Camera error: ${err.message || err.name}`);
      if (placeholderEl) {
        placeholderEl.classList.remove('hidden');
        const hint = placeholderEl.querySelector('.placeholder-hint');
        if (hint) hint.textContent = `Camera denied: ${err.name}`;
      }
    }
  }

  // ── stopCamera ───────────────────────────────────────────────────
  function stopCamera() {
    if (!_stream) return;
    _stream.getTracks().forEach(t => t.stop());
    _stream = null;
    videoEl.srcObject = null;
    if (placeholderEl) placeholderEl.classList.remove('hidden');
    if (stopBtn)       stopBtn.classList.add('hidden');
    window.showToast?.('Camera stopped');
  }

  // ── captureFrame ─────────────────────────────────────────────────
  // Returns a base64 data-URI JPEG string, or null if no stream active.
  function captureFrame() {
    if (!_stream || !videoEl.videoWidth) return null;
    snapCanvas.width  = videoEl.videoWidth;
    snapCanvas.height = videoEl.videoHeight;
    snapCtx.drawImage(videoEl, 0, 0);
    return snapCanvas.toDataURL('image/jpeg', 0.7);
  }

  // ── sendFrameToBackend ────────────────────────────────────────────
  // Captures a JPEG and sends it over the existing WebSocket.
  function sendFrameToBackend(mode) {
    const dataUrl = captureFrame();
    if (!dataUrl) return;
    window.sendCommand?.({
      type: 'frame',
      mode: mode || window.currentMode || 'NAVIGATE',
      data: dataUrl,
    });
  }

  // ── triggerFindCapture ────────────────────────────────────────────
  // Called by voice.js when user confirms capture in FIND mode.
  // Sends one frame tagged as FIND then sends the find_capture action.
  function triggerFindCapture() {
    sendFrameToBackend('FIND');
    window.sendCommand?.({ type: 'command', action: 'find_capture' });
  }

  // ── Stop button ──────────────────────────────────────────────────
  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      stopCamera();
      // Tell app to drop back to idle (no mode capture timers)
      window.onModeChange?.('STOPPED');
    });
  }

  // ── Expose globals ────────────────────────────────────────────────
  window.startCamera        = startCamera;
  window.stopCamera         = stopCamera;
  window.captureFrame       = captureFrame;
  window.sendFrameToBackend = sendFrameToBackend;
  window.triggerFindCapture = triggerFindCapture;
})();
