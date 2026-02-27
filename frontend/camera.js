// camera.js — MJPEG camera feed from Android IP Webcam app
// Rules: Never getUserMedia(). Camera is always <img src="http://IP/video">

const cameraFeed        = document.getElementById('camera-feed');
const cameraPlaceholder = document.getElementById('camera-placeholder');

const STORAGE_KEY = 'visiontalk_phone_ip';
let errorRetryTimer = null;

function applyCameraSource(ip) {
  if (!ip) return;
  ip = ip.trim();
  localStorage.setItem(STORAGE_KEY, ip);
  const url = ip.startsWith('http') ? ip : `http://${ip}`;
  // Clear any pending retry
  clearTimeout(errorRetryTimer);
  cameraFeed.src = `${url}/video`;
  cameraPlaceholder.classList.add('hidden');
}

// Restore saved IP on load — show placeholder if no IP stored
(function restoreIp() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    document.getElementById('ip-input').value = saved;
    applyCameraSource(saved);
  } else {
    // No IP saved — keep placeholder visible explicitly
    cameraPlaceholder.classList.remove('hidden');
  }
})();

// Apply button
document.getElementById('btn-apply-ip').addEventListener('click', () => {
  const ip = document.getElementById('ip-input').value.trim();
  if (ip) {
    applyCameraSource(ip);
    // Also tell the backend to switch its camera stream
    const url = ip.startsWith('http') ? ip : `http://${ip}`;
    window.sendCommand?.({ type: 'command', action: 'set_camera', source: `${url}/video` });
    showToast('Camera source applied');
    // Auto-close settings panel
    document.getElementById('settings-panel').classList.add('hidden');
  }
});
document.getElementById('ip-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-apply-ip').click();
});

// Camera error — show placeholder, retry after 5 s
cameraFeed.addEventListener('error', () => {
  cameraPlaceholder.classList.remove('hidden');
  // Retry by reloading src after a delay (MJPEG streams can drop transiently)
  clearTimeout(errorRetryTimer);
  const currentSrc = cameraFeed.src;
  if (currentSrc && currentSrc !== window.location.href) {
    errorRetryTimer = setTimeout(() => {
      // Force browser to re-request by appending a cache-bust param
      const base = currentSrc.split('?')[0];
      cameraFeed.src = `${base}?t=${Date.now()}`;
    }, 5000);
  }
});

// Camera loaded — hide placeholder
cameraFeed.addEventListener('load', () => {
  clearTimeout(errorRetryTimer);
  cameraPlaceholder.classList.add('hidden');
});
