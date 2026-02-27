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

// Register in index.html — add before </body>:
// <script>if('serviceWorker' in navigator) navigator.serviceWorker.register('/static/sw.js');</script>
