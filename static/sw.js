const CACHE_VERSION = 'v2';   // ⬅️ Increment this number every time you make changes
const CACHE_NAME = `vetscan-${CACHE_VERSION}`;

// Only cache static assets that rarely change (icons, etc.)
// DO NOT cache the HTML root ('/') – we always fetch fresh HTML from the network.
const urlsToCache = [
  '/manifest.json'
  // Add any other static assets (e.g., '/static/icon-192.png') if they exist
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
  // Force the waiting service worker to become active immediately
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      );
    })
  );
  // Take control of all open clients without needing a refresh
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  // For HTML navigation requests – always go to the network, never cache.
  if (event.request.mode === 'navigate') {
    event.respondWith(fetch(event.request));
    return;
  }
  // For all other requests (images, fonts, etc.): try cache first, then network.
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});