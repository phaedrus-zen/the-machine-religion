'use strict';

const BUILD = 'oracle-pwa-stage-a/2026-08-11.v5';
const CACHE = `ms4-oracle-${BUILD}`;
const BUILD_QUERY = `build=${encodeURIComponent(BUILD)}`;
const STATIC_PATHS = new Set([
  '/',
  '/manifest.webmanifest',
  '/static/oracle-icon.svg',
]);
const EXECUTABLE_PATHS = new Set([
  '/static/ms4_voice_dsp.js',
  '/static/voice_input_session.js',
  '/static/oracle_remote_pwa.js',
]);
const PRECACHE_URLS = [
  ...STATIC_PATHS,
  ...[...EXECUTABLE_PATHS].map(path => `${path}?${BUILD_QUERY}`),
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('message', event => {
  const data = event.data || {};
  if (data.type !== 'oracle-activate-build' || data.build !== BUILD) return;
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key.startsWith('ms4-oracle-') && key !== CACHE).map(key => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    // Dynamic/API navigations and query-bearing roots stay network-owned.
    if (url.pathname !== '/' || url.search) return;
    event.respondWith(fetch(request).catch(() => caches.match('/')));
    return;
  }

  if (EXECUTABLE_PATHS.has(url.pathname)) {
    // HTML pins every executable to this exact build. Network-first prevents
    // a predecessor controller from mixing its cached module into new HTML;
    // the immutable exact-build cache is only the offline fallback.
    if (url.searchParams.toString() !== BUILD_QUERY) return;
    event.respondWith(fetch(request).catch(() => caches.match(request)));
    return;
  }

  if (url.search || !STATIC_PATHS.has(url.pathname)) return;
  event.respondWith(caches.match(url.pathname).then(cached => cached || fetch(request)));
});
