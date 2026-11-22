// Minimal service worker: makes the site installable as an app.
// It does not cache anything; every request goes straight to the server,
// so the scam/audio/currency features always use the live backend.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  // Only page loads get a friendly message when the phone is offline.
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    fetch(event.request).catch(
      () => new Response("<h2>Scam Shield is offline</h2><p>Check your connection and try again.</p>",
                         { headers: { "Content-Type": "text/html" } })
    )
  );
});
