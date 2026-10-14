// Cross-origin isolation for static hosts that cannot send headers (GitHub
// Pages): a service worker adds COOP/COEP to this site's responses, so ONNX
// Runtime may use threads. Loaded as a classic <script> it registers itself and
// reloads the page once; cross-origin files (CDN, Hugging Face) load with CORS.
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    if (e.request.method !== "GET" || new URL(e.request.url).origin !== self.location.origin) return;
    e.respondWith(
      fetch(e.request).then((res) => {
        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
      }),
    );
  });
} else if (!window.crossOriginIsolated && window.isSecureContext && navigator.serviceWorker) {
  navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
    const reload = () => {
      // Once per tab: if isolation still fails, the page runs single-threaded.
      try {
        if (sessionStorage.getItem("coi-reloaded")) return;
        sessionStorage.setItem("coi-reloaded", "1");
      } catch {
        return;
      }
      location.reload();
    };
    const sw = reg.installing || reg.waiting;
    if (!sw) reload();
    else sw.addEventListener("statechange", () => sw.state === "activated" && reload());
  });
}
