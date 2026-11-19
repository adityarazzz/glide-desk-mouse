"""Local HTTP bridge between the app window and the service.

127.0.0.1 only. Everything except the static page needs the per-launch token. Commands must be a
JSON POST carrying it in a custom header, which browsers cannot send cross-origin without a CORS
preflight that this server never grants, so no web page can drive the mouse through Glide.

GET  /                     the app (glide/web)
GET  /events?t=            live state, Server-Sent Events (~30 Hz)
GET  /stream.mjpg?t=       camera preview, turned so the fingers point up
GET  /api/<name>?t=        demos | settings | sessions | pointing | thumb/<camera>
POST /api/cmd              {"action": ..., ...}  with header X-Glide-Token
"""
import json
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pointing
from .config import ROOT
from .demo import export
from .replay import load_meta

WEB = Path(__file__).parent / "web"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
         ".ico": "image/x-icon"}
PORTS = range(47620, 47640)


def sessions(limit=12):
    out = []
    for p in sorted((ROOT / "logs").glob("session_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta = load_meta(p) or {}
        out.append({"name": p.name, "script": meta.get("script") or p.stem.split("_")[1],
                    "when": time.strftime("%d %b %Y, %H:%M", time.localtime(p.stat().st_mtime)),
                    "summary": meta.get("summary"), "coached": bool(meta)})
    return out


class Handler(BaseHTTPRequestHandler):
    svc = token = None
    demos = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except ConnectionError:                    # the window closed a kept-alive connection
            pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")          # refuses DNS-rebinding hosts

    def _send(self, code, body=b"", ctype="application/json", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode())

    def do_GET(self):
        if not self._host_ok():
            return self._send(403)
        url = urlparse(self.path)
        q = parse_qs(url.query)
        path = url.path
        if path in ("/", "/index.html"):
            path = "/index.html"
        f = WEB / path.lstrip("/")
        if f.suffix in TYPES and f.parent == WEB and f.exists():
            return self._send(200, f.read_bytes(), TYPES[f.suffix])
        if q.get("t", [""])[0] != self.token:
            return self._send(403)
        if path == "/events":
            return self._events()
        if path == "/stream.mjpg":
            return self._mjpeg()
        if path == "/api/demos":
            hand = q.get("hand", ["right"])[0]
            if hand not in self.demos:
                self.demos[hand] = json.dumps(export("left" if hand == "left" else "right")).encode()
            return self._send(200, self.demos[hand])
        if path == "/api/settings":
            return self._json(self.svc.settings())
        if path == "/api/sessions":
            return self._json(sessions())
        if path == "/api/pointing":
            hist = pointing.history()
            return self._json({"tests": hist, "compare": pointing.compare(hist)})
        if path.startswith("/api/thumb/"):
            idx = path.rsplit("/", 1)[1].split(".")[0]
            jpg = self.svc.camera_thumbs.get(int(idx)) if idx.isdigit() else None
            return self._send(200, jpg, "image/jpeg") if jpg else self._send(404)
        self._send(404)

    def do_POST(self):
        n = min(int(self.headers.get("Content-Length") or 0), 65536)
        raw = self.rfile.read(n)                     # read first, so a refusal never resets the socket
        if not self._host_ok() or self.headers.get("X-Glide-Token") != self.token:
            return self._send(403)
        if urlparse(self.path).path != "/api/cmd":
            return self._send(404)
        try:
            body = json.loads(raw or b"{}")
            action = body.pop("action")
        except (ValueError, KeyError):
            return self._send(400)
        if action == "show" and self.server.on_show:
            self.server.on_show()
        else:
            self.svc.post(action, **body)
        self._json({"ok": True})

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        svc, last, sent = self.svc, -1, 0.0
        try:
            while True:
                with svc.cond:
                    svc.cond.wait_for(lambda: svc.snap_seq != last, timeout=1.0)
                    snap, last = svc.snapshot, svc.snap_seq
                wait = 1 / 30 - (time.perf_counter() - sent)
                if wait > 0:
                    time.sleep(wait)
                self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
                self.wfile.flush()
                sent = time.perf_counter()
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        svc, last = self.svc, -1
        svc.viewers += 1
        try:
            while True:
                with svc.cond:
                    svc.cond.wait_for(lambda: svc.jpeg_seq != last, timeout=1.0)
                    jpg, last = svc.jpeg, svc.jpeg_seq
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                 str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            svc.viewers -= 1


class Server(ThreadingHTTPServer):
    """Exclusive bind: Python's default address reuse lets a second process listen on the same
    port on Windows and take over requests meant for Glide."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(svc, on_show=None):
    """Starts the server on a background thread; returns (url, server)."""
    token = secrets.token_urlsafe(24)
    handler = type("GlideHandler", (Handler,), {"svc": svc, "token": token, "demos": {}})
    for port in list(PORTS) + [0]:
        try:
            httpd = Server(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    httpd.on_show = on_show
    threading.Thread(target=httpd.serve_forever, name="glide-ui-server", daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/?t={token}", httpd
