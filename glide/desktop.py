"""The Glide desktop app: a native window (WebView2) over the local UI server, plus a tray icon.

Closing the window keeps Glide running (your hand is still the mouse); quit from the tray, the app
or Ctrl+Alt+Q. Starting Glide a second time brings the running window forward instead.
"""
import ctypes
import json
import threading
import urllib.request

from PIL import Image, ImageDraw

from .config import ROOT

ICON = ROOT / "models" / "glide.ico"
INSTANCE = ROOT / "models" / ".instance.json"
MAT, GRID, RULE, BONE = (31, 84, 66), (58, 118, 96), (233, 207, 91), (239, 233, 218)


def make_icon(size=256):
    """Mat-green tile, centimetre grid, ruler ticks along the bottom, bone cursor arrow."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 256
    d.rounded_rectangle((8 * s, 8 * s, 248 * s, 248 * s), radius=52 * s, fill=MAT)
    for k in range(1, 5):
        x = 8 * s + k * 48 * s
        d.line((x, 30 * s, x, 226 * s), fill=GRID, width=max(1, round(2 * s)))
        d.line((30 * s, x, 226 * s, x), fill=GRID, width=max(1, round(2 * s)))
    for k in range(13):
        x = 40 * s + k * 14.5 * s
        h = 22 * s if k % 4 == 0 else 12 * s
        d.line((x, 226 * s - h, x, 226 * s), fill=RULE, width=max(1, round(4 * s)))
    arrow = [(92, 52), (92, 176), (122, 148), (142, 194), (164, 184), (144, 140), (184, 138)]
    d.polygon([(x * s, y * s) for x, y in arrow], fill=BONE, outline=(20, 40, 32))
    return im


def ensure_icon():
    if not ICON.exists():
        ICON.parent.mkdir(exist_ok=True)
        make_icon().save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    return ICON


def set_window_icon(title, path):
    """Posts WM_SETICON to this process's visible top-level window with that title."""
    from ctypes import wintypes
    u = ctypes.windll.user32
    pid, found = ctypes.windll.kernel32.GetCurrentProcessId(), []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def each(h, _):
        owner = wintypes.DWORD()
        u.GetWindowThreadProcessId(h, ctypes.byref(owner))
        n = u.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, buf, n + 1)
        if owner.value == pid and buf.value == title and u.IsWindowVisible(h):
            found.append(h)
        return True

    u.EnumWindows(each, 0)
    u.LoadImageW.restype = wintypes.HANDLE
    for size, kind in ((16, 0), (32, 1)):                 # ICON_SMALL, ICON_BIG
        icon = u.LoadImageW(None, str(path), 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
        for h in found:
            u.PostMessageW(h, 0x80, kind, icon)          # WM_SETICON


def forward_to_running():
    """If Glide is already running, ask it to show its window. True when it answered."""
    try:
        info = json.loads(INSTANCE.read_text(encoding="utf-8"))
        req = urllib.request.Request(info["base"] + "/api/cmd", data=b'{"action": "show"}', method="POST",
                                     headers={"X-Glide-Token": info["token"], "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()
        return True
    except (OSError, ValueError, KeyError):
        return False


def run(single_instance, paused=False):
    import webview
    import pystray

    from .service import Service
    from .server import serve

    mutex = single_instance()
    if mutex is None:
        if not forward_to_running():
            ctypes.windll.user32.MessageBoxW(None, "Glide is already running. Look for it in the tray.", "Glide", 0x40)
        return
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Glide.DeskMouse")
    ensure_icon()
    svc = Service(paused)
    state = {"window": None, "quitting": False, "hidden_once": False}

    def show():
        w = state["window"]
        if w:
            w.show()
            w.restore()

    url, httpd = serve(svc, on_show=show)
    base, token = url.split("/?t=")
    INSTANCE.write_text(json.dumps({"base": base, "token": token}), encoding="utf-8")

    def quit_all(*_):
        if state["quitting"]:
            return
        state["quitting"] = True
        tray.stop()
        svc.stop()
        httpd.shutdown()
        INSTANCE.unlink(missing_ok=True)
        state["window"].destroy()

    def toggle(*_):
        svc.post("enable")

    tray = pystray.Icon("Glide", make_icon(64), "Glide - bare-hand mouse", menu=pystray.Menu(
        pystray.MenuItem("Open Glide", lambda *_: show(), default=True),
        pystray.MenuItem("Pause", toggle, checked=lambda _: not svc.enabled),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Glide", quit_all)))
    svc.on_quit = lambda: threading.Thread(target=quit_all, daemon=True).start()
    svc.start()
    tray.run_detached()

    window = webview.create_window("Glide", url, width=1280, height=820, min_size=(1000, 680),
                                   background_color="#1F5442", text_select=False)
    state["window"] = window

    def on_closing():
        if state["quitting"]:
            return True
        window.hide()                                   # keep running in the tray
        if not state["hidden_once"]:
            state["hidden_once"] = True
            try:
                tray.notify("Glide keeps running in the tray. Your hand is still the mouse.", "Glide")
            except Exception:
                pass
        return False

    def on_shown():
        # our icon on the window and taskbar. pywebview runs this handler off the UI thread, so
        # touching the WinForms window here deadlocks; post WM_SETICON instead, which never waits
        try:
            set_window_icon("Glide", ICON)
        except Exception:
            pass

    window.events.closing += on_closing
    window.events.shown += on_shown
    webview.start(private_mode=False, storage_path=str(ROOT / "models" / "webview"))
    if not state["quitting"]:
        quit_all()
