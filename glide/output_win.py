"""Windows output adapter (TRD 4.10) and global keyboard watcher.

Cursor moves use SetCursorPos (exact physical pixels, bypasses "Enhance pointer precision" so
our own gain curve is the only acceleration); buttons and wheel use SendInput.
Note: like all injected input, this cannot reach windows running as administrator (UIPI).
"""
import ctypes
import threading
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# per-monitor DPI awareness so coordinates are physical pixels on scaled displays
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except (AttributeError, OSError):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass

ULONG_PTR = ctypes.c_size_t
INPUT_MOUSE = 0
MOUSEEVENTF = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    # KEYBDINPUT/HARDWAREINPUT are smaller than MOUSEINPUT; pad to the full union size
    _fields_ = [("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.GetAsyncKeyState.restype = ctypes.c_short


def _send(flags, data=0):
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(0, 0, ctypes.c_uint32(data).value, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class WinOutput:
    def __init__(self):
        self.held = set()

    def screen_width(self):
        return user32.GetSystemMetrics(0)

    def screen_size(self):
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def cursor(self):
        p = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(p))
        return p.x, p.y

    def move_to(self, x, y):
        user32.SetCursorPos(int(x), int(y))

    def move_by(self, dx, dy):
        x, y = self.cursor()
        self.move_to(x + dx, y + dy)

    def button(self, name, down):
        if down == (name in self.held):
            return
        _send(MOUSEEVENTF[name][0 if down else 1])
        (self.held.add if down else self.held.discard)(name)

    def click(self, name):
        self.button(name, True)
        self.button(name, False)

    def scroll(self, vertical, horizontal=0):
        if vertical:
            _send(MOUSEEVENTF_WHEEL, vertical)
        if horizontal:
            _send(MOUSEEVENTF_HWHEEL, horizontal)

    def release_all(self):
        for name in list(self.held):
            self.button(name, False)


class SmoothOutput(WinOutput):
    """Spreads each camera frame's cursor step over the frame interval, so a 17 fps camera moves the
    cursor in small ~4 ms steps instead of jumping up to 100+ px per frame. Absolute moves, buttons
    and the wheel first land any motion still in flight, so clicks happen where the cursor is going."""

    TICK_S = 0.004

    def __init__(self, interval=lambda: 1 / 30):
        super().__init__()
        self.interval = interval            # seconds between camera frames (measured)
        self.lock = threading.Lock()
        self.pending = [0.0, 0.0]           # px still to move
        self.rate = [0.0, 0.0]              # px/s to move them at
        self.frac = [0.0, 0.0]
        self.wake = threading.Event()
        self.running = True
        threading.Thread(target=self._run, name="glide-cursor", daemon=True).start()

    def move_by(self, dx, dy):
        with self.lock:
            span = min(max(self.interval(), 0.01), 0.1)
            for k, d in enumerate((dx, dy)):
                self.pending[k] += d
                self.rate[k] = self.pending[k] / span
        self.wake.set()

    def flush(self):
        with self.lock:
            dx, dy = round(self.pending[0] + self.frac[0]), round(self.pending[1] + self.frac[1])
            self.pending, self.rate, self.frac = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        if dx or dy:
            self._nudge(dx, dy)

    def _nudge(self, dx, dy):
        x, y = self.cursor()
        WinOutput.move_to(self, x + dx, y + dy)        # not self.move_to: that cancels the motion

    def move_to(self, x, y):
        with self.lock:
            self.pending, self.rate, self.frac = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        super().move_to(x, y)

    def button(self, name, down):
        self.flush()
        super().button(name, down)

    def scroll(self, vertical, horizontal=0):
        self.flush()
        super().scroll(vertical, horizontal)

    def stop(self):
        self.running = False
        self.wake.set()

    def _run(self):
        last = time.perf_counter()
        while self.running:
            if not any(self.pending):
                self.wake.wait(0.5)
                self.wake.clear()
                last = time.perf_counter()
                continue
            time.sleep(self.TICK_S)
            now = time.perf_counter()
            dt, last = now - last, now
            with self.lock:
                step = [0, 0]
                for k in range(2):
                    want = self.rate[k] * dt
                    want = max(min(want, self.pending[k]), 0.0) if self.pending[k] >= 0 else                         min(max(want, self.pending[k]), 0.0)
                    self.pending[k] -= want
                    self.frac[k] += want
                    if abs(self.pending[k]) < 1e-6:     # done: land the sub-pixel remainder too
                        self.pending[k] = 0.0
                        step[k] = round(self.frac[k])
                    else:
                        step[k] = int(self.frac[k])
                    self.frac[k] -= step[k]
            if step[0] or step[1]:
                self._nudge(*step)


VK_CTRL, VK_ALT = 0x11, 0x12
HOTKEYS = {"toggle": 0x47, "quit": 0x51, "misfire": 0x58}  # Ctrl+Alt+G, Ctrl+Alt+Q, Ctrl+Alt+X
# keys that count as typing; modifiers, Windows keys and mouse buttons excluded
TYPING_KEYS = [0x08, 0x09, 0x0D, *range(0x20, 0x29), 0x2E, *range(0x30, 0x3A), *range(0x41, 0x5B),
               *range(0x60, 0x70), *range(0xBA, 0xC1), *range(0xDB, 0xDF)]


def _down(vk):
    return user32.GetAsyncKeyState(vk) & 0x8000 != 0


class KeyWatcher:
    """Polled once per frame: typing guard (TRD 4.8 intent gate) plus global hotkeys."""

    def __init__(self, typing_ms=500):
        self.typing_s = typing_ms / 1000
        self.last_key = 0.0
        self.hot_prev = {k: False for k in HOTKEYS}

    def poll(self):
        now = time.perf_counter()
        events = []
        mods = _down(VK_CTRL) and _down(VK_ALT)
        for name, vk in HOTKEYS.items():
            on = mods and _down(vk)
            if on and not self.hot_prev[name]:
                events.append(name)
            self.hot_prev[name] = on
        if not mods and any(_down(vk) for vk in TYPING_KEYS):
            self.last_key = now
        return now - self.last_key < self.typing_s, events
