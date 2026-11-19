"""Immediate feedback without haptics: a click-through ring beside the cursor and a soft tick.

Hand-tracking guidelines (Meta, Apple, Ultraleap) agree that visual + audio confirmation is
what makes bare-hand input trustworthy. The ring shows the state at a glance and fills up during
a pinch-hold, so the right-click long press is never a guess.
"""
import ctypes
import math
import struct
import threading
import wave
import winsound
from pathlib import Path

from .config import ROOT

SIZE = 40
OFFSET = 16           # px down-right of the hotspot so the ring never covers what you point at
KEY = "#ff00ff"       # transparent colour
STYLE = {             # state -> (colour, filled)
    "TRACK": ("#3ccf5a", False), "PINCH": ("#ffffff", True), "DRAG": ("#3c8cff", True),
    "HOLD": ("#ffb000", True), "SCROLL": ("#ff8c1a", False), "CLUTCH": ("#9a9a9a", False),
    "PAUSED": ("#e04040", False),
}


def _write_tone(path, notes, volume=5000):
    """notes: [(hz, ms)], each with a soft attack and an exponential decay."""
    rate, frames = 44100, []
    for hz, ms in notes:
        n = int(rate * ms / 1000)
        frames += [int(volume * min(1.0, i / 60) * math.exp(-i / (n / 4)) * math.sin(2 * math.pi * hz * i / rate))
                   for i in range(n)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", f) for f in frames))


SOUNDS = {   # countdown beat, GO, trial seen, trial missed, step finished
    "tick": [(2200, 25)], "beat": [(740, 70)], "go": [(1480, 140)],
    "ok": [(988, 70), (1319, 110)], "miss": [(330, 160)], "step_done": [(784, 80), (988, 80), (1319, 140)],
}


def play(name):
    path = ROOT / "models" / f"{name}.wav"
    try:
        if not path.exists():
            _write_tone(path, SOUNDS[name])
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except (RuntimeError, OSError):
        pass


def tick():
    play("tick")


class Overlay(threading.Thread):
    """Tk runs on its own thread; the app only calls set() from the main loop."""

    def __init__(self, cursor):
        super().__init__(daemon=True)
        self.cursor = cursor
        self.state, self.hold, self.glide = "IDLE", 0.0, None
        self.running = True
        self.poll = None          # called every frame before drawing (the ring process syncs here)

    def set(self, state, hold=0.0, glide=None):
        """glide = screen-direction unit vector while edge-gliding, else None."""
        self.state, self.hold = state, hold
        self.glide = None if glide is None else (round(glide[0], 1), round(glide[1], 1))

    def stop(self):
        self.running = False

    def run(self):
        import tkinter as tk
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", KEY)
        root.configure(bg=KEY)
        canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg=KEY, highlightthickness=0)
        canvas.pack()
        root.update_idletasks()
        # click-through, never focused, no taskbar entry
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ex = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, ex | 0x80000 | 0x20 | 0x80 | 0x08000000)
        last = None

        def frame():
            nonlocal last
            if self.poll:
                self.poll(self)
            if not self.running:
                root.destroy()
                return
            x, y = self.cursor()
            root.geometry(f"{SIZE}x{SIZE}+{x + OFFSET}+{y + OFFSET}")
            key = (self.state, round(self.hold, 2), self.glide)
            if key != last:
                last = key
                canvas.delete("all")
                style = STYLE.get(self.state)
                if style is None:
                    root.withdraw()
                else:
                    root.deiconify()
                    colour, filled = style
                    m = 8
                    canvas.create_oval(m, m, SIZE - m, SIZE - m, outline=colour, width=3,
                                       fill=colour if filled and self.state != "PINCH" else "")
                    if self.state == "PINCH":      # long-press progress toward right click
                        canvas.create_arc(3, 3, SIZE - 3, SIZE - 3, start=90, extent=-359.9 * self.hold,
                                          style="arc", outline="#ffb000", width=4)
                    c = SIZE / 2
                    if self.state == "SCROLL":
                        canvas.create_line(c, c - 7, c, c + 7, fill=colour, width=3, arrow="both")
                    if self.glide is not None:     # edge glide: arrow points where the cursor is heading
                        gx, gy = self.glide
                        canvas.create_line(c - 8 * gx, c - 8 * gy, c + 10 * gx, c + 10 * gy,
                                           fill="#00c8ff", width=3, arrow="last")
            root.after(15, frame)

        frame()
        root.mainloop()
        # Tk objects must die on the thread that made them, or Tcl aborts at interpreter exit
        del canvas, root
        import gc
        gc.collect()


# ---- the ring in its own process ----------------------------------------------------------
# Tk and the app's WebView2 window deadlock when they share a process (both pump Win32 messages
# and Tk waits on the GIL), so the desktop app runs the ring as a small child process and only
# shares six numbers with it.

RING_STATES = list(STYLE) + ["IDLE"]


def _ring_main(shared):
    from .output_win import WinOutput          # sets per-monitor DPI awareness first

    def sync(ov):
        ov.running = shared[5] > 0
        ov.state = RING_STATES[int(shared[0])]
        ov.hold = shared[1]
        ov.glide = (round(shared[2], 1), round(shared[3], 1)) if shared[4] > 0 else None

    ov = Overlay(WinOutput().cursor)
    ov.poll = sync
    ov.run()


class OverlayProcess:
    """Same interface as Overlay (start / set / stop), drawn by a child process."""

    def __init__(self):
        import multiprocessing as mp
        self.ctx = mp.get_context("spawn")
        self.shared = self.ctx.Array("d", 6, lock=False)
        self.shared[0], self.shared[5] = RING_STATES.index("IDLE"), 1
        self.proc = None

    def start(self):
        self.proc = self.ctx.Process(target=_ring_main, args=(self.shared,), daemon=True, name="glide-ring")
        self.proc.start()

    def set(self, state, hold=0.0, glide=None):
        s = self.shared
        s[0] = RING_STATES.index(state) if state in RING_STATES else RING_STATES.index("IDLE")
        s[1] = hold
        s[2], s[3] = glide if glide is not None else (0.0, 0.0)
        s[4] = 1 if glide is not None else 0

    def stop(self):
        self.shared[5] = 0
        if self.proc:
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
