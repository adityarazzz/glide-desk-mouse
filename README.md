# Glide P1 prototype: bare-hand desk mouse

This is Phase 1 of the project pack (`../Glide_BareHand_Desk_Mouse_Project_Docs.pdf`). It works with any
webcam that looks down at the desk, and it's tuned on recorded sessions from the ZEB LIVE PRO.

## Start

Double-click **`Glide.vbs`** (or the Start-menu shortcut below). The Glide window opens. The first time, it walks you through setup.
Afterwards it is a small control center:

- **Home** shows the live camera on a centimetre grid, what your hand is doing now, and a speed slider.
- **Gestures** shows an animated demo of every gesture, all at the same scale.
- **Settings** has speed, feel, right-click hold, scroll gesture/speed/direction, cursor ring, sounds,
  appearance, hand, camera and recalibration. No config files.
- **Training** has the **pointing test** (ISO 9241-411, below), records coached sessions and lists past
  ones with their results.

Closing the window keeps Glide running in the tray, because your hand is still the mouse. Quit from the
tray, Settings, or **Ctrl+Alt+Q**. **Ctrl+Alt+G** pauses or resumes from anywhere. Starting Glide a
second time brings the running window forward.

```
powershell -ExecutionPolicy Bypass -File install_shortcut.ps1 [-Desktop]   # Start-menu (and desktop) shortcut
glide.bat --paused              # open with the hand mouse paused
glide.bat --classic             # the developer HUD (OpenCV window) instead of the app
glide.bat --record scroll       # a coached recording in the classic window (basic|angles|precision|scroll)
```

The app is a WebView2 window (`glide/desktop.py`) over a local server (`glide/server.py`). The server
listens on 127.0.0.1 only and needs a per-launch token. The camera → tracker → engine loop runs on its
own thread (`glide/service.py`), so the interface can never slow the mouse down. The interface itself
is in `glide/web/`, and its design system is in `DESIGN.md`.

### First-time setup (about 3 minutes, doc §10.1)

1. **Camera.** Found cameras are shown with thumbnails; pick the one looking down at the desk.
2. **Placement.** A live checklist with fix-it tips: light, 24+ fps, hand seen steadily, tracking
   confidence, whole hand in view, room to slide, hand big enough.
3. **Calibrate.** Rest your hand still (this sets the desk axes and scale), then do 5 paced pinches with
   a 3-2-1 countdown and a demo hand (this sets your personal pinch depth and right-click hold).
4. **Learn.** Each gesture is taught live against a practice screen. Your real cursor doesn't move yet.
5. **Practice.** The ISO 9241-411 ring of 9 targets with the real cursor. It reports your throughput
   in bits/s.

## Gestures

Rest your hand on the desk the way you'd hold a mouse.

| Do this | Result |
|---|---|
| Slide your hand | Move the cursor. Faster movements travel further |
| Quick pinch: **thumb tip to middle fingertip** (thumb+index or an index tap also work) | Left click. Two quick ones = double click |
| Pinch, then slide | Drag |
| Pinch and hold still until the ring fills | Right click |
| Make a **fist** and move it | Scroll ("grab the page"); it keeps gliding briefly when you let go |
| Lift your hand high, or take it out of view | Clutch: reposition without moving the cursor |
| Type | Everything is ignored for 0.5 s |

A small ring next to the cursor shows the state, and a soft tick confirms each action:
- green ring = tracking
- white ring = pinched, with an amber arc filling toward right click
- blue = dragging
- orange arrows = scrolling
- grey = clutched

## Why these gestures

The choices come from published hand-tracking guidance and from measurements made with this camera.

- **Pinch to select, pinch-and-hold to drag, hand resting on a surface.** This matches the visionOS
  model ([Apple](https://support.apple.com/en-asia/117741),
  [guideline summary](https://skills.cat/skills/ehmo/platform-design-skills/visionos-design-guidelines)).
- **Gestures should face the camera to avoid self-occlusion.**
  ([Meta](https://developers.meta.com/horizon/design/hands-gesture-design/)) From above, the
  thumb+index contact is hidden under the hand. In recorded sessions the thumb–index gap barely moved
  (0.83 vs 0.61–1.07 at rest), while the thumb–middle **image** gap dipped about 50% on every pinch or tap.
  Glide therefore detects a relative dip of either gap, and thumb+middle is the gesture taught.
- **Immediate visual and audio feedback, because there are no haptics.**
  ([Meta](https://developers.meta.com/horizon/design/hands-best-practices/),
  [Ultraleap](https://docs.ultraleap.com/xr-guidelines/Getting%20started/design-principles.html))
  This is the reason for the cursor ring and the tick.
- **Scroll with a distinct pose.** Index and middle straight, ring and little finger tucked, then
  slide. In the coached recording this pose was seen in 100% of frames. A mouse grip and a fist never
  scrolled. An 8 mm dead zone stops a resting hand from scrolling. The fist does nothing by default.
- **A ghost click destroys trust** (PRD 3.8), so the click detector refuses to press when:
  - tracking confidence is below 0.9 (real pinches scored ≥ 0.98, tracking glitches 0.6–0.87)
  - the gap didn't drop by at least 0.25 within 300 ms (a hand slowly settling into a pinch-like
    pose). The drop is measured from the highest value in that window, because the open level
    wanders between 0.85 and 0.95. The earlier fixed-level clock started seconds too early and
    rejected 6 of the 18 golden pinches.
  - the hand is moving faster than 250 mm/s
  - you are typing

## Testing and the test report

```
.venv\Scripts\python -m unittest discover -s tests -t .     # 180 automated tests, ~15 s
.venv\Scripts\python -m glide.report --live 20               # TEST_REPORT.md (start Glide first)
```

`glide.report` traces every requirement of the project pack (FR-01…FR-10, NFR-01…NFR-09, the
success metrics in §3.8 and the P1 exit criterion) to **38 test points**, at the levels of the test
plan (§6.1):

- **Unit**: property tests on synthetic hands and trajectories.
- **Replay**: the whole pipeline on recorded, labelled sessions (landmarks only).
- **Live**: measured on the running app (frame rate, capture-to-cursor latency, CPU, memory, and
  network connections, which must all be on 127.0.0.1).
- **Manual**: needs a person or a high-speed camera. Each has a written procedure in the report, and
  results go in `tests/manual_results.json`.

It writes `TEST_REPORT.md` (a traceability matrix with measured values against each criterion) and
`TEST_REPORT.json`. `tests/test_report.py` makes sure every test point still names tests that exist.

Measured on 2026-11-19 (i7-1255U, ZEB LIVE PRO):

| Metric | Result | Spec |
|---|---|---|
| Static cursor jitter, hand still | 0.56 px | NFR-02 ≤ 1 px |
| Filter + prediction lag vs raw tracking | 5 ms | NFR-01 |
| Cued clicks detected (golden session) | 16/18 | FR-03 (v1.0: 97% with the IR camera) |
| Click drift, median | 1.0 px | FR-06 |
| Ghost presses while resting, sliding, hovering | 0 in all recordings | PRD 3.8 |
| Capture → cursor command, P50 / P95 | 53 / 78 ms | NFR-01 Tier 0 P50 ≤ 90 ms (motion-to-photon) |
| Engine time per frame | 0.2–0.5 ms | NFR-05 |
| CPU, all cores | 3.6% (no hand) | NFR-05 ≤ 10% |
| Memory, working set / private | 320 / 314 MB | NFR-05 ≤ 300 MB (a v1.0 budget) |
| Network | listens on 127.0.0.1 only, 0 outside connections | NFR-06 |

### Pointing test (ISO 9241-411)

Training → **Pointing test** runs the standard multi-directional tapping task in the app: four rounds of
nine circles (two distances × 32 and 64 px circles). Run it **With Glide**, then **With mouse or
touchpad**. Each round is scored as effective throughput, TP = IDe / MT. The effective width is
We = 4.133 × SD of the endpoints along the movement axis (Soukoreff & MacKenzie 2004), and the rounds
are averaged. Results are saved to `logs/pointing_*.json` (positions and times only). Training then
shows Glide's throughput as a share of the mouse's, against the PRD target of 80%. `glide/pointing.py`
re-scores what the page measured, and `tests/test_pointing.py` checks it against the ISO formula.

## Tuning from your own data

```
glide.bat --record [basic|angles|precision|scroll]                 # coached, self-paced session
.venv\Scripts\python -m glide.analyze logs\session_*.jsonl          # spec metrics
.venv\Scripts\python -m glide.analyze logs\session_*.jsonl --tune --apply   # grid search -> config.json
```

Recordings hold landmarks and timestamps only. No images are ever stored (NFR-06).

### The coached recorder

The first recorder ran on fixed timers and flashed NOW every 1.8 s. People were still reading when a
phase began and couldn't tell which movement was meant. The coached recorder (`glide/coach.py`,
`glide/record.py`, `glide/demo.py`) works differently:

- **It shows the gesture.** An animated hand demonstrates each gesture beside your own camera view.
  - The camera view is turned so your fingers point up, the same way the demo is drawn.
  - The demo hand uses MediaPipe's 21-landmark layout, and tests run every demo pose through the
    real feature extractor. A demo can't teach a pose the detector rejects.
- **You set the pace.**
  - A step can't start until you've had time to read it (about 240 words per minute).
  - You then start it by resting your hand still, or with SPACE.
  - P pauses, R redoes the step, S skips it, and ESC stops and keeps the finished steps.
- **You get a predictable rhythm.** A 3-2-1 countdown with beeps comes before every GO, since a
  warned cue gives steadier reactions than a surprise flash. Each trial waits until your hand is back,
  open and steady.
- **Each gesture starts with a practice trial.** It isn't scored. If the gesture wasn't seen, you get
  a hint and a second practice trial.
- **It gives feedback.**
  - You see seen / not seen after every trial.
  - Movement steps show a live check, for example "Scroll pose seen".
  - Each step's result (trials seen, fraction of frames passing the check) is saved in the session
    file. The recording therefore also shows when the camera can't see a pose.
- **The clock stops while your hand is out of view.**

Ground truth for the analyzer is unchanged: step labels plus cue times. Reading, waiting and practice
frames are labelled `wait` / `practice`, so they count as neither clicks nor quiet time.
Setup's calibration and lessons use the same demo hand and countdown.

## Engineering notes

- **Camera.** Media Foundation, because DirectShow tops out at about 17 fps on the ZEB. MSMF delivers
  frames in bursts (about 1 ms apart, then a gap), so capture time comes from the driver's sample clock
  (`CAP_PROP_POS_MSEC`, exactly 33.3 ms steps) mapped onto `perf_counter`. Filters run on real capture
  time (TRD 4.4).
  - In a dim room, auto exposure runs about 60 ms, so the sensor makes only about 17 pictures a second.
    MSMF repeats each one to fill 30 fps (44% repeats measured). A repeat read as "the hand stopped" and
    the next frame as a double step, so the cursor moved stop-go.
  - Repeats are now dropped: they are pixel-identical, a difference of 0.0 against 1.2 or more for any
    real frame. The fps readout counts real pictures, and Home suggests a lamp when it stays under 24.
  - `SmoothOutput` spreads each frame's cursor step over the frame interval in about 4 ms ticks. Buttons
    and the wheel land the motion still in flight first.
- **Pointer.**
  - While the hand is just moving, the cursor follows a "mouse sensor" point on the knuckle line. That's
    where a mouse sits, so swivelling the wrist moves it 53–70 mm per radian. The palm-base anchor
    (wrist, thumb base, pinky knuckle) sits at the wrist's pivot and moved only 20–25 mm per radian, so
    users had to slide the whole palm.
  - While a pinch, scroll or fist forms, the cursor blends back to the steady palm-base anchor within
    120 ms, offset so it never jumps. Drag and click decisions always use the steady anchor, because
    the knuckles shift about 10 mm as a pinch forms.
  - The "Show Glide your mouse" session (`glide/mousefit.py`) records the hand and the real cursor while
    you use your own mouse. It fits the point whose motion best explains the cursor and saves it to the
    profile as `sensor`.
  - It's converted to desk mm using the palm-bone px-per-metre scale, so no mat is needed.
  - Then a 1€ filter (fc_min 0.4 Hz, β 0.16, cutoff capped at 8 Hz), 35 ms velocity prediction ramped in
    above 40 mm/s, and sub-pixel accumulation.
  - The gain curve is logistic, scaled to screen width and anchored so a still hand gets exactly
    g_min. It is driven by its own speed estimate, which rises at 8 Hz and falls at 16 Hz.
- **Learns your speed from your clicks (`glide/autogain.py`).** Modelled on AutoGain (Lee et al., CHI 2020)
  and Meyer's optimized submovement model.
  - Every fast submovement aims at the spot you finally click. Glide compares where each one ended
    with that click, and nudges a gain multiplier in five hand-speed bins, weighted by how much cursor
    travel happened at each speed.
  - The goal is that a submovement covers 93% of the way. Updates are at most 10% per click, within
    0.6–1.6x, and skip movements that involved a glide, lift, arrival or scroll.
  - In simulation it converged from 40% too fast to 1.005, and from 30% too slow to 0.992. A correct
    gain stayed at 0.997. What it learns is saved to the profile as `autogain`.
- **Room fit at walls.** The free-desk fit counts glide bands only where the free desk meets the view's
  edge, because walls have none. Counting them at walls once made the fit see 1.1 cm of forward room
  where 7.1 cm was free.
- **The desk camera is remembered by name, not number (`glide/camlist.py`).**
  - Windows numbers only the cameras that work right now. When the laptop's own camera came back, it
    took number 0 and the ZEB became 1, so Glide silently showed the laptop camera.
  - Glide now asks Media Foundation for each camera's name and USB id, in the same order OpenCV
    numbers them, and saves the chosen camera's id in `config.json`.
  - Each time it opens the camera, it looks up that camera's current number.
  - If the desk camera is unplugged, Home says "ZEB LIVE PRO isn't connected" instead of opening
    another camera.
  - Settings and setup list cameras by name and mark the laptop's own.
- **Never stuck.**
  - If no camera is remembered and the camera won't open or shows only black for 8 s,
    `camera.find_working` looks for one that gives a picture. A remembered camera that is dark is
    never swapped for another.
  - If no frames arrive for 3 s, Glide shows "Reconnecting the camera" and reopens it, without
    waiting on the hung read.
  - Recalibration has Cancel (Esc), keeping the old calibration, and gives up by itself after 60 s
    without a hand in view.
  - Training results close with Enter or Esc, and the hand drives the pointer on the result screen.
- **Phantom clicks.**
  - No press counts until a returning hand has landed and been down for 500 ms
    (`click.settle_ms`). That's long enough to learn its open finger gap; before it, fingers
    relaxing as the hand lands looked like a pinch.
  - A clear lift during a pinch cancels it, because the hand closes a little as it starts to lift.
  - A long press only right-clicks if the hand stayed within 8 mm (`click.hold_mm`); a drifting one
    is cancelled.
  - In the recordings, the remaining phantom presses are geometrically the same as real pinches seen
    from above. So **Ctrl+Alt+X** ("that click was a mistake") saves the last 15 s of landmarks and
    cursor positions to `logs/misfire_*.jsonl`, to tune on real misfires.
  - Analysis of the first 18 reports (2026-11-19):
    - Only 1 had a click in its 6 s window, because reaching the keys took longer. The window is
      now 15 s.
    - That press was compared with 42 real pinches on 11 measures: change in palm size, finger
      extension, palm tilt, finger elevation, image and 3-D thumb gaps. It fell inside the real
      range on every one, so tightening the detector would cost real clicks.
    - Telling the two apart needs the touch signal of the IR camera (project pack §2.1, phase P2).
- **Doubtful frames don't move the cursor.** A hand tilting off the desk as it leaves can score 0.58–0.67
  in tracking confidence. Its landmarks then flung the cursor 1160 px in one frame before the leave
  rollback brought it back.
  - A frame scoring below 0.70 that would also leap more than 12 mm is now skipped.
  - Speed is then measured over the real time since the last frame used, so the gain doesn't overshoot.
  - Every replay metric is unchanged, and `DoubtfulFrameTests` replays the captured exit.
- **The free desk is re-checked only after the hand is seen this session.** A camera pointing elsewhere
  between sessions was read as a cramped "desk", and the speed fit was refitted to it.
- **Memory.** NumPy's OpenBLAS committed a buffer for each of 12 CPU threads when it loaded, 363 MB,
  for matrix products on 2×2 arrays. It now runs on one thread (`glide/__init__.py`), and private
  memory fell from 679 to 314 MB.
- **Mouse-frame axes (default `pointer.axes = "mouse"`).** A mouse's sensor is fixed to the mouse, so its
  axes turn with the hand. A sweep pivoting at the elbow or wrist moves the hand on an arc, yet the cursor
  runs straight. Glide's axes now follow the hand's heading continuously, smoothed at 6 Hz and frozen
  during pinches, drags, scrolls and arrivals. In a simulated elbow sweep the cursor path bends 5 px
  over 840 px, against 92 px with fixed axes. Recorded hands turned 4–14° per stroke.
- **Taking the hand away and putting it back.**
  - Any return counts as an arrival, including one within `lost_ms`. Before, a quick return got no
    settling at all.
  - The cursor holds until the hand is fully in view and down on the desk, meaning it looks its
    resting size again; it comes back from above, bigger and still drifting. After that the motion
    fades in.
  - While part of the palm is out of view (its landmarks are guesses), the hand doesn't move the cursor.
  - When the hand leaves without gliding, the pull-away is undone back to where it began, like a mouse
    lifted off the pad.
  - The resting size and the mouse-like lever are kept between visits, not relearned from the worst frames.
- **Free-desk detector (`glide/surface.py`).** A bag, a keyboard or the desk's edge can cover much of
  the view. Every 3 s, on a worker thread, Glide checks where the desk is free:
  - It learns the desk's colour next to the hand, keeps desk-coloured areas (shadows allowed), fills
    in crumbs, and picks the rectangle that gives the hand the most room to slide. This takes about 7 ms.
  - Two checks in a row must agree before anything changes. The speed fit then uses the free desk, and
    the glide edges sit at its border.
  - On a cramped axis each glide band shrinks to half the spare room, so a resting hand never glides.
    Touching the obstacle glides one way and the desk edge the other.
  - Home outlines the free desk, names what blocks it ("something blue on the far side") and says how
    far to move it.
- **Lift detection across the view.** The palm looks bigger the further forward it is under this camera:
  0.9–1.65‰ per pixel in six recordings, or 30–60% across the forward room. Without a correction, a long
  push up the screen read as a lift and froze the cursor. The size is now corrected by a gradient learned
  online (least squares, about 60 s memory, saved to the profile as `lift_grad`). A rise that comes with
  more than 40 mm of sliding counts as perspective, not a lift; real lifts slid 8–16 mm.
- **Click stabilizer.** The click lands where the cursor was when the dip *began*. A press only becomes
  a drag after 18 mm of held displacement, because clicking itself shifts the landmarks 5–10 mm.
- **Fail-safes.**
  - A one-frame tracking jump of more than 60 mm is ignored.
  - Held buttons are released after 300 ms without a hand, on pause, and on exit.
  - When the hand comes into view, the cursor holds for 150 ms, then motion fades in over 300 ms,
    and edge glide stays off for 0.6 s. Otherwise a hand that is still sliding in flings the cursor
    (235 px in a recorded dropout; 13 px with the hold).
  - Only one copy of Glide can run at a time.
- **Known limit.** Injected input can't reach windows running as administrator (Windows UIPI).

## Presenting it

`DEMO.md` has the setup checklist, a seven-minute demo script and what to do if something goes wrong
on the day.
