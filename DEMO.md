# Presenting Glide

## The day before

1. **Light.** Put a desk lamp over the hand area. In room light the ZEB makes only about 17 pictures a
   second; under a lamp it makes 30, and the cursor is visibly smoother.
2. **Desk.** Clear about 25 × 20 cm under the camera. Move the bag and anything blue or dark out of the
   view; Home outlines the free desk and says what is in the way.
3. **Camera.** Plug in the ZEB LIVE PRO. Glide finds it by name even if Windows renumbers it. Settings
   → Camera should read *ZEB LIVE PRO*.
4. **Recalibrate** if the camera moved (Settings → Recalibrate, about 30 s).
5. **Run the pointing test twice**: Training → Pointing test → *With Glide*, then *With mouse or
   touchpad*. Training then shows Glide as a percentage of the touchpad.
6. **Fill in the manual test points** you can do on your own (`tests/manual_results.json`). The
   procedures are at the end of `TEST_REPORT.md`, and each takes a few minutes:
   - TP-04 recalibration time
   - TP-15 setup time
   - TP-16 live view
   - TP-32 runs without admin
   - TP-37 10 minutes of browsing

   Example entry: `"TP-04": {"result": "pass", "value": "26 s", "note": "phone stopwatch"}`
7. **Make the final report** with the lamp on and your hand moving under the camera for the 20 s:

   ```
   cd F:\Glide\proto
   .venv\Scripts\python -m glide.report --live 20
   ```

8. **Record a backup video** of a working run with the Snipping Tool's screen recorder (Win+Shift+R).
   Make it about 2 minutes: move, click, drag, right click, scroll, the pointing test. If anything fails
   live, you play this.

## Ten minutes before

- Close other apps that use a camera (Camera, Teams, Zoom, browser tabs with video calls).
- Start **Glide.vbs**. On Home check:
  - Camera says ZEB LIVE PRO and about 30 fps.
  - Hand says *Seen · 100% sure* when your hand is down.
  - There is no yellow banner.
- Keep a mouse or touchpad within reach for yourself. Glide doesn't need it, but you might.

## The demo (about 7 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | Your hand resting on the bare desk | "A mouse is a thing you have to grip. Glide makes the desk itself the mouse: one camera above, your bare hand, no gloves, nothing to charge." |
| 0:30 | **Home**: live view, hand skeleton, centimetre grid, free-desk outline | "It sees 21 points of my hand, measures the desk in centimetres, and finds where the desk is free. The camera picture never leaves the computer and is never saved." |
| 1:30 | **Move**: slide around, then small precise moves | "Slow movements are precise, fast ones travel far: a speed-dependent gain, after a 1€ filter that removes the jitter." |
| 2:00 | **Click**: thumb to middle finger. Double-click a folder. | "From above, the thumb hides under the index finger, so Glide watches the thumb-to-middle gap. The click lands where the pinch began, because the hand shifts while pinching." |
| 2:30 | **Drag** a window, **right click** (hold until the ring fills), **scroll** a web page with two fingers | "Every gesture is a native Windows event, so it works in any app." |
| 3:15 | **Reposition**: lift, carry back, put down. Then **Ctrl+Alt+G** to pause and resume. | "Like lifting a mouse. The hotkey is a kill switch that works anywhere." |
| 3:45 | **Training → Pointing test, With Glide** (about a minute) | "This is ISO 9241-411, the standard test for pointing devices. It gives throughput in bits per second." |
| 5:00 | The comparison bar on Training | "Glide reaches N% of my touchpad. The project's target is 80%, after 30 minutes of practice." |
| 5:30 | **TEST_REPORT.md** | "180 automated tests and 38 test points, each traced to a requirement in the project document, with the measured value next to the criterion." |
| 6:30 | Limitations and next step | "It runs at 17 fps in a dim room, the memory is a bit over the v1.0 budget, and from one camera above some curled-hand poses look like a pinch. The next phase adds an infrared camera that can see actual touch." |

## If something goes wrong

| Problem | Fix |
|---|---|
| Home says *ZEB LIVE PRO isn't connected* | Replug it. Glide picks it up within 2 s. |
| Home says *Camera not available* | Another app holds the camera: close it. |
| The cursor feels jumpy | Turn the lamp on (check the fps on Home). Then Settings → Recalibrate. |
| Clicks don't register | Keep the pinching finger straight and pinch quickly. Recalibrate to relearn your pinch depth. |
| A screen seems stuck | Press **Esc**. Every full-window screen closes with Esc. |
| Anything else | **Ctrl+Alt+Q** quits, Glide.vbs starts it again (10 s). Or play the backup video. |

Nothing in the report needs the camera. `TEST_REPORT.md` is already written, and the replay tests run
offline:

```
.venv\Scripts\python -m unittest discover -s tests -t .
.venv\Scripts\python -m glide.analyze tests\data\golden_session.jsonl
```

## Questions you may get

- **"Isn't this just a webcam virtual mouse?"** No.
  - Those map the hand's position in the air to the screen (absolute), which magnifies camera noise
    and tires the arm.
  - Glide is relative, like a mouse: the hand rests on the desk and lifts to reposition.
  - It adds the click stabilizer, a mouse-like tracking point, free-desk detection, and a gain that
    learns from where you click (AutoGain, CHI 2020).
- **"How do you know it's accurate?"**
  - Static jitter is 0.56 px, against a requirement of 1 px or less.
  - Click drift is 1 px (median).
  - The ISO 9241-411 throughput compares it with the same person's touchpad.
- **"How fast is it?"**
  - 53 ms from camera capture to the cursor command (P50).
  - The full motion-to-photon test needs a 240 fps phone video; the procedure is TP-19.
- **"What about privacy?"**
  - Frames are processed in memory only, and recordings hold 21 landmark points per frame.
  - The app's only network socket is on 127.0.0.1 with a per-launch token. The report checks the
    running app's connections.
  - A test scans the code for any image or video writing.
- **"Does it click by accident?"**
  - In every recording: 0 ghost presses while resting, sliding or hovering.
  - In deliberately awkward poses (curled into a mouse grip, tilted on its side) it pressed 3 times in
    54 s.
  - The 18 misfire reports were analysed. The one real phantom was geometrically the same as a real
    pinch on all 11 measures, which is why the next phase adds touch sensing.
- **"What would you do next?"** Follow the project document's roadmap:
  - P2: IR camera, ArUco calibration and real touch detection.
  - P3: a native core (Rust) that meets the 300 MB / 10% CPU budget.
  - A user study (n ≥ 16) for the 80%-of-mouse target and fairness across hands and skin tones.
