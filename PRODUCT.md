# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

(Desktop app for Windows 11: a WebView2 window rendered from local HTML/CSS/JS, driven by a Python engine.)

## Stack

Native window + web UI: pywebview (WebView2) window and a pystray tray icon. The UI is local static HTML/CSS/JS served by a stdlib HTTP server inside the app, with live state over Server-Sent Events and the camera preview as MJPEG. It runs fully offline: no CDN, no web fonts. The Python engine (MediaPipe hand tracking, pointer pipeline, click detector, Win32 input injection) is unchanged underneath.

## Users

Everyday, non-technical PC users who want to control the mouse with a bare hand on the desk, under an overhead webcam. They are sitting at their own desk and expect it to "just work" like a mouse: double-click to start, a short guided first run, then ignore the app while they work.

## Product Purpose

Glide turns one external camera into a mouse. You move with your hand flat on the desk, pinch thumb and middle finger to click, hold the pinch to right-click, pinch and slide to drag, and use a two-finger pose to scroll. The app exists to make that reliable and approachable. Success means a first-time user can set up and click accurately within a few minutes, without reading docs or editing config files, and it feels real-time (≈10 ms processing, 30 fps, ≤1 px jitter).

## Positioning

A desk-surface, relative-motion mouse tuned on real recordings from an overhead camera. It is not a mid-air selfie-cam "virtual mouse": there is no absolute fingertip mapping and no pixel-threshold pinch. Every gesture threshold is measured and regression-tested on recorded sessions.

## Operating Context

- The app runs all day in the background, and the window is opened rarely: first run, pause, adjusting speed, re-calibrating, recording a training session.
- While Glide is on, the user's hand is the mouse, so the app's own UI has to be usable with the hand-mouse too: big targets, nothing that needs precise hovering.
- Hotkeys Ctrl+Alt+G (pause) and Ctrl+Alt+Q (quit) work everywhere.
- A click-through ring near the cursor shows the gesture state.

## Capabilities and Constraints

- Gestures: move, click, right click (hold), drag, two-finger scroll, edge glide for reach, lift to reposition.
- Setup steps: camera choice, placement checks (light, fps, hand in view, room to move), calibration (still hand plus 5 paced pinches), learn the gestures, practice.
- Training: coached recording scripts (basic, angles, precision, scroll) that save landmarks only, never images (NFR-06).
- Settings: pointer speed, scroll speed and direction, right-click hold time, sounds, cursor ring, dominant hand, camera, presets (precision / accessibility / fast).
- Windows only. Injected input can't reach elevated (admin) windows.
- The camera can be used by only one app at a time.

## Brand Commitments

Name: Glide. No logo or brand palette exists yet.

## Evidence on Hand

Real measured metrics live in README.md and tests/: golden session 16/18 cued clicks, 0 ghost clicks, 0.65 px jitter; the two-finger scroll pose was seen 100% of the time in the coached session. There are no testimonials, users or press; none may be invented.

## Product Principles

1. It works before it impresses: real-time quality and zero ghost clicks outrank visuals.
2. Show, don't tell: every gesture is demonstrated by an animated hand, never only described.
3. The user sets the pace: no step advances on a bare timer.
4. Invisible when working: the window is a control center you visit, not a dashboard you watch.
5. Honest feedback: the UI always says what Glide sees (hand found, pose seen, paused).

## Accessibility & Inclusion

Large hit targets (the pointer may be the hand-mouse itself), keyboard-operable controls, reduced-motion respected, readable contrast in both dark and light themes, left-handed support.
