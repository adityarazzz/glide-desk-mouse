---
name: Glide
description: Bare-hand desk mouse. The app is drawn on a self-healing cutting mat with a real centimetre grid.
colors:
  mat: "#1f5442"
  mat-raised: "#245d4a"
  mat-hover: "#2a6854"
  mat-deep: "#19473a"
  mat-sunk: "#173f33"
  ruler-yellow: "#e9cf5b"
  bone: "#f2ede1"
  bone-2: "#c3d7cb"
  bone-3: "#b3cfc1"
  state-live: "#c9ea7a"
  state-drag: "#8fd0ff"
  state-scroll: "#f3b44d"
  state-paused: "#ff8f7f"
  skin: "#e6d3bd"
  skin-edge: "#4c3e33"
  light-mat: "#dde8e1"
  light-raised: "#eef4f0"
  light-ink: "#12231b"
  light-accent: "#1f6a50"
typography:
  display:
    fontFamily: "Segoe UI Variable Display, Segoe UI, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  flow-title:
    fontFamily: "Segoe UI Variable Display, Segoe UI, system-ui, sans-serif"
    fontSize: "34px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  section:
    fontFamily: "Segoe UI Variable Display, Segoe UI, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  measure-state:
    fontFamily: "Bahnschrift, Segoe UI, sans-serif"
    fontSize: "44px"
    fontWeight: 600
    lineHeight: 1
  measure-label:
    fontFamily: "Bahnschrift, Segoe UI, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1
  wordmark:
    fontFamily: "Bahnschrift, Segoe UI, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1
rounded:
  sm: "8px"
  md: "12px"
  pill: "15px"
spacing:
  xs: "4px"
  sm: "10px"
  md: "16px"
  lg: "22px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ruler-yellow}"
    textColor: "#1c2a1f"
    rounded: "{rounded.sm}"
    padding: "0 18px"
    height: "44px"
  button-primary-big:
    backgroundColor: "{colors.ruler-yellow}"
    textColor: "#1c2a1f"
    rounded: "{rounded.sm}"
    padding: "0 26px"
    height: "52px"
  button-secondary:
    backgroundColor: "{colors.mat-raised}"
    textColor: "{colors.bone}"
    rounded: "{rounded.sm}"
    height: "44px"
  switch-on:
    backgroundColor: "{colors.ruler-yellow}"
    rounded: "{rounded.pill}"
    width: "52px"
    height: "30px"
  panel:
    backgroundColor: "{colors.mat-raised}"
    rounded: "{rounded.md}"
    padding: "18px 20px"
  nav-item-active:
    backgroundColor: "{colors.mat-raised}"
    textColor: "{colors.bone}"
    rounded: "{rounded.sm}"
    height: "44px"
---

# Design System: Glide

## Overview

The desk is the mouse pad, so the app is drawn on the desk's own measuring surface: a green self-healing cutting mat. The camera view, the demo hands and the setup screens all sit on a centimetre grid, and around the live camera that grid is the real one. Its square size comes from the measured image scale at your hand, and rulers along two edges count true centimetres. This is the one signature. Everything else is a plain, well-made Windows tool: a left navigation rail, rows of standard controls, big targets.

Mode is Operate. Users visit the window, change one thing and leave; Glide then runs from the tray. The pointer may be the hand-mouse itself, so controls are at least 44 px tall and nothing depends on precise hover.

## Colors

### Primary
- **Mat green** (`mat`, `#1f5442`) is the ground of every screen. Raised panels step lighter (`mat-raised`, `mat-hover`); the rail and the grid plates step darker (`mat-deep`, `mat-sunk`).
- **Ruler yellow** (`ruler-yellow`, `#e9cf5b`) is the mat's marking ink: rulers, the reach cut line, the live hand skeleton, the current setup stage, the primary button, and slider fills.

### Neutral
- **Bone** (`bone`, `#f2ede1`) for text; `bone-2` for secondary text, `bone-3` for tertiary. All meet AA on `mat-raised`.

### State colours
These sit on top of the world, and only where they carry meaning: live / tracking `state-live`, drag `state-drag`, scroll and right-click hold `state-scroll`, paused `state-paused`.

### Light theme
A pale grey-green mat (`light-mat`, `light-raised`) with dark green ink (`light-ink`) and deep green as the marking and accent colour (`light-accent`). The theme follows Windows unless the user picks one under Settings > Appearance.

### Named Rules
- **The Ink Rule.** Yellow is measuring ink. It marks rulers, cut lines, the hand and the one primary action on a screen; it never decorates.
- **The True Grid Rule.** A grid appears only where it is a measuring surface: around the live camera it is the measured centimetre; on demo plates it is the demo hand's own scale (8.5–12 px per cm).

## Typography

- **Segoe UI Variable** (Text and Display) carries all interface text: titles 30 px (34 px in the full-window flows), section headings 18 px, body 15 px.
- **Bahnschrift**, the DIN face that ships with Windows, carries everything a measuring tool would print: ruler numerals, the wordmark, the live state word (44 px, 85% width), fps and latency, results, key caps. It always uses tabular figures.

### Named Rules
- **The Measure Rule.** A number that measures something is set in Bahnschrift with tabular figures; a sentence never is.

## Layout

- The shell is a 232 px rail plus a main area with 30–36 px padding.
- Home is the live stage (camera on the mat) beside a 310 px column holding the state readout, facts, the speed slider and actions. Below a 1060 px viewport the column stacks under the stage.
- Full-window flows (setup, recording) have three parts: stage bands at the top, a visual and a side column in the middle (about 1.45 : 1), and actions at the bottom right.
- Gesture plates use a 4-column grid, falling to 3 below 1320 px and 2 below 1060 px.

## Elevation & Depth

The surface is flat. Panels separate by tone (`mat-raised` on `mat`), not by shadows. Shadows appear only where something physically sits above the mat: the demo hand's own shadow, which grows as the hand lifts, the selected segment, the slider thumb and the practice monitor.

## Shapes

Panels use 12 px radii and controls 8 px. The switch is a pill. Rulers, ticks and cut lines are 1–3 px strokes.

## Components

### Buttons
Primary is ruler yellow with dark ink, 44 px tall (52 px for a flow's main action), one per screen. Secondary is a raised mat tile; quiet is text-only. A disabled primary turns neutral, so it never looks clickable.

### Switches, segmented controls, sliders
Standard controls. The slider track is a ruler: a yellow fill over tick marks at every tenth, with its value in Bahnschrift.

### Navigation
The rail holds the wordmark, four destinations with drawn 1.75-stroke icons, and the always-visible on/off card with its hotkey.

### The measured stage (signature)
The live camera view, turned so your fingers point up, sits on the mat grid. Rulers count centimetres along the top and left edges, a dashed yellow cut line marks the edge-glide band (solid while gliding), the hand skeleton is drawn in yellow, and an anchor dot takes the state colour.

### Demo hands
A canvas hand with skin tones, a desk shadow, fingers folded under the palm drawn darker, and yellow movement paths. Every gesture plate uses the same scale.

### Stage bands (setup)
One band per stage, the current one lit yellow like a departure sign, with finished stages checked. Recalibration shows its own three stages.

## Do's and Don'ts

### Do:
- Do put every hand on the mat at a scale you can state.
- Do show the gesture with a demo hand; text only says when.
- Do keep one primary action per screen and give the next step only.

### Don't:
- Don't add grids, stripes or ruler marks that don't measure anything.
- Don't use eyebrow labels above headings or section numbers.
- Don't use glow, glass or gradient text. This is a working tool on a desk.
