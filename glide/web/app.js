"use strict";
/* Glide app: state arrives from the service over Server-Sent Events; commands go back as POSTs.
   Canvas draws the live hand on the measured mat and the animated demo hands. */

const T = new URLSearchParams(location.search).get("t") || "";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const withToken = (p) => p + (p.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(T);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const mod = (a, n) => ((a % n) + n) % n;
// device names come from Windows: never treat them as markup
const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function cmd(action, body = {}) {
  return fetch("/api/cmd", {
    method: "POST", headers: { "Content-Type": "application/json", "X-Glide-Token": T },
    body: JSON.stringify({ action, ...body }),
  }).catch(() => {});
}
const getJSON = (p) => fetch(withToken(p)).then((r) => r.json());

let S = {};
let recvAt = performance.now();
let SET = null;
let DEMOS = null;

/* ---- theme colours from CSS tokens --------------------------------------------------------- */
const C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["skin", "skin-dark", "skin-edge", "shadow", "rule", "ink", "ink-2", "ink-3", "line", "mat-sunk",
    "state-live", "state-click", "state-drag", "state-scroll", "state-paused", "state-idle", "warn", "ok"]) {
    C[k] = cs.getPropertyValue("--" + k).trim();
  }
}
/* appearance is a per-machine convenience, so it lives in the window's own storage */
function applyTheme(v) {
  if (v) document.documentElement.dataset.theme = v; else delete document.documentElement.dataset.theme;
  $$("#theme button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.themeValue === (v || ""))));
  readColors();
}
let savedTheme = "";
try { savedTheme = localStorage.getItem("glide-theme") || ""; } catch (e) {}
applyTheme(savedTheme);
matchMedia("(prefers-color-scheme: light)").addEventListener("change", readColors);
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("#theme button");
  if (!b) return;
  try { localStorage.setItem("glide-theme", b.dataset.themeValue); } catch (e) {}
  applyTheme(b.dataset.themeValue);
});

const STATES = {
  TRACK: ["Moving", "Slide your hand to move the cursor. Pinch thumb and middle finger to click.", "state-live"],
  PINCH: ["Pinch", "Open to click. Keep holding for a right click.", "state-click"],
  DRAG: ["Dragging", "Open your hand to drop.", "state-drag"],
  HOLD: ["Right click", "Open your hand to finish.", "state-scroll"],
  SCROLL: ["Scrolling", "Slide forward or back. Relax your fingers to stop.", "state-scroll"],
  CLUTCH: ["Lifted", "Put your hand down to carry on from here.", "state-idle"],
  PAUSED: ["Paused", "Switch Glide on, or press Ctrl+Alt+G.", "state-paused"],
  IDLE: ["No hand", "Rest your hand flat under the camera.", "state-idle"],
};

/* ---- canvas helpers ---------------------------------------------------------------------- */
function fit(cv) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}
const visible = (el) => el && el.offsetParent !== null && el.clientWidth > 0;

/* ---- demo hands (data from glide.demo.export) --------------------------------------------- */
const LIMBS = [[0, 1, 17], [1, 2, 16], [2, 3, 15], [3, 4, 14]];
for (const [m, w] of [[5, 15], [9, 15], [13, 14], [17, 13]]) for (let i = 0; i < 3; i++) LIMBS.push([m + i, m + i + 1, w]);

function envelope(s, rise = 0.15, hold = 0.25, fall = 0.2) {
  if (s < 0) return 0;
  if (s < rise) return s / rise;
  if (s < rise + hold) return 1;
  return Math.max(0, 1 - (s - rise - hold) / fall);
}

function demoState(d, t, g) {
  let act = g;
  if (d.act && act == null) act = envelope(mod(t, d.period) - 0.6, ...d.env);
  act = clamp(act || 0, 0, 1);
  let pts = d.frames[0];
  if (d.act) {
    const f = act * (d.frames.length - 1), i = Math.floor(f), j = Math.min(i + 1, d.frames.length - 1), u = f - i;
    pts = d.frames[i].map((p, k) => [p[0] + (d.frames[j][k][0] - p[0]) * u, p[1] + (d.frames[j][k][1] - p[1]) * u,
      p[2] + (d.frames[j][k][2] - p[2]) * u]);
  }
  let dx = 0, dy = 0;
  if (d.move) {
    const n = d.move.length, f = mod(t, d.loop) * 30, i = Math.floor(f) % n, j = (i + 1) % n, u = f - Math.floor(f);
    dx = d.move[i][0] + (d.move[j][0] - d.move[i][0]) * u;
    dy = d.move[i][1] + (d.move[j][1] - d.move[i][1]) * u;
  }
  const note = d.notes ? d.notes[Math.floor(mod(t, d.loop) * 10) % d.notes.length] : null;
  return { pts, dx, dy, act, note };
}

/** Draws demo `name` into canvas `cv` at t seconds; g = activation from a live cue (or null). */
function drawDemo(cv, name, t, g = null, opt = {}) {
  const d = DEMOS && DEMOS[name];
  if (!d || !visible(cv)) return null;
  const [ctx, w, h] = fit(cv);
  const s = opt.scale || Math.min(w / 270, h / 290);       // px per mm
  const cx = w / 2, cy = h / 2 + 8;
  const st = demoState(d, reduced ? 0.6 : t, reduced && d.act && g == null ? 1 : g);
  const P = (x, y, z) => {
    const f = 450 / (450 - clamp(z, -50, 300));
    return [cx + (x + st.dx) * s * f, cy - (y + st.dy - 95) * s * f];
  };
  const Q = (x, y) => [cx + x * s, cy - (y - 95) * s];       // desk plane, no movement

  if (d.path) {
    ctx.strokeStyle = C.rule; ctx.fillStyle = C.rule; ctx.lineWidth = 2; ctx.lineCap = "round";
    const pts = d.path.map(([x, y]) => Q(x, y + 50));
    if (pts.length > 5) {
      ctx.setLineDash([6, 7]); ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke(); ctx.setLineDash([]);
    } else {
      for (let i = 0; i + 1 < pts.length; i++) arrow(ctx, pts[i], pts[i + 1]);
    }
    if (d.move) {
      const [x, y] = Q(st.dx, st.dy + 50);
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill();
    }
  }
  const pts = st.pts;
  const shadow = pts.map(([x, y, z]) => { const [a, b] = P(x + z * 0.35, y - z * 0.45, 0); return [a, b]; });
  const proj = pts.map(([x, y, z]) => P(x, y, z));
  limbsAndPalm(ctx, shadow, pts, s, true);
  limbsAndPalm(ctx, proj, pts, s, false);
  if (d.ring && st.act > 0.9) {
    const [x, y] = proj[d.ring];
    ctx.strokeStyle = C.ok; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(x, y, 13 * s * 1.1, 0, 7); ctx.stroke();
  }
  if (opt.holdRing && st.act > 0.9) {           // right click: the ring that fills while holding
    const [x, y] = proj[4], age = mod(t, d.period) - 0.6 - d.env[0];
    ctx.strokeStyle = C["state-scroll"]; ctx.lineWidth = 4; ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(x, y, 24 * s, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * clamp(age / 1.1, 0, 1)); ctx.stroke();
  }
  return st;
}

function limbsAndPalm(ctx, q, pts, s, shadow) {
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 21; i < 27; i++) (i === 21 ? ctx.moveTo : ctx.lineTo).call(ctx, q[i][0], q[i][1]);
  ctx.closePath();
  ctx.fillStyle = shadow ? C.shadow : C.skin; ctx.fill();
  if (!shadow) { ctx.strokeStyle = C["skin-edge"]; ctx.lineWidth = 1.6; ctx.stroke(); }
  const wrist = pts[0];
  const far = (i) => Math.hypot(pts[i][0] - wrist[0], pts[i][1] - wrist[1]);
  const order = LIMBS.slice(4).sort((a, b) => pts[a[1]][2] - pts[b[1]][2]).concat(LIMBS.slice(0, 4));
  for (const [a, b, wid] of order) {
    const th = Math.max(2, wid * s);
    const tucked = a > 4 && far(b) < far(a) - 2;          // folding back under the palm
    if (!shadow) {
      ctx.strokeStyle = C["skin-edge"]; ctx.lineWidth = th + 2.6;
      ctx.beginPath(); ctx.moveTo(...q[a]); ctx.lineTo(...q[b]); ctx.stroke();
    }
    ctx.strokeStyle = shadow ? C.shadow : tucked ? C["skin-dark"] : C.skin; ctx.lineWidth = th;
    ctx.beginPath(); ctx.moveTo(...q[a]); ctx.lineTo(...q[b]); ctx.stroke();
  }
  if (!shadow) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    for (const i of [4, 8, 12, 16, 20]) { ctx.beginPath(); ctx.arc(q[i][0], q[i][1], Math.max(1.6, 2.6 * s), 0, 7); ctx.fill(); }
  }
}

function arrow(ctx, a, b) {
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), L = 11;
  ctx.beginPath(); ctx.moveTo(...b);
  ctx.lineTo(b[0] - L * Math.cos(ang - 0.45), b[1] - L * Math.sin(ang - 0.45));
  ctx.lineTo(b[0] - L * Math.cos(ang + 0.45), b[1] - L * Math.sin(ang + 0.45));
  ctx.closePath(); ctx.fill();
}

/* ---- the live stage: camera on the measured mat ------------------------------------------- */
const stage = $("#stage-tpl").content.firstElementChild.cloneNode(true);
const stImg = $("img", stage), stCv = $("canvas", stage), stMsg = $(".stage-msg", stage);
let stageSlot = null;
stImg.addEventListener("error", () => setTimeout(() => { if (stageSlot) stImg.src = withToken("/stream.mjpg?r=" + Date.now()); }, 1000));

function mountStage(slot) {
  if (stageSlot === slot) return;
  stageSlot = slot;
  if (!slot) { stage.remove(); stImg.removeAttribute("src"); return; }
  slot.appendChild(stage);
  if (!stImg.getAttribute("src")) stImg.src = withToken("/stream.mjpg");
}

function imageRect(w, h) {
  const [vw, vh] = S.view || [16, 9];
  const box = { x: 34, y: 28, w: w - 46, h: h - 40 };
  const k = Math.min(box.w / vw, box.h / vh);
  return { x: box.x + (box.w - vw * k) / 2, y: box.y + (box.h - vh * k) / 2, w: vw * k, h: vh * k, k };
}

function drawStage() {
  if (!stageSlot || !visible(stCv)) return;
  const [ctx, w, h] = fit(stCv);
  const err = S.camera_error, waiting = !S.view, stalled = S.camera_stalled;
  stMsg.hidden = !(err || waiting || S.dark);
  if (err || waiting) {
    const missing = S.camera_missing;
    $("h2", stMsg).textContent = stalled ? "Reconnecting the camera" : missing ? `${missing} isn't connected`
      : err ? "Camera not available" : "Starting the camera";
    $("p", stMsg).textContent = stalled ? "It stopped sending pictures. Glide is reconnecting it; this takes a few seconds."
      : missing ? "Plug it in and Glide picks it up by itself. To use a different camera, choose it in Settings."
      : err ? "Close other apps that use the camera (Camera, Teams, Zoom), or choose another camera in Settings."
      : "This takes a second or two.";
    return;
  }
  if (S.dark) {
    $("h2", stMsg).textContent = "The camera sees only black";
    $("p", stMsg).textContent = "Turn on a light over your desk, or check that nothing covers the camera.";
  }
  const r = imageRect(w, h);
  const mmpx = S.mm_per_px;
  if (mmpx) {
    const cm = (10 / mmpx) * r.k;                          // screen px per real centimetre at the hand
    stage.style.setProperty("--cm", cm.toFixed(2) + "px");
    stage.style.setProperty("--gx", r.x.toFixed(1) + "px");
    stage.style.setProperty("--gy", r.y.toFixed(1) + "px");
    rulers(ctx, r, cm);
  }
  const e = S.engine || {};
  // free desk (surface detector): what lies outside it is covered, so it is set back under the mat
  const wk = S.work ? { x: r.x + S.work[0] * r.w, y: r.y + S.work[1] * r.h, w: (S.work[2] - S.work[0]) * r.w, h: (S.work[3] - S.work[1]) * r.h } : r;
  if (S.work) {
    ctx.save();
    ctx.fillStyle = C["mat-sunk"]; ctx.globalAlpha = 0.72;
    ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.rect(wk.x, wk.y, wk.w, wk.h); ctx.fill("evenodd");
    ctx.globalAlpha = 0.9; ctx.strokeStyle = C.rule; ctx.lineWidth = 1.5;
    ctx.strokeRect(wk.x, wk.y, wk.w, wk.h);
    const fc = S.desk && S.desk.free_cm;
    if (fc && wk.w > 120) {
      ctx.font = "600 12px Bahnschrift, 'Segoe UI', sans-serif"; ctx.fillStyle = C.rule; ctx.globalAlpha = 1;
      ctx.textBaseline = "bottom";
      ctx.fillText(`FREE DESK  ${fc[0]} × ${fc[1]} CM`, wk.x + 2, wk.y - 5);
    }
    ctx.restore();
  }
  // reach band: past this dashed cut line the cursor keeps gliding
  if (S.edge_inset) {
    const ins = S.edge_inset * Math.min(r.w, r.h);
    const g = S.work && S.glide ? { x: r.x + S.glide[0] * r.w, y: r.y + S.glide[1] * r.h, w: (S.glide[2] - S.glide[0]) * r.w, h: (S.glide[3] - S.glide[1]) * r.h }
      : { x: wk.x + ins, y: wk.y + ins, w: wk.w - 2 * ins, h: wk.h - 2 * ins };
    ctx.save();
    ctx.strokeStyle = C.rule; ctx.globalAlpha = e.edge > 0 ? 0.95 : 0.5; ctx.lineWidth = e.edge > 0 ? 2.5 : 1.5;
    ctx.setLineDash(e.edge > 0 ? [] : [8, 6]);
    ctx.strokeRect(g.x, g.y, g.w, g.h);
    ctx.restore();
  }
  const hand = S.hand;
  if (!hand) return;
  const p = hand.pts.map(([x, y]) => [r.x + x * r.w, r.y + y * r.h]);
  const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
  ctx.lineCap = "round"; ctx.lineWidth = 3; ctx.strokeStyle = C.rule;
  ctx.beginPath();
  for (const [a, b] of BONES) { ctx.moveTo(...p[a]); ctx.lineTo(...p[b]); }
  ctx.stroke();
  ctx.fillStyle = C.ink;
  for (const q of p) { ctx.beginPath(); ctx.arc(q[0], q[1], 3, 0, 7); ctx.fill(); }
  const state = S.mode === "run" || (S.setup && S.setup.step === "practice") ? (e.state || "IDLE") : null;
  if (state) {
    const a = [0, 1, 17].map((i) => p[i]).reduce((m, q) => [m[0] + q[0] / 3, m[1] + q[1] / 3], [0, 0]);
    ctx.fillStyle = C[(STATES[state] || STATES.IDLE)[2]];
    ctx.beginPath(); ctx.arc(a[0], a[1], 8, 0, 7); ctx.fill();
    if (state === "PINCH" && e.hold > 0) {
      ctx.strokeStyle = C["state-scroll"]; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(a[0], a[1], 15, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * e.hold); ctx.stroke();
    }
  }
}

function rulers(ctx, r, cm) {
  ctx.save();
  ctx.strokeStyle = C.rule; ctx.fillStyle = C.rule; ctx.lineWidth = 1;
  ctx.font = "600 11px Bahnschrift, 'Segoe UI', sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  const n = Math.floor(r.w / cm);
  for (let i = 0; i <= n; i++) {
    const x = Math.round(r.x + i * cm) + 0.5, big = i % 5 === 0;
    ctx.beginPath(); ctx.moveTo(x, r.y - 2); ctx.lineTo(x, r.y - (big ? 12 : 6)); ctx.stroke();
    if (big && i) ctx.fillText(String(i), x, r.y - 13);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const m = Math.floor(r.h / cm);
  for (let i = 0; i <= m; i++) {
    const y = Math.round(r.y + i * cm) + 0.5, big = i % 5 === 0;
    ctx.beginPath(); ctx.moveTo(r.x - 2, y); ctx.lineTo(r.x - (big ? 12 : 6), y); ctx.stroke();
    if (big && i) ctx.fillText(String(i), r.x - 14, y);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText("cm", r.x - 30, r.y - 13);
  ctx.restore();
}

/* ---- navigation -------------------------------------------------------------------------- */
function route() {
  const v = (location.hash || "#home").slice(1);
  const name = ["home", "gestures", "settings", "training"].includes(v) ? v : "home";
  $$(".view").forEach((el) => el.classList.toggle("on", el.id === "view-" + name));
  $$(".nav a").forEach((a) => { if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  if (name === "training") loadTraining();
  if (name === "settings") loadSettings();
  placeStage();
}
addEventListener("hashchange", route);

function placeStage() {
  if (S.mode === "record") return mountStage($('[data-slot="record"]'));
  if (S.mode === "setup") return mountStage($("#setup-stage") || null);
  const home = $("#view-home").classList.contains("on");
  mountStage(home ? $('[data-slot="home"]') : null);
}

document.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-cmd]");
  if (!b) return;
  const body = b.dataset.step ? { step: b.dataset.step } : {};
  cmd(b.dataset.cmd, body);
});

/* ---- power + toast ------------------------------------------------------------------------ */
$("#power").addEventListener("click", () => cmd("enable", { on: !S.enabled }));
let toastTimer = 0;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.add("on");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("on"), msg.length > 60 ? 7000 : 2600);
}

/* ---- home -------------------------------------------------------------------------------- */
function updateHome() {
  const on = S.enabled;
  $("#power").setAttribute("aria-checked", String(!!on));
  $("#power-label").textContent = on ? "Glide is on" : "Glide is paused";
  $("#home-title").textContent = on ? "Glide is on" : "Glide is paused";
  $("#home-sub").textContent = !on ? "Your hand won't move the cursor until you switch Glide back on."
    : S.hand ? "Your hand is the mouse. Slide to move, pinch to click." : "Put your hand flat under the camera to start.";
  $("#calib-banner").hidden = !(S.has_profile && !S.calibrated);
  // a dim desk makes the camera repeat pictures to fill 30 fps; only real ones count, so fps drops
  dimFor = S.fps && S.fps < 24 && S.repeats > 0.2 ? dimFor + 1 : 0;
  const dim = dimFor > 90;                                // ~3 s of updates, not a start-up dip
  const db = $("#desk-banner");
  db.hidden = !(S.desk && !deskDismissed() && $("#calib-banner").hidden);
  if (!db.hidden) $("#desk-text").textContent = deskText();
  const lb = $("#light-banner");
  lb.hidden = !(dim && !lightDismissed() && $("#calib-banner").hidden && db.hidden);
  if (!lb.hidden) $("#light-text").textContent = `Your desk is dim, so the camera takes only ${Math.round(S.fps)} pictures a second ` +
    `and the cursor moves in small steps. A lamp over your hand brings it to 30: smoother and quicker to react.`;
  const rb = $("#room-banner");
  rb.hidden = !(S.raise_cm >= 3 && S.room_cm && !roomDismissed() && $("#calib-banner").hidden && lb.hidden && db.hidden);
  if (!rb.hidden) $("#room-text").textContent = `Your hand has ${S.room_cm[0]} × ${S.room_cm[1]} cm of room, so it can run out of view. ` +
    `Raising the camera about ${S.raise_cm} cm lets it reach the whole screen. Until then, lift and carry back, or glide at the edge.`;
  const e = S.engine || {};
  const key = !on ? "PAUSED" : S.hand ? (e.state || "TRACK") : "IDLE";
  const [word, hint, color] = STATES[key] || STATES.IDLE;
  const sw = $("#state-word");
  sw.textContent = word; sw.style.color = `var(--${color})`;
  $("#state-hint").textContent = hint;
  const hold = $("#hold");
  hold.style.visibility = key === "PINCH" ? "visible" : "hidden";
  $("i", hold).style.width = ((e.hold || 0) * 100).toFixed(0) + "%";
  const fh = $("#f-hand");
  fh.textContent = S.hand ? `Seen · ${Math.round(S.hand.score * 100)}% sure` : "Not seen";
  $("#f-cam").textContent = S.camera_error ? "Not available" : S.fps ? `${Math.round(S.fps)} fps${dim ? " · dim" : ""} · ${S.latency_ms} ms` : "Starting";
  $("#f-cam").classList.toggle("bad", !!S.camera_error || (S.fps && S.fps < 20));
  const learned = e.learned_clicks || 0;
  $("#learned-note").textContent = learned ? `Fine-tuned from ${learned} of your clicks.`
    : "Glide fine-tunes this from your clicks as you use it.";
  const slide = S.desk && S.desk.slide_cm;
  $("#f-room").textContent = slide ? `${slide[0]} × ${slide[1]} cm` : S.room_cm ? `${S.room_cm[0]} × ${S.room_cm[1]} cm` : "Measured at setup";
  $("#f-room").classList.toggle("bad", !!slide && Math.min(...slide) < 4);
}

const SIDE = { far: "the far side", near: "your side", left: "the left", right: "the right" };
const AWAY = { far: "further away", left: "further left", right: "further right" };
function deskText() {
  const D = S.desk;
  const things = D.blocked.filter(([side]) => side !== "near").map(([side, colour]) => `something ${colour} on ${SIDE[side]}`);
  if (D.blocked.some(([side]) => side === "near")) things.push("the desk's edge on your side");
  const list = things.length > 1 ? things.slice(0, -1).join(", ") + " and " + things[things.length - 1] : things[0];
  const [sx, sf] = D.slide_cm || [0, 0];
  let t = `The camera sees ${list}, so your hand has ${sf} cm to slide forward and back and ${sx} cm left and right.`;
  for (const a of D.advice || []) {
    t += a.move_cm ? ` Moving the ${a.colour} thing about ${a.move_cm} cm ${AWAY[a.side]} gives it proper room.`
      : ` Moving the ${a.colour} thing out of view gives it more room.`;
  }
  return t + ((D.advice || []).length ? " Until then, push" : " To reach further, push") +
    " your hand against the edge of the free desk and the cursor keeps gliding that way.";
}
const deskKey = () => JSON.stringify(S.desk && S.desk.blocked);
function deskDismissed() {
  try { return localStorage.getItem("glide-desk-ok") === deskKey(); } catch (e) { return false; }
}
$("#desk-ok").addEventListener("click", () => {
  try { localStorage.setItem("glide-desk-ok", deskKey()); } catch (e) {}
  $("#desk-banner").hidden = true;
});

let dimFor = 0;
function lightDismissed() {
  try { return localStorage.getItem("glide-light-ok") === "1"; } catch (e) { return false; }
}
$("#light-ok").addEventListener("click", () => {
  try { localStorage.setItem("glide-light-ok", "1"); } catch (e) {}
  $("#light-banner").hidden = true;
});

function roomDismissed() {
  try { return localStorage.getItem("glide-room-ok") === String(S.raise_cm); } catch (e) { return false; }
}
$("#room-ok").addEventListener("click", () => {
  try { localStorage.setItem("glide-room-ok", String(S.raise_cm)); } catch (e) {}
  $("#room-banner").hidden = true;
});

/* ---- gestures ---------------------------------------------------------------------------- */
const PLATES = [
  ["move_slow", "Move", "Slide your hand flat on the desk."],
  ["pinch_mid", "Click", "Touch your thumb tip to your middle fingertip, then open."],
  ["hold", "Right click", "Pinch and hold still until the ring fills."],
  ["drag", "Drag", "Pinch, keep holding, and slide. Open to drop."],
  ["two_scroll", "Scroll", "Index and middle straight, ring and little finger curled. Slide forward or back."],
  ["reach", "Reach far", "Push into the edge of the view and hold, or keep going out of view. The cursor glides until your hand comes back."],
  ["reposition", "Reposition", "Lift your hand a few centimetres, carry it back and put it down. The cursor stays put."],
  [null, "Pause and quit", "These keys work in every app. Ctrl+Alt+X tells Glide a click was a mistake, so it can learn from it."],
];
function buildPlates() {
  const fist = SET && SET.values["scroll.gesture"] === "fist";
  $("#plates").innerHTML = PLATES.map(([demo, name, how]) => {
    if (fist && name === "Scroll") { demo = "fist"; how = "Make a fist and slide forward or back."; }
    const art = demo ? `<canvas class="mat-grid" data-demo="${demo}" style="--cm:8.5px" aria-hidden="true"></canvas>`
      : `<div class="keys mat-grid" style="--cm:8.5px"><div><kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>G</kbd> pause</div><div><kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Q</kbd> quit</div></div>`;
    return `<article class="plate">${art}<div class="cap"><h3>${name}</h3><p>${how}</p></div></article>`;
  }).join("");
}

/* ---- settings ---------------------------------------------------------------------------- */
const FORMAT = {
  "gain.speed": (v) => Number(v).toFixed(2) + "×",
  "click.long_ms": (v) => (v / 1000).toFixed(2) + " s",
  "scroll.units_per_mm": (v) => String(Math.round(v)),
};
async function loadSettings() {
  SET = await getJSON("/api/settings");
  const vals = { ...SET.values, "click.long_ms": SET.long_ms };
  $$("[data-setting]").forEach((el) => {
    const k = el.dataset.setting, v = vals[k];
    if (el.type === "range") { el.value = v; showRange(el); }
    else if (el.classList.contains("switch")) el.setAttribute("aria-checked", String(!!v));
    else if (el.classList.contains("seg")) $$("button", el).forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === String(v ?? ""))));
  });
  const sel = $("#cam-select");
  const cams = SET.cameras || [{ index: vals["camera.index"], fps: null }];
  sel.innerHTML = cams.map((c) => `<option value="${c.index}">${esc(c.name || (c.index === vals["camera.index"] && S.camera_name) || `Camera ${c.index}`)}${c.fps ? ` · ${c.fps} fps` : ""}${c.dark ? " · dark" : ""}</option>`).join("");
  sel.value = String(vals["camera.index"]);
  if (SET.version) $("#about-version").textContent = `Bare-hand desk mouse, P1 prototype · version ${SET.version}`;
  buildPlates();
}
function showRange(el) {
  const p = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--p", p + "%");
  const out = el.parentElement.querySelector("output");
  if (out) out.textContent = (FORMAT[el.dataset.setting] || String)(el.value);
}
function setSetting(path, value) {
  cmd("set", { path, value });
  clearTimeout(setSetting.t);
  setSetting.t = setTimeout(loadSettings, 400);
}
document.addEventListener("input", (ev) => {
  const el = ev.target;
  if (el.type === "range" && el.dataset.setting) {
    showRange(el);
    $$(`input[data-setting="${el.dataset.setting}"]`).forEach((o) => { if (o !== el) { o.value = el.value; showRange(o); } });
  }
});
document.addEventListener("change", (ev) => {
  const el = ev.target;
  if (el.type === "range" && el.dataset.setting) setSetting(el.dataset.setting, Number(el.value));
});
document.addEventListener("click", (ev) => {
  const sw = ev.target.closest(".switch[data-setting]");
  if (sw) {
    const on = sw.getAttribute("aria-checked") !== "true";
    sw.setAttribute("aria-checked", String(on));
    setSetting(sw.dataset.setting, on);
  }
  const seg = ev.target.closest(".seg[data-setting] button");
  if (seg) {
    const g = seg.parentElement;
    $$("button", g).forEach((b) => b.setAttribute("aria-checked", String(b === seg)));
    setSetting(g.dataset.setting, seg.dataset.value);
  }
});
$("#cam-select").addEventListener("change", (ev) => setSetting("camera.index", Number(ev.target.value)));
$("#cam-find").addEventListener("click", (ev) => {
  const b = ev.currentTarget;
  b.disabled = true; b.textContent = "Searching";
  cmd("cameras");
  setTimeout(() => { b.disabled = false; b.textContent = "Find cameras"; loadSettings(); }, 7000);
});

/* ---- training ---------------------------------------------------------------------------- */
const SCRIPT_TITLES = { basic: "Basic gestures", angles: "Hand angles", precision: "Precision", scroll: "Scroll check", reposition: "Reposition",
  mouse: "Show Glide your mouse" };
async function loadTraining() {
  loadPointing();
  if (!SET) SET = await getJSON("/api/settings");
  $("#sessions").innerHTML = SET.scripts.map((s) => `
    <div class="session"><div><b>${s.title}</b><span>${s.what}</span></div>
    <span class="num">${s.minutes} min</span>
    <button class="btn primary" data-record="${s.id}">Start</button></div>`).join("");
  const past = await getJSON("/api/sessions");
  $("#past").innerHTML = past.length ? past.map((p) => `<tr><td class="num">${p.when}</td>
    <td>${SCRIPT_TITLES[p.script] || p.script}</td><td>${summarize(p)}</td></tr>`).join("")
    : `<tr><td colspan="3" class="empty">No recordings yet.</td></tr>`;
}
function summarize(p) {
  if (!p.coached || !p.summary) return `<span class="num">Recorded before coaching</span>`;
  const bits = [];
  let seen = 0, trials = 0;
  for (const s of p.summary) {
    if (s.trials) { seen += s.seen; trials += s.trials; }
    if (s.check === "two" && s.check_frac != null) bits.push(`scroll pose ${Math.round(s.check_frac * 100)}%`);
  }
  if (trials) bits.unshift(`${seen}/${trials} gestures seen`);
  return `<span class="num">${bits.join(" · ") || "Saved"}</span>`;
}
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-record]");
  if (b) cmd("record", { script: b.dataset.record });
  const c = ev.target.closest("[data-coach]");
  if (c && !c.disabled) cmd("coach", { key: c.dataset.coach });
});

/* ---- coached recording -------------------------------------------------------------------- */
const NOW_WORD = { pinch8: "Pinch now", pinch12: "Pinch now", tap: "Tap now", lift: "Lift now", fist: "Fist now" };
let recShownStep = -2;
function recStatus(r) {
  const hand = !!S.hand;
  if (r.paused) return ["Paused. Press P to carry on.", "warn"];
  if (!hand && ["intro", "ready", "count", "run"].includes(r.state))
    return [r.state === "run" ? "Hand out of view. The clock is paused." : "Put your hand under the camera.", "warn"];
  switch (r.state) {
    case "intro": return r.intro_left > 0 ? ["Watch the demo first.", ""] : ["Ready? Rest your hand still for a moment, or press Space.", "info"];
    case "ready": return ["Open your hand and hold it steady.", ""];
    case "count": return [r.cued ? "Get ready…" : "Starting…", "info"];
    case "respond": return r.hit ? ["Seen", "ok"] : [NOW_WORD[r.gesture] || "Now", "ok"];
    case "result": {
      const [practice, hit] = r.last || [false, false];
      const pre = practice ? "Practice: " : "";
      return hit ? [pre + "Got it" + (practice && !r.practice ? ". Now for real." : ""), "ok"]
        : [pre + "Not seen. " + (r.tip || "Try a bit bigger."), "warn"];
    }
    case "run": {
      const left = Math.ceil(r.left);
      if (r.live == null) return [`Keep going · ${left} s left`, "info"];
      return r.live ? [`${r.live_text} · ${left} s left`, "ok"] : [`${r.live_text}. ${r.tip || ""}`, "warn"];
    }
  }
  return ["", ""];
}

function updateRecord() {
  const r = S.record;
  if (!r) return;
  const res = r.result;
  $("#rec-meta").textContent = `${r.title_session} · step ${Math.min(r.i + 1, r.n)} of ${r.n}`;
  $("#rec-title").textContent = res ? (res.path ? "Session saved" : "Nothing saved") : r.title || "";
  $("#rec-steps").innerHTML = Array.from({ length: r.n }, (_, j) =>
    `<i class="${j < r.i ? "done" : j === r.i && !res ? "here" : ""}"></i>`).join("");
  const body = $("#rec-body"), btns = $("#rec-btns");
  if (res) {
    body.style.display = "none";
    const m = res.metrics || {};
    const lines = res.analyzing ? ["Checking how well Glide read your hand…"]
      : [m.click_hits && m.click_hits !== "n/a" ? `Clicks recognised: ${m.click_hits}` : null,
        m.ghosts != null ? `Accidental clicks: ${m.ghosts}` : null,
        m.jitter_px != null ? `Cursor steadiness: ${m.jitter_px} px` : null,
        m.two_scroll_units ? "Scroll pose: scrolled as expected" : null].filter(Boolean);
    const mo = res.mouse;
    if (mo && !res.analyzing) {
      lines.length = 0;
      lines.push(mo.error === "no_mouse" ? "The pointer never moved, so no mouse seems to be plugged in. This session needs a working mouse. You can skip it: Glide also learns your speed from your clicks as you use it."
        : mo.error ? "Glide couldn't see enough mouse movement. Keep your hand and the mouse under the camera, and try again."
        : `Glide now follows your hand where your mouse sits` +
          (mo.err_now > mo.err ? `: it matches your real cursor ${Math.round((1 - mo.err / Math.max(mo.err_now, 1e-6)) * 100)}% more closely than before.` : "."));
    }
    $("#rec-how").textContent = res.path ? lines.join("   ·   ") || "Thank you. This recording will be used to tune Glide."
      : "No step was finished, so there was nothing to keep.";
    $("#rec-status").textContent = ""; $("#rec-trials").innerHTML = "";
    btns.innerHTML = `<button class="btn primary big" data-cmd="record_close">Done <kbd>Enter</kbd></button>`;
    recShownStep = -2;
    return;
  }
  body.style.display = "";
  if (recShownStep !== r.i) {
    recShownStep = r.i;
    btns.innerHTML = `
      <button class="btn primary" data-coach="go">Start <kbd>Space</kbd></button>
      <button class="btn" data-coach="toggle_pause">Pause <kbd>P</kbd></button>
      <button class="btn" data-coach="redo">Redo step <kbd>R</kbd></button>
      <button class="btn" data-coach="skip">Skip <kbd>S</kbd></button>
      <button class="btn quiet" data-coach="stop">Stop and save <kbd>Esc</kbd></button>`;
  }
  $("#rec-how").textContent = r.how || "";
  const [msg, cls] = recStatus(r);
  const st = $("#rec-status");
  st.textContent = msg; st.className = "rec-status " + cls;
  const go = $('[data-coach="go"]', btns);
  if (go) {
    go.disabled = !(r.state === "intro" || r.state === "ready");
    go.firstChild.textContent = r.state === "ready" ? "Go now " : "Start ";
  }
  const pz = $('[data-coach="toggle_pause"]', btns);
  if (pz) pz.firstChild.textContent = r.paused ? "Resume " : "Pause ";
  $("#rec-trials").innerHTML = r.cued ? `<span class="num" style="color:var(--ink-3);margin-right:6px">${r.practice ? "Practice" : "Trials"}</span>` +
    Array.from({ length: r.reps }, (_, j) => `<i class="${j < r.trials.length ? (r.trials[j] ? "done" : "miss") : j === r.trials.length && !r.practice ? "here" : ""}"></i>`).join("") : "";
  const big = $("#rec-big");
  if (r.state === "count" && r.countdown) { big.textContent = r.countdown; big.className = "big-num"; }
  else if (r.state === "respond" && r.go_age != null && r.go_age < 0.6 && !r.hit) { big.textContent = "GO"; big.className = "big-num go"; }
  else big.textContent = "";
  $("#rec-ring").innerHTML = r.state === "intro" && r.intro_left <= 0 && S.hand && !r.paused ? ringSVG(r.fill, "hold still") : "";
}

function ringSVG(frac, label) {
  const R = 46, L = 2 * Math.PI * R;
  return `<svg width="120" height="120" viewBox="0 0 120 120" aria-hidden="true">
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="var(--line)" stroke-width="3"/>
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="var(--rule)" stroke-width="7" stroke-linecap="round"
      stroke-dasharray="${(L * clamp(frac, 0, 1)).toFixed(1)} ${L.toFixed(1)}" transform="rotate(-90 60 60)"/>
    <text x="60" y="65" text-anchor="middle" fill="var(--ink)" font-family="Bahnschrift" font-size="14">${label}</text></svg>`;
}

function liveG(age, name) {
  const d = DEMOS && DEMOS[name];
  if (age == null || !d) return null;
  return envelope(age + (performance.now() - recvAt) / 1000, ...d.env);
}

function drawRecordDemo() {
  const r = S.record;
  if (!r || r.result || S.mode !== "record") return;
  let g = null;
  if (r.cued && ["ready", "count"].includes(r.state)) g = 0;
  if (r.cued && ["respond", "result"].includes(r.state)) g = liveG(r.go_age, r.demo) ?? 0;
  const t = (r.demo_t || 0) + (performance.now() - recvAt) / 1000;
  const st = drawDemo($("#rec-demo"), r.demo, t, g);
  $("#rec-note").textContent = st && st.note ? st.note : "";
}

addEventListener("keydown", (ev) => {
  if (PT) {                                               // the pointing test is up: Esc stops, Enter closes the result
    if (ev.key === "Escape" || (ev.key === "Enter" && PT.done)) { ev.preventDefault(); ptClose(); }
    return;
  }
  if (S.mode === "record" && S.record && !S.record.result) {
    const k = { " ": "go", p: "toggle_pause", r: "redo", s: "skip", Escape: "stop" }[ev.key.length === 1 ? ev.key.toLowerCase() : ev.key];
    if (k) { ev.preventDefault(); cmd("coach", { key: k }); }
  } else if (S.mode === "record" && S.record && S.record.result && (ev.key === "Enter" || ev.key === "Escape")) {
    ev.preventDefault(); cmd("record_close");                    // the result screen: Done
  } else if (S.mode === "setup" && ev.key === "Escape" && S.setup && S.setup.only_calibrate && S.has_profile) {
    ev.preventDefault(); cmd("setup_cancel");
  } else if (S.mode === "setup" && ev.key === "Enter") {
    const b = $("#setup-foot .btn.primary:not([disabled])");
    if (b) { ev.preventDefault(); b.click(); }
  }
});

/* ---- setup flow -------------------------------------------------------------------------- */
const SIGNS = [["welcome", "Welcome"], ["camera", "Camera"], ["placement", "Placement"], ["calibrate", "Calibrate"],
  ["learn", "Learn"], ["practice", "Practice"], ["done", "Done"]];
let setupShown = null;

function buildSetup(su) {
  const step = su.step;
  const only = su.only_calibrate;
  renderSigns(su);
  const vis = $("#setup-visual"), side = $("#setup-side"), foot = $("#setup-foot");
  // recalibrating keeps the old calibration until the new one is done, so it can always be cancelled
  const back = only && S.has_profile ? `<button class="btn quiet" data-cmd="setup_cancel">Cancel <kbd>Esc</kbd></button>`
    : only || step === "welcome" ? "<span></span>" : `<button class="btn quiet" data-cmd="setup_back">Back</button>`;
  const stageHTML = `<div class="stage-slot" id="setup-stage" style="position:absolute;inset:0"></div>`;
  const plate = (name) => `<div class="plate-solo mat-grid" style="--cm:12px"><canvas data-setup-demo="${name}"></canvas></div>`;
  if (step === "welcome") {
    vis.innerHTML = plate("move_slow");
    side.innerHTML = `<h1>Your hand is the mouse.</h1>
      <p class="lead">Glide watches your hand on the desk through the camera above it. Slide to move the cursor. Pinch your thumb and middle finger to click.</p>
      <p class="sub">Setup takes about three minutes: choose the camera, place your hand, teach Glide your pinch, then try each gesture.</p>`;
    foot.innerHTML = `${back}<button class="btn primary big" data-cmd="setup_next">Start setup<svg class="i"><use href="#i-arrow"/></svg></button>`;
  } else if (step === "camera") {
    vis.innerHTML = stageHTML;
    side.innerHTML = `<h1>Is this your desk camera?</h1>
      <p class="lead">You should see the desk from above, with room for your hand to slide.</p>
      <div class="cams" id="cams"><p class="sub">Looking for cameras…</p></div>`;
    foot.innerHTML = `${back}<button class="btn primary big" data-cmd="setup_next">This is my desk camera<svg class="i"><use href="#i-arrow"/></svg></button>`;
  } else if (step === "placement") {
    vis.innerHTML = stageHTML;
    side.innerHTML = `<h1>Rest your hand where you'll use it</h1>
      <p class="lead">Flat on the desk, fingers pointing away from you. Glide checks the light, the view and your room to slide.</p>
      <div class="meter"><span class="num" id="pl-score">0</span><span class="sub" style="margin:0">of 100</span></div>
      <ul class="checks" id="pl-checks"></ul>`;
    foot.innerHTML = `${back}<button class="btn primary big" id="pl-next" data-cmd="setup_next">Continue<svg class="i"><use href="#i-arrow"/></svg></button>`;
  } else if (step === "calibrate") {
    vis.innerHTML = stageHTML + `<div class="big-num" id="cal-big"></div>`;
    side.innerHTML = `<h1 id="cal-title"></h1><p class="lead" id="cal-lead"></p>
      <div class="demo-card mat-grid" style="--cm:12px"><canvas id="cal-demo"></canvas></div>
      <div class="progress"><i id="cal-progress"></i></div><p class="sub" id="cal-msg"></p>`;
    foot.innerHTML = `${back}<div class="row-btns"><button class="btn quiet" id="cal-skip" data-cmd="setup_skip">Skip the pinches</button>
      <button class="btn primary big" id="cal-next" data-cmd="setup_next" disabled>Continue<svg class="i"><use href="#i-arrow"/></svg></button></div>`;
  } else if (step === "learn") {
    vis.innerHTML = `<div class="screen"><div class="monitor" id="monitor"><span class="label">Practice screen · your real cursor stays put</span>
      <div class="dot" id="vdot"><svg viewBox="0 0 34 34"><circle id="vhold" cx="17" cy="17" r="15" fill="none" stroke="var(--state-scroll)" stroke-width="3" stroke-dasharray="0 100"/></svg></div></div></div>`;
    side.innerHTML = `<h1 id="ls-title"></h1><p class="lead" id="ls-how"></p>
      <div class="demo-card mat-grid" style="--cm:12px"><canvas id="ls-demo"></canvas></div><ul class="lessons" id="ls-list"></ul>`;
    foot.innerHTML = `${back}<div class="row-btns"><button class="btn quiet" data-cmd="setup_skip">Skip this one</button>
      <button class="btn primary big" id="ls-next" data-cmd="setup_next">Continue<svg class="i"><use href="#i-arrow"/></svg></button></div>`;
  } else if (step === "practice") {
    vis.innerHTML = `<div class="board" id="board"></div>`;
    side.innerHTML = `<h1>Try it for real</h1>
      <p class="lead">Glide now moves your real cursor. Click the highlighted circle each time, as quickly and accurately as feels comfortable.</p>
      <div id="pr-result"><p class="sub">Ten clicks. The first one starts the clock.</p></div>`;
    foot.innerHTML = `${back}<div class="row-btns"><button class="btn quiet" data-cmd="setup_next">Skip</button>
      <button class="btn primary big" id="pr-next" data-cmd="setup_next" disabled>Continue<svg class="i"><use href="#i-arrow"/></svg></button></div>`;
    startPractice();
  } else if (step === "done") {
    vis.innerHTML = plate("pinch_mid");
    side.innerHTML = `<h1>Glide is on.</h1>
      <p class="lead">Your hand is the mouse now. You can close this window: Glide keeps running in the tray.</p>
      <ul class="checks"><li class="ok"><svg class="i"><use href="#i-check"/></svg><span><kbd>Ctrl</kbd> <kbd>Alt</kbd> <kbd>G</kbd> pauses or resumes, anywhere</span></li>
      <li class="ok"><svg class="i"><use href="#i-check"/></svg><span><kbd>Ctrl</kbd> <kbd>Alt</kbd> <kbd>Q</kbd> quits Glide</span></li>
      <li class="ok"><svg class="i"><use href="#i-check"/></svg><span>Gestures and settings are one click away in this window</span></li></ul>`;
    foot.innerHTML = `<span></span><button class="btn primary big" data-cmd="setup_finish">Start using Glide<svg class="i"><use href="#i-arrow"/></svg></button>`;
  }
}

/* one band per stage, the current one lit like a departure sign; recalibration shows its own stages */
function renderSigns(su) {
  let signs = SIGNS, here = su.step;
  if (su.only_calibrate) {
    const c = su.calibrate || {};
    signs = [["rest", "Hold still"], ["clicks", "Five pinches"], ["done", "Done"]];
    here = c.done ? "done" : c.step || "rest";
  }
  const idx = signs.findIndex(([k]) => k === here);
  const html = signs.map(([k, name], i) => `<span class="${i === idx ? "here" : i < idx ? "done" : ""}">` +
    `${i < idx ? '<svg class="i"><use href="#i-check"/></svg>' : ""}${name}</span>`).join("");
  const el = $("#signs");
  if (el.dataset.sig !== html) { el.dataset.sig = html; el.innerHTML = html; }
}

function updateSetup() {
  const su = S.setup;
  if (!su) return;
  const key = su.step + (su.only_calibrate ? "*" : "");
  if (setupShown !== key) { setupShown = key; buildSetup(su); placeStage(); }
  renderSigns(su);
  if (su.step === "camera") {
    const cams = su.cameras;
    if (cams) {
      const box = $("#cams");
      const sig = JSON.stringify(cams) + S.camera_index;
      if (box.dataset.sig !== sig) {
        box.dataset.sig = sig;
        box.innerHTML = cams.length ? cams.map((c) => `<button class="cam" role="radio" aria-checked="${c.index === S.camera_index}" data-cam="${c.index}">
          ${c.active ? '<span class="ph" style="display:grid;place-items:center"><svg class="i"><use href="#i-camera"/></svg></span>' : `<img src="${withToken(`/api/thumb/${c.index}.jpg`)}" alt="">`}
          <span><b>${esc(c.name || `Camera ${c.index}`)}${c.index === S.camera_index ? " · showing now" : ""}</b><span>${c.fps} fps${c.dark ? " · too dark" : ""}${c.built_in ? " · the laptop's own camera" : ""}</span></span></button>`).join("")
          : `<p class="sub">No other camera found. Plug the camera in and close apps that might be using it.</p>`;
      }
    }
  } else if (su.step === "placement" && su.placement) {
    const p = su.placement;
    $("#pl-score").textContent = p.score;
    $("#pl-checks").innerHTML = p.checks.map((c) => `<li class="${c.ok ? "ok" : "no"}"><svg class="i"><use href="#i-${c.ok ? "check" : "alert"}"/></svg>
      <span>${c.name}</span>${c.tip ? `<small>${c.tip}</small>` : ""}</li>`).join("");
    const nb = $("#pl-next");
    nb.firstChild.textContent = p.score >= 85 ? "Continue" : "Continue anyway";
    nb.classList.toggle("primary", p.score >= 70);
  } else if (su.step === "calibrate" && su.calibrate) {
    const c = su.calibrate;
    const title = c.done ? "Calibrated" : c.step === "rest" ? "Hold your hand still" : "Pinch on GO";
    $("#cal-title").textContent = title;
    $("#cal-lead").textContent = c.done ? "Glide has learned your hand and your pinch."
      : c.step === "rest" ? "Flat on the desk, fingers pointing away from you, completely still."
      : `Touch your thumb tip to your middle fingertip, then open. Pinch ${Math.min(c.cues + (c.waiting ? 1 : 0), 5) || 1} of 5.`;
    $("#cal-progress").style.width = ((c.done ? 1 : c.progress) * 100).toFixed(0) + "%";
    $("#cal-msg").textContent = c.message || (c.waiting ? "Open your hand to start the next pinch." : "");
    const big = $("#cal-big");
    big.textContent = c.countdown ? c.countdown : c.go ? "GO" : "";
    big.className = "big-num" + (c.go && !c.countdown ? " go" : "");
    $("#cal-next").disabled = !c.done;
    $("#cal-skip").hidden = c.step !== "clicks" || c.done;
  } else if (su.step === "learn" && su.learn) {
    const l = su.learn, cur = l.lessons[Math.min(l.i, l.lessons.length - 1)];
    $("#ls-title").textContent = l.finished ? "All gestures done" : cur.name;
    $("#ls-how").textContent = l.finished ? "Next, try them with your real cursor." : cur.how;
    $("#ls-list").innerHTML = l.lessons.map((x, j) => `<li class="${x.done ? "done" : j === l.i ? "here" : ""}">
      <svg class="i"><use href="#i-${x.done ? "check" : "ring"}"/></svg>${x.name}</li>`).join("");
    const dot = $("#vdot"), mon = $("#monitor");
    const color = `var(--${(STATES[l.state] || STATES.TRACK)[2]})`;
    dot.style.transform = `translate(${l.cursor[0] * mon.clientWidth}px, ${l.cursor[1] * mon.clientHeight}px)`;
    dot.style.borderColor = color; dot.style.color = color;
    dot.classList.toggle("filled", ["PINCH", "DRAG", "HOLD"].includes(l.state));
    $("#vhold").setAttribute("stroke-dasharray", `${(l.state === "PINCH" ? l.hold : 0) * 94.2} 100`);
    $("#ls-next").firstChild.textContent = l.finished ? "Continue" : "Skip the rest";
  }
}
document.addEventListener("click", (ev) => {
  const c = ev.target.closest("[data-cam]");
  if (c) cmd("set", { path: "camera.index", value: Number(c.dataset.cam) });
});

function drawSetupDemos(now) {
  const su = S.setup;
  if (!su || S.mode !== "setup") return;
  $$("[data-setup-demo]").forEach((cv) => drawDemo(cv, cv.dataset.setupDemo, now / 1000, null, { scale: Math.min(cv.clientWidth / 300, cv.clientHeight / 300) }));
  if (su.step === "calibrate" && su.calibrate) {
    const c = su.calibrate;
    if (c.step === "clicks" && !c.done) drawDemo($("#cal-demo"), "pinch_mid", now / 1000, c.go_age != null ? liveG(c.go_age, "pinch_mid") : 0);
    else drawDemo($("#cal-demo"), "rest", now / 1000);
  }
  if (su.step === "learn" && su.learn) {
    const l = su.learn, cur = l.lessons[Math.min(l.i, l.lessons.length - 1)];
    drawDemo($("#ls-demo"), cur.demo, now / 1000, null, { holdRing: cur.demo === "hold" });
  }
}

/* practice: ISO 9241-411 multi-directional task, measured in the page with the real cursor */
const N_T = 9, ORDER = Array.from({ length: N_T + 1 }, (_, k) => (k * (Math.floor(N_T / 2) + 1)) % N_T);
let PR = null;
function startPractice() {
  PR = { i: 0, trials: [], t0: 0, last: null };
  const board = $("#board");
  const place = () => {
    const w = board.clientWidth, h = board.clientHeight, R = Math.min(w, h) * 0.36;
    PR.pos = Array.from({ length: N_T }, (_, k) => [w / 2 + R * Math.sin((2 * Math.PI * k) / N_T), h / 2 - R * Math.cos((2 * Math.PI * k) / N_T)]);
    board.innerHTML = PR.pos.map(([x, y], k) => `<div class="target" data-k="${k}" style="left:${x}px;top:${y}px"></div>`).join("");
    mark();
  };
  const mark = () => $$(".target", board).forEach((t) => t.classList.toggle("next", PR.i <= N_T && Number(t.dataset.k) === ORDER[PR.i]));
  requestAnimationFrame(place);
  board.onpointerdown = (ev) => {
    if (ev.button !== 0 || PR.i > N_T) return;
    const rect = board.getBoundingClientRect(), c = [ev.clientX - rect.left, ev.clientY - rect.top];
    const now = performance.now();
    if (PR.i > 0) PR.trials.push({ a: PR.pos[ORDER[PR.i - 1]], b: PR.pos[ORDER[PR.i]], c, t: (now - PR.t0) / 1000 });
    PR.t0 = now; PR.i += 1; mark();
    if (PR.i > N_T) showPractice();
  };
}
function throughput(trials) {
  const dx = [], de = [], mt = [];
  for (const { a, b, c, t } of trials) {
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, u = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
    const proj = (c[0] - a[0]) * u[0] + (c[1] - a[1]) * u[1];
    dx.push(proj - L); de.push(proj); mt.push(t);
  }
  const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const sd = Math.sqrt(mean(dx.map((x) => (x - mean(dx)) ** 2)));
  const we = Math.max(4.133 * sd, 1), ide = Math.log2(mean(de) / we + 1);
  return [ide / mean(mt), mean(mt)];
}
function showPractice() {
  const [tp, mt] = throughput(PR.trials);
  const misses = PR.trials.filter(({ b, c }) => Math.hypot(c[0] - b[0], c[1] - b[1]) > 28).length;
  $("#pr-result").innerHTML = `<div class="meter"><span class="result-num">${tp.toFixed(1)}</span><span class="sub" style="margin:0">bits per second</span></div>
    <dl class="facts"><dt>Average time per click</dt><dd>${mt.toFixed(2)} s</dd><dt>Missed circles</dt><dd>${misses} of ${PR.trials.length}</dd></dl>
    <p class="sub">A hand mouse usually scores 1 to 3 bits per second at first, and you will get faster with practice.</p>`;
  $("#pr-next").disabled = false;
}

/* ---- pointing test: ISO 9241-411 multi-directional tapping ------------------------------- */
// Four rounds (two distances x two circle sizes) of nine circles. The first click of a round starts
// its clock, so each round gives nine timed trials. Scored per round as IDe / MT with the effective
// width We = 4.133 x SD of the endpoints along the movement axis, then averaged (glide/pointing.py
// recomputes and saves the same numbers for the test report).
const PT_ROUNDS = [[1.0, 32], [1.0, 64], [0.5, 32], [0.5, 64]];   // [distance, share of the largest; circle px]
const PT_DEVICE = { glide: "With Glide", mouse: "With mouse or touchpad" };
let PT = null;
function ptOpen(device) {
  PT = { device, round: 0, i: 0, t0: 0, rounds: PT_ROUNDS.map(() => []), wasOn: !!S.enabled, done: false };
  cmd("enable", { on: device === "glide" });             // the other device's test: keep the hand out of it
  $("#pointing").classList.add("on");
  $("#pt-result").hidden = true;
  $("#pt-btns").innerHTML = `<button class="btn quiet" data-pt="close">Stop <kbd>Esc</kbd></button>`;
  requestAnimationFrame(ptRound);
}
function ptRound() {
  const board = $("#pt-board"), [share, w] = PT_ROUNDS[PT.round];
  const bw = board.clientWidth, bh = board.clientHeight, R = (Math.min(bw, bh) / 2 - 56) * share;
  Object.assign(PT, { i: 0, w, bw, bh });
  PT.pos = Array.from({ length: N_T }, (_, k) => [bw / 2 + R * Math.sin((2 * Math.PI * k) / N_T), bh / 2 - R * Math.cos((2 * Math.PI * k) / N_T)]);
  board.innerHTML = PT.pos.map(([x, y], k) => `<div class="target" data-k="${k}" style="left:${x}px;top:${y}px;--w:${w}px"></div>`).join("")
    + `<div class="start-tip" id="pt-tip"></div>`;
  $("#pt-meta").textContent = `${PT_DEVICE[PT.device]} · round ${PT.round + 1} of ${PT_ROUNDS.length}`;
  $("#pt-rounds").innerHTML = PT_ROUNDS.map((_, j) => `<i class="${j < PT.round ? "done" : j === PT.round ? "here" : ""}"></i>`).join("");
  $("#pt-hint").textContent = PT.round === 0
    ? "Click the yellow circle to start. Then click each new yellow circle as quickly and accurately as you can."
    : "Next round: click the yellow circle to start it.";
  ptMark();
}
function ptMark() {
  $$(".target", $("#pt-board")).forEach((t) => t.classList.toggle("next", PT.i <= N_T && Number(t.dataset.k) === ORDER[PT.i]));
  const tip = $("#pt-tip"), [x, y] = PT.pos[ORDER[0]];
  tip.style.left = x + "px"; tip.style.top = (y - PT.w / 2) + "px";
  tip.textContent = PT.i === 0 ? "Start here" : "";
  if (PT.i === 1) $("#pt-hint").textContent = "Keep going: click each new yellow circle. A miss counts too, so just carry on.";
}
$("#pt-board").addEventListener("pointerdown", (ev) => {
  if (!PT || PT.done || ev.button !== 0) return;
  const board = ev.currentTarget, rect = board.getBoundingClientRect();
  const c = [ev.clientX - rect.left, ev.clientY - rect.top], now = performance.now();
  if (PT.i === 0) {                                      // the clock starts on the start circle itself
    const s = PT.pos[ORDER[0]];
    if (Math.hypot(c[0] - s[0], c[1] - s[1]) > PT.w / 2 + 8) return;
  } else {
    const a = PT.pos[ORDER[PT.i - 1]], b = PT.pos[ORDER[PT.i]];
    PT.rounds[PT.round].push({ a, b, c, t: (now - PT.t0) / 1000 });
    const miss = Math.hypot(c[0] - b[0], c[1] - b[1]) > PT.w / 2;
    board.insertAdjacentHTML("beforeend", `<i class="hit${miss ? " miss" : ""}" style="left:${c[0]}px;top:${c[1]}px"></i>`);
  }
  PT.t0 = now; PT.i += 1;
  if (PT.i <= N_T) { ptMark(); return; }
  PT.round += 1;
  if (PT.round < PT_ROUNDS.length) ptRound(); else ptFinish();
});
function ptScore(trials, w) {
  const [tp, mt] = throughput(trials);
  return { tp, mt, w, errors: trials.filter(({ b, c }) => Math.hypot(c[0] - b[0], c[1] - b[1]) > w / 2).length, n: trials.length };
}
async function ptFinish() {
  PT.done = true;
  const per = PT.rounds.map((tr, j) => ptScore(tr, PT_ROUNDS[j][1]));
  const avg = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const tp = avg(per.map((p) => p.tp)), mt = avg(per.map((p) => p.mt));
  const errors = per.reduce((s, p) => s + p.errors, 0), n = per.reduce((s, p) => s + p.n, 0);
  cmd("enable", { on: PT.wasOn });
  $("#pt-rounds").innerHTML = PT_ROUNDS.map(() => `<i class="done"></i>`).join("");
  $("#pt-meta").textContent = `${PT_DEVICE[PT.device]} · finished`;
  const res = $("#pt-result");
  res.innerHTML = `<div><div class="meter"><span class="result-num">${tp.toFixed(2)}</span><span class="sub" style="margin:0">bits per second</span></div>
      <dl class="facts"><dt>Average time per click</dt><dd>${mt.toFixed(2)} s</dd><dt>Clicks outside the circle</dt><dd>${errors} of ${n} (${Math.round((100 * errors) / n)}%)</dd></dl>
      <p class="sub" id="pt-vs" style="margin-top:16px">Saving…</p></div>
    <table><thead><tr><th>Round</th><th>Circle</th><th>Distance</th><th>Time</th><th>Missed</th><th>bits/s</th></tr></thead><tbody>
      ${per.map((p, j) => `<tr><td class="num">${j + 1}</td><td class="num">${p.w} px</td><td class="num">${PT_ROUNDS[j][0] === 1 ? "far" : "near"}</td>
        <td class="num">${p.mt.toFixed(2)} s</td><td class="num">${p.errors}/${p.n}</td><td class="num">${p.tp.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
  res.hidden = false;
  $("#pt-hint").textContent = "Throughput is the standard measure for comparing pointing devices: higher is better.";
  $("#pt-btns").innerHTML = `<button class="btn" data-pt="again">Run again</button><button class="btn primary" data-pt="close">Done <kbd>Enter</kbd></button>`;
  await cmd("pointing_save", { device: PT.device, rounds: PT.rounds.map((tr, j) => ({ w: PT_ROUNDS[j][1], trials: tr })), board: [PT.bw, PT.bh] });
  const hist = await getJSON("/api/pointing");
  const vs = $("#pt-vs");
  if (vs) vs.textContent = ptCompareText(hist.compare) || "Saved. Run it with the other device too, to compare them.";
}
function ptCompareText(c) {
  return c ? `Glide reaches ${Math.round(c.ratio * 100)}% of your mouse or touchpad (${c.glide.toFixed(2)} vs ${c.mouse.toFixed(2)} bits/s). The project's target is 80%.` : "";
}
function ptClose() {
  if (PT && !PT.done) cmd("enable", { on: PT.wasOn });
  PT = null;
  $("#pointing").classList.remove("on");
  if ($("#view-training").classList.contains("on")) loadTraining();
}
document.addEventListener("click", (ev) => {
  const p = ev.target.closest("[data-pointing]");
  if (p) ptOpen(p.dataset.pointing);
  const b = ev.target.closest("[data-pt]");
  if (b && b.dataset.pt === "close") ptClose();
  if (b && b.dataset.pt === "again" && PT) ptOpen(PT.device);
});
async function loadPointing() {
  const hist = await getJSON("/api/pointing");
  const box = $("#pt-compare"), c = hist.compare, last = hist.tests;
  if (!last.length) { box.hidden = true; return; }
  const g = last.find((t) => t.device === "glide"), m = last.find((t) => t.device === "mouse");
  const line = (t, label) => t ? `${label} <b class="num">${t.tp.toFixed(2)}</b> bits/s · ${t.mt.toFixed(2)} s per click · ${Math.round(t.error_rate * 100)}% missed` : "";
  box.innerHTML = c
    ? `<b>${ptCompareText(c)}</b><div class="bar"><i style="width:${Math.min(c.ratio, 1) * 100}%"></i><s style="left:80%" title="target: 80%"></s></div>
       <span>${line(g, "Glide")}<br>${line(m, "Mouse or touchpad")}</span>`
    : `<b>${g ? line(g, "Glide") : line(m, "Mouse or touchpad")}</b><br><span>Run the test with ${g ? "your mouse or touchpad" : "Glide"} too, to compare them.</span>`;
  box.hidden = false;
}

/* ---- live state --------------------------------------------------------------------------- */
let lastMode = null, lastToast = null;
function onState(s) {
  S = s; recvAt = performance.now();
  if (S.mode !== lastMode) {
    lastMode = S.mode;
    $("#setup").classList.toggle("on", S.mode === "setup");
    $("#record").classList.toggle("on", S.mode === "record");
    if (S.mode !== "setup") setupShown = null;
    if (S.mode !== "record") recShownStep = -2;
    if (S.mode === "run" && $("#view-training").classList.contains("on")) loadTraining();
    placeStage();
  }
  if (S.toast && S.toast !== lastToast) toast(S.toast);
  lastToast = S.toast;
  updateHome();
  if (S.mode === "setup") updateSetup();
  if (S.mode === "record") updateRecord();
}

function connect() {
  const es = new EventSource(withToken("/events"));
  es.onmessage = (ev) => onState(JSON.parse(ev.data));
  es.onerror = () => { es.close(); setTimeout(connect, 1000); };
}

function frame(now) {
  drawStage();
  if ($("#view-gestures").classList.contains("on") && S.mode === "run") {
    $$("canvas[data-demo]").forEach((cv) => drawDemo(cv, cv.dataset.demo, now / 1000, null,
      { scale: 0.85, holdRing: cv.dataset.demo === "hold" }));
  }
  drawSetupDemos(now);
  drawRecordDemo();
  requestAnimationFrame(frame);
}

(async function start() {
  route();
  DEMOS = await getJSON(`/api/demos?hand=right`);
  await loadSettings();
  if (SET.values["hand.dominant"] === "left") DEMOS = await getJSON(`/api/demos?hand=left`);
  connect();
  requestAnimationFrame(frame);
})();
