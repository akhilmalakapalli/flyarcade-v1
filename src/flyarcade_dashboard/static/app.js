"use strict";
// FlyArcade dashboard: vanilla JS, Canvas 2D, polling. Visualization only.

const $ = (id) => document.getElementById(id);
const BASE_RATE = 6;          // decisions per second at 1x
const RENDER_FPS = 30;
const TRACE_LEN = 100;
const POP = { vp: "visual_projection", cx: "cb_intrinsic", dn: "descending_neuron" };
const POP_COLORS = { vp: "#4f9cf9", cx: "#f1b44c", dn: "#34c38f" };
const PROB_COLORS = ["#e377c2", "#bcbd22", "#17becf", "#ff7f0e"];

const S = {
  tasks: [], network: null, task: null, state: null,
  running: false, inflight: false, speed: 1, timer: null,
  frame: null, frameAt: 0, counts: null,
  traces: [], episodeSum: null, episodeSteps: 0,
  view: "2d", yaw: 0.6, pitch: -0.35, zoom: 1.0, drag: null,
  projected: null, selected: null,
  replay: null, replayIndex: 0, dirtyNeural: true,
};

window.addEventListener("error", (e) => showError(`JS error: ${e.message}`));
window.addEventListener("unhandledrejection", (e) => showError(`JS error: ${e.reason}`));

function showError(message) {
  const bar = $("error-bar");
  bar.textContent = message;
  bar.hidden = !message;
  if (message) document.body.dataset.jsError = message;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

// ------------------------------------------------------------------ colour
const PALETTE = (() => {
  const stops = [[11, 14, 20], [30, 50, 120], [30, 150, 200], [240, 200, 60], [255, 255, 240]];
  const out = [];
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
    const c = stops[k].map((v, j) => Math.round(v + (stops[k + 1][j] - v) * f));
    out.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return out;
})();

function decodeCounts(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function popKeyOf(index) {
  const p = S.network.meta[index].population;
  return p === POP.vp ? "vp" : p === POP.cx ? "cx" : "dn";
}

function intensity(index) {
  if (!S.counts || !S.frame) return 0;
  const scales = currentScales();
  const fraction = S.counts[index] / S.frame.ticks;
  return Math.max(0, Math.min(1, fraction / scales[popKeyOf(index)]));
}

function currentScales() {
  const cal = S.replay ? S.replay.calibration : S.state && S.state.calibration;
  return cal ? cal.scales : { vp: 1, cx: 1, dn: 1 };
}

// ------------------------------------------------------------------ setup
async function init() {
  const [tasks, network] = await Promise.all([api("/api/tasks"), api("/api/network")]);
  S.tasks = tasks;
  S.network = network;
  S.popIndex = {};
  for (const key of Object.keys(POP)) S.popIndex[key] = network.order[POP[key]];
  $("network-label").textContent = network.label;
  $("activity-note").textContent = network.note;
  const nav = $("task-buttons");
  for (const t of tasks) {
    const b = document.createElement("button");
    b.textContent = t.title;
    b.dataset.task = t.task;
    if (!t.available) { b.classList.add("unavail"); b.title = t.unavailable_reason; }
    b.onclick = () => selectTask(t.task);
    nav.appendChild(b);
  }
  $("btn-run").onclick = run;
  $("btn-pause").onclick = pause;
  $("btn-step").onclick = () => { pause(); stepOnce(); };
  $("btn-reset").onclick = () => sessionCall("/api/session/reset");
  $("btn-new").onclick = () => sessionCall("/api/session/new_demo");
  $("btn-save").onclick = saveEpisode;
  $("btn-replay").onclick = openReplays;
  $("picker-close").onclick = () => ($("replay-picker").hidden = true);
  $("speed").onchange = (e) => { S.speed = Number(e.target.value); if (S.running) schedule(0); };
  $("model-seed").onchange = (e) => selectTask(S.task, Number(e.target.value));
  $("view-2d").onclick = () => setView("2d");
  $("view-3d").onclick = () => setView("3d");
  setupNeuralInteraction();
  drawLegend();
  // Optional URL parameters: ?task=pong&view=3d&autorun=1&speed=2 (bookmarkable).
  const params = new URLSearchParams(location.search);
  const wanted = tasks.find((t) => t.task === params.get("task"));
  const first = wanted || tasks.find((t) => t.available) || tasks[0];
  if (params.get("speed")) { S.speed = Number(params.get("speed")); $("speed").value = params.get("speed"); }
  await selectTask(first.task);
  setView(params.get("view") === "3d" ? "3d" : "2d");
  document.body.dataset.ready = "1";
  requestAnimationFrame(renderLoop);
  if (params.get("autorun") === "1") run();
}

async function selectTask(task, modelSeed) {
  pause();
  exitReplay();
  const info = S.tasks.find((t) => t.task === task);
  const seedSelect = $("model-seed");
  if (S.task !== task) {
    seedSelect.innerHTML = "";
    for (const s of info.model_seeds) {
      const o = document.createElement("option");
      o.value = s; o.textContent = s; seedSelect.appendChild(o);
    }
  }
  S.task = task;
  document.querySelectorAll("#task-buttons button").forEach((b) =>
    b.classList.toggle("active", b.dataset.task === task));
  const state = await api("/api/session/new", { task, model_seed: modelSeed ?? null, demo_index: 0 });
  applyState(state);
}

async function sessionCall(path) {
  const wasRunning = S.running;
  pause();
  exitReplay();
  applyState(await api(path, {}));
  if (wasRunning) run();
}

function applyState(state) {
  S.state = state;
  S.frame = null; S.counts = null; S.traces = [];
  S.episodeSum = new Float64Array(S.network.n); S.episodeSteps = 0;
  S.dirtyNeural = true;
  $("game-title").textContent = state.title || "Game";
  const available = state.available;
  $("unavailable").hidden = available;
  $("unavailable").textContent = state.unavailable_reason || "";
  ["btn-run", "btn-step", "btn-reset", "btn-new", "btn-save"].forEach((id) => ($(id).disabled = !available));
  $("model-seed").disabled = !available;
  if (available) {
    const m = state.model;
    $("model-info").textContent = `${m.study} · training seed ${m.training_seed}`;
    $("game-caption").textContent =
      `Demo seed ${state.demo_seed} (demo #${state.demo_index}) · ${m.architecture} · ${m.ticks} ticks/decision · ${m.readout}`;
    $("model-seed").value = state.model_seed;
  } else {
    $("model-info").textContent = "";
    $("game-caption").textContent = "Initial board only; no trained policy is run.";
  }
  const cal = state.calibration;
  $("legend-text").textContent = cal
    ? `spike fraction ÷ fixed scale (VP ${cal.scales.vp.toFixed(3)}, CX ${cal.scales.cx.toFixed(3)}, DN ${cal.scales.dn.toFixed(3)}; ${cal.statistic}, calibration seed ${cal.seed})`
    : "no activity (no model)";
  updateDecision(null);
  drawTraces();
  drawHeatmaps();
}

// ------------------------------------------------------------------ live loop
function interval() { return 1000 / (BASE_RATE * S.speed); }

function run() {
  if (S.replay) { S.running = true; schedule(0); setBadge(); return; }
  if (!S.state || !S.state.available) return;
  S.running = true;
  setBadge();
  schedule(0);
}

function pause() {
  S.running = false;
  clearTimeout(S.timer);
  setBadge();
}

function schedule(delay) {
  clearTimeout(S.timer);
  S.timer = setTimeout(tick, delay);
}

async function tick() {
  if (!S.running) return;
  const started = performance.now();
  if (S.replay) {
    advanceReplay();
  } else {
    const result = await stepOnce();
    if (result && result.done) {
      // Hold the final frame briefly, then continue with the next demo episode.
      S.timer = setTimeout(async () => {
        if (!S.running) return;
        applyState(await api("/api/session/new_demo", {}));
        schedule(0);
      }, 1200);
      return;
    }
  }
  schedule(Math.max(0, interval() - (performance.now() - started)));
}

async function stepOnce() {
  if (S.inflight || !S.state || !S.state.available || S.replay) return null;
  S.inflight = true;
  try {
    const result = await api("/api/session/step", {});
    if (result.frame && result.frame.step !== (S.frame ? S.frame.step : -1)) showFrame(result.frame);
    return result;
  } catch (error) {
    showError(error.message);
    pause();
    return null;
  } finally {
    S.inflight = false;
  }
}

function showFrame(frame) {
  S.frame = frame;
  S.frameAt = performance.now();
  S.counts = decodeCounts(frame.counts_b64);
  for (let i = 0; i < S.counts.length; i++) S.episodeSum[i] += S.counts[i] / frame.ticks;
  S.episodeSteps += 1;
  S.traces.push({ means: frame.population_means, probs: frame.probs });
  if (S.traces.length > TRACE_LEN) S.traces.shift();
  S.dirtyNeural = true;
  updateDecision(frame);
  drawHeatmaps();
  drawTraces();
  refreshInspector();
  document.body.dataset.frames = String((Number(document.body.dataset.frames) || 0) + 1);
}

// ------------------------------------------------------------------ replay
async function openReplays() {
  pause();
  const list = await api("/api/replays");
  const ul = $("replay-list");
  ul.innerHTML = "";
  if (!list.length) ul.innerHTML = "<li>No saved episodes yet. Use SAVE EPISODE first.</li>";
  for (const r of list) {
    const li = document.createElement("li");
    li.textContent = `${r.name} · ${r.steps} steps`;
    li.onclick = () => loadReplay(r.name);
    ul.appendChild(li);
  }
  $("replay-picker").hidden = false;
}

async function loadReplay(name) {
  $("replay-picker").hidden = true;
  const data = await api("/api/replay/load", { name });
  S.replay = data;
  S.replayIndex = 0;
  S.task = data.task;
  document.querySelectorAll("#task-buttons button").forEach((b) =>
    b.classList.toggle("active", b.dataset.task === data.task));
  applyState({
    title: `${S.tasks.find((t) => t.task === data.task).title} (replay)`, available: true,
    model: data.model, model_seed: data.model_seed, demo_seed: data.demo_seed,
    demo_index: data.demo_index, calibration: data.calibration,
  });
  $("btn-new").disabled = true; $("btn-save").disabled = true;
  $("btn-reset").onclick = () => { S.replayIndex = 0; applyState(S.state); };
  $("btn-step").onclick = () => { pause(); if (S.replay) advanceReplay(); else stepOnce(); };
  setBadge();
  advanceReplay();
}

function advanceReplay() {
  if (!S.replay) return;
  if (S.replayIndex >= S.replay.frames.length) { pause(); return; }
  showFrame(S.replay.frames[S.replayIndex]);
  S.replayIndex += 1;
}

function exitReplay() {
  if (!S.replay) return;
  S.replay = null;
  $("btn-reset").onclick = () => sessionCall("/api/session/reset");
  $("btn-step").onclick = () => { pause(); stepOnce(); };
  setBadge();
}

async function saveEpisode() {
  try {
    const { name } = await api("/api/session/save", {});
    $("step-label").textContent = `saved ${name}`;
  } catch (error) { showError(error.message); }
}

function setBadge() {
  const badge = $("mode-badge");
  badge.textContent = S.replay ? (S.running ? "REPLAY ▶" : "REPLAY") : (S.running ? "LIVE ▶" : "LIVE");
  badge.classList.toggle("replay", !!S.replay);
}

// ------------------------------------------------------------------ decision
function updateDecision(frame) {
  const labels = (S.state && S.state.actions) || (S.replay && S.tasks.find((t) => t.task === S.replay.task).actions) || [];
  $("d-task").textContent = S.state && S.state.title ? S.state.title : "—";
  const bars = $("prob-bars");
  bars.innerHTML = "";
  if (!frame) {
    ["d-step", "d-action", "d-reward", "d-return", "d-score"].forEach((id) => ($(id).textContent = "—"));
    return;
  }
  $("d-step").textContent = frame.step;
  let action = frame.action_label;
  if (frame.task === "snake") {
    const opposite = { 0: 1, 1: 0, 2: 3, 3: 2 };
    if (frame.render_before.body.length > 1 && frame.action === opposite[frame.render_before.heading]) {
      action += " (reversal refused by the environment)";
    }
  }
  $("d-action").textContent = action;
  $("d-reward").textContent = frame.reward.toFixed(3);
  $("d-return").textContent = frame.episode_return.toFixed(3);
  $("d-score").textContent = scoreText(frame);
  $("step-label").textContent = frame.done ? "episode finished" : "";
  frame.probs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "bar" + (i === frame.action ? " chosen" : "");
    row.innerHTML = `<span>${labels[i] || i}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${p.toFixed(3)}</span>`;
    bars.appendChild(row);
  });
}

function scoreText(frame) {
  const m = frame.metrics;
  switch (frame.task) {
    case "catch": case "dodge":
      return `${m.score} / ${m.landings} landings (success ${m.success.toFixed(3)})`;
    case "pong":
      return `hits ${m.returns} / ${m.attempted_returns} returns (success ${m.success.toFixed(3)})`;
    case "flappy":
      return `obstacles passed ${m.obstacles_passed} / 6 (success ${m.success.toFixed(3)})${m.death ? " · " + m.death : ""}`;
    case "snake":
      return `food ${m.food} · length ${m.length} · steps ${m.steps}${m.death ? " · " + m.death : ""}`;
    default:
      return JSON.stringify(m);
  }
}

// ------------------------------------------------------------------ game
function lerp(a, b, t) { return a + (b - a) * t; }

function renderLoop(now) {
  if (!renderLoop.last || now - renderLoop.last >= 1000 / RENDER_FPS) {
    renderLoop.last = now;
    drawGame(now);
    if (S.dirtyNeural || S.view === "3d") { drawNeural(); S.dirtyNeural = false; }
  }
  requestAnimationFrame(renderLoop);
}

function drawGame(now) {
  const canvas = $("game"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);
  let before, after, t = 1;
  if (S.frame) {
    before = S.frame.render_before; after = S.frame.render_after;
    const span = S.running ? interval() : 180;
    t = Math.max(0, Math.min(1, (now - S.frameAt) / span));
  } else if (S.state && S.state.render) {
    before = after = S.state.render;
  } else return;
  const task = S.replay ? S.replay.task : S.task;
  ({ catch: drawLanes, dodge: drawLanes, pong: drawPong, flappy: drawFlappy,
     snake: drawSnake, breakout: drawBreakout })[task](ctx, W, H, before, after, t, task);
  const caption = S.frame ? `decision ${S.frame.step}: observed state → ${S.frame.action_label}` : "";
  ctx.fillStyle = "#98a2b3"; ctx.font = "12px sans-serif"; ctx.fillText(caption, 10, H - 10);
}

function text(ctx, s, x, y, size = 14, color = "#e6e9ef") {
  ctx.fillStyle = color; ctx.font = `${size}px sans-serif`; ctx.fillText(s, x, y);
}

function drawLanes(ctx, W, H, a, b, t, task) {
  const lanes = a.lanes, laneW = W / lanes, top = 40, bottom = H - 60;
  for (let i = 0; i < lanes; i++) {
    ctx.fillStyle = i % 2 ? "#121722" : "#0f141d";
    ctx.fillRect(i * laneW, top, laneW, bottom - top + 30);
  }
  const phaseY = (p) => top + ((p + 1) / 4) * (bottom - top - 20);
  // Object: tween down within an object's fall; a new object appears without tweening.
  const newObject = b.phase < a.phase || b.object !== a.object;
  const oy = newObject ? (t < 0.5 ? phaseY(a.phase) : phaseY(b.phase)) : lerp(phaseY(a.phase), phaseY(b.phase), t);
  const ox = (newObject && t >= 0.5 ? b.object : a.object) * laneW + laneW / 2;
  ctx.fillStyle = task === "catch" ? "#34c38f" : "#f46a6a";
  ctx.beginPath(); ctx.arc(ox, oy, laneW * 0.18, 0, Math.PI * 2); ctx.fill();
  const px = lerp(a.player, b.player, t) * laneW + laneW / 2;
  ctx.fillStyle = "#4f9cf9";
  ctx.fillRect(px - laneW * 0.32, bottom, laneW * 0.64, 18);
  text(ctx, `${task === "catch" ? "Catch" : "Dodge"} · step ${b.time}/${b.horizon}`, 10, 24);
}

function drawPong(ctx, W, H, a, b, t) {
  const pad = 30, cw = W - 2 * pad, ch = H - 2 * pad - 20;
  ctx.strokeStyle = "#262c37"; ctx.strokeRect(pad, pad, cw, ch);
  ctx.setLineDash([4, 6]); ctx.beginPath(); ctx.moveTo(pad + cw / 2, pad); ctx.lineTo(pad + cw / 2, pad + ch); ctx.stroke(); ctx.setLineDash([]);
  const serve = Math.abs(b.x - a.x) > 0.2;
  const k = serve ? (t < 0.5 ? 0 : 1) : t;
  const bx = lerp(a.x, b.x, k), by = lerp(a.y, b.y, k);
  const paddle = lerp(a.paddle, b.paddle, t);
  const halfLen = 0.14 * ch;
  ctx.fillStyle = "#4f9cf9";
  ctx.fillRect(pad - 8, pad + (1 - paddle) * ch - halfLen, 10, 2 * halfLen);
  ctx.fillStyle = "#f1b44c";
  ctx.beginPath(); ctx.arc(pad + bx * cw, pad + (1 - by) * ch, 7, 0, Math.PI * 2); ctx.fill();
  text(ctx, `Pong · hits ${b.hits} · misses ${b.misses} · step ${b.time}/${b.horizon}`, pad, 20);
}

function drawFlappy(ctx, W, H, a, b, t) {
  const top = 30, ch = H - 70, birdX = W * 0.22;
  const Y = (v) => top + (1 - v) * ch;
  ctx.fillStyle = "#2a1c1f";
  ctx.fillRect(0, Y(1), W, Y(0.975) - Y(1)); ctx.fillRect(0, Y(0.025), W, Y(0) - Y(0.025));
  const reset = b.x > a.x + 0.2;
  const k = reset ? (t < 0.5 ? 0 : 1) : t;
  const px = lerp(a.x, b.x, k), gap = k < 0.5 || !reset ? a.gap : b.gap;
  const pipeX = birdX + (px / 0.9) * (W - birdX - 40), pipeW = 34;
  const tol = a.tolerance;
  ctx.fillStyle = "#1f6f4a";
  ctx.fillRect(pipeX - pipeW / 2, Y(1), pipeW, Y(Math.min(1, gap + tol)) - Y(1));
  ctx.fillRect(pipeX - pipeW / 2, Y(Math.max(0, gap - tol)), pipeW, Y(0) - Y(Math.max(0, gap - tol)));
  ctx.strokeStyle = "#34c38f"; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(pipeX - pipeW, Y(gap)); ctx.lineTo(pipeX + pipeW, Y(gap)); ctx.stroke(); ctx.setLineDash([]);
  const by = lerp(a.y, b.y, t);
  ctx.fillStyle = "#f1b44c";
  ctx.beginPath(); ctx.arc(birdX, Y(by), 9, 0, Math.PI * 2); ctx.fill();
  text(ctx, `Flappy · passed ${b.passed}/6 · step ${b.time}/${b.horizon}${b.death ? " · " + b.death : ""}`, 10, 20);
}

function drawSnake(ctx, W, H, a, b, t) {
  const s = t < 0.5 ? a : b;
  const n = s.grid, pad = 30, cell = (Math.min(W, H) - 2 * pad - 20) / n;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    ctx.fillStyle = (r + c) % 2 ? "#121722" : "#0f141d";
    ctx.fillRect(pad + c * cell, pad + r * cell, cell, cell);
  }
  if (s.food) {
    ctx.fillStyle = "#f46a6a";
    ctx.beginPath(); ctx.arc(pad + (s.food[1] + 0.5) * cell, pad + (s.food[0] + 0.5) * cell, cell * 0.3, 0, Math.PI * 2); ctx.fill();
  }
  s.body.forEach(([r, c], i) => {
    ctx.fillStyle = i === 0 ? "#6fe0b0" : "#2f9e72";
    ctx.fillRect(pad + c * cell + 2, pad + r * cell + 2, cell - 4, cell - 4);
  });
  text(ctx, `Snake · food ${s.eaten} · length ${s.body.length} · step ${s.time}${s.death ? " · " + s.death : ""}`, pad, 20);
}

function drawBreakout(ctx, W, H, a, b, t) {
  const pad = 30, cw = W - 2 * pad, ch = H - 2 * pad - 20;
  ctx.strokeStyle = "#262c37"; ctx.strokeRect(pad, pad, cw, ch);
  const brickW = cw / 5;
  b.bricks.forEach((alive, i) => {
    ctx.fillStyle = alive ? "#b5566b" : "#1a1f29";
    ctx.fillRect(pad + i * brickW + 3, pad + 4, brickW - 6, 18);
  });
  const serve = Math.abs(b.y - a.y) > 0.3 || Math.abs(b.x - a.x) > 0.3;
  const k = serve ? (t < 0.5 ? 0 : 1) : t;
  ctx.fillStyle = "#f1b44c";
  ctx.beginPath(); ctx.arc(pad + lerp(a.x, b.x, k) * cw, pad + (1 - lerp(a.y, b.y, k)) * ch, 7, 0, Math.PI * 2); ctx.fill();
  const paddle = lerp(a.paddle, b.paddle, t);
  ctx.fillStyle = "#4f9cf9";
  ctx.fillRect(pad + (paddle - 0.14) * cw, pad + ch - 6, 0.28 * cw, 8);
  text(ctx, `Breakout · bricks left ${b.bricks.filter(Boolean).length} · misses ${b.misses}`, pad, 20);
}

// ------------------------------------------------------------------ neural
function setView(view) {
  S.view = view;
  $("view-2d").classList.toggle("active", view === "2d");
  $("view-3d").classList.toggle("active", view === "3d");
  $("neural-caption").textContent = view === "3d"
    ? `${S.network.layout3d_label} · drag to rotate, scroll to zoom`
    : "Fixed 2D population map (network layout — not anatomical position); order within each block: annotated type, then body ID.";
  S.dirtyNeural = true;
}

function drawNeural() {
  const canvas = $("neural"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);
  const net = S.network, n = net.n;
  const proj = S.projected && S.projected.length === n ? S.projected : new Array(n);
  if (S.view === "2d") {
    const size = 5;
    for (let i = 0; i < n; i++) {
      const [x, y] = net.layout2d[i];
      const px = x * W, py = y * H;
      ctx.fillStyle = PALETTE[Math.round(intensity(i) * 255)];
      ctx.fillRect(px - size / 2, py - size / 2, size, size);
      proj[i] = [px, py, 0];
    }
    ctx.font = "12px sans-serif";
    const labels = [["Visual projection · 218", "vp"], ["CX intrinsic · 529", "cx"], ["Descending · 1,293", "dn"]];
    for (const [label, key] of labels) {
      const xs = S.popIndex[key].map((i) => net.layout2d[i][0]);
      ctx.fillStyle = POP_COLORS[key];
      ctx.fillText(label, Math.min(...xs) * W - 2, 16);
    }
  } else {
    const cy = Math.cos(S.yaw), sy = Math.sin(S.yaw), cp = Math.cos(S.pitch), sp = Math.sin(S.pitch);
    const scale = Math.min(W, H) * 0.42 * S.zoom;
    const order = [];
    for (let i = 0; i < n; i++) {
      const [x, y, z] = net.layout3d[i];
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      const persp = 3 / (3 + z2);
      proj[i] = [W / 2 + x1 * scale * persp, H / 2 + y1 * scale * persp, z2];
      order.push(i);
    }
    order.sort((p, q) => proj[q][2] - proj[p][2]);
    const missing = new Set(net.missing_soma);
    for (const i of order) {
      const [px, py] = proj[i];
      const colour = PALETTE[Math.round(intensity(i) * 255)];
      if (missing.has(i)) { ctx.strokeStyle = colour; ctx.strokeRect(px - 2, py - 2, 4, 4); }
      else { ctx.fillStyle = colour; ctx.fillRect(px - 1.5, py - 1.5, 3, 3); }
    }
    ctx.fillStyle = "#98a2b3"; ctx.font = "12px sans-serif";
    ctx.fillText("soma positions (MaleCNS somaLocation) · no edges drawn", 10, 16);
  }
  S.projected = proj;
  if (S.selected !== null && proj[S.selected]) {
    const [px, py] = proj[S.selected];
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.stroke(); ctx.lineWidth = 1;
  }
}

function setupNeuralInteraction() {
  const canvas = $("neural");
  const toCanvas = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  };
  canvas.addEventListener("mousedown", (e) => { S.drag = { x: e.clientX, y: e.clientY, moved: false }; });
  window.addEventListener("mouseup", (e) => {
    if (S.drag && !S.drag.moved && e.target === canvas) pickNeuron(...toCanvas(e));
    S.drag = null;
  });
  window.addEventListener("mousemove", (e) => {
    if (!S.drag || S.view !== "3d") return;
    const dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) S.drag.moved = true;
    S.yaw += dx * 0.01; S.pitch = Math.max(-1.5, Math.min(1.5, S.pitch + dy * 0.01));
    S.drag.x = e.clientX; S.drag.y = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    if (S.view !== "3d") return;
    e.preventDefault();
    S.zoom = Math.max(0.4, Math.min(4, S.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
  }, { passive: false });
}

function pickNeuron(x, y) {
  if (!S.projected) return;
  let best = null, bestD = 64;
  S.projected.forEach((p, i) => {
    if (!p) return;
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  S.selected = best;
  S.dirtyNeural = true;
  refreshInspector();
}

function refreshInspector() {
  const box = $("inspector");
  if (S.selected === null) { box.textContent = "Click a neuron to inspect it."; return; }
  const m = S.network.meta[S.selected];
  const show = (v) => (v === null || v === undefined || v === "" ? "unavailable" : v);
  const current = S.counts && S.frame ? (S.counts[S.selected] / S.frame.ticks).toFixed(3) : "unavailable";
  const mean = S.episodeSteps ? (S.episodeSum[S.selected] / S.episodeSteps).toFixed(3) : "unavailable";
  box.innerHTML =
    `<b>body ID</b> ${m.bodyId} · <b>population</b> ${m.population}<br>` +
    `<b>type</b> ${show(m.type)} · <b>instance</b> ${show(m.instance)} · <b>transmitter</b> ${show(m.consensusNt)} · <b>soma side</b> ${show(m.somaSide)}<br>` +
    `<b>current activity</b> ${current} (spike fraction this decision) · <b>episode mean</b> ${mean}<br>` +
    `<b>in-degree</b> ${m.in_degree} · <b>out-degree</b> ${m.out_degree} <span class="meta">(${S.network.degree_note})</span>`;
}

function drawLegend() {
  const c = $("legend"), ctx = c.getContext("2d");
  for (let x = 0; x < c.width; x++) {
    ctx.fillStyle = PALETTE[Math.round((x / (c.width - 1)) * 255)];
    ctx.fillRect(x, 0, 1, c.height);
  }
}

// ------------------------------------------------------------------ heatmaps & traces
function drawHeatmaps() {
  for (const key of ["vp", "cx", "dn"]) {
    const canvas = $(`heat-${key}`), ctx = canvas.getContext("2d");
    const members = S.popIndex[key], W = canvas.width, H = canvas.height;
    const cols = { vp: 55, cx: 89, dn: 144 }[key];
    const rows = Math.ceil(members.length / cols), cw = W / cols, rh = H / rows;
    ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);
    members.forEach((neuron, k) => {
      ctx.fillStyle = PALETTE[Math.round(intensity(neuron) * 255)];
      ctx.fillRect((k % cols) * cw, Math.floor(k / cols) * rh, Math.ceil(cw), Math.ceil(rh));
    });
  }
}

function drawTraces() {
  const canvas = $("traces"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, half = H / 2, left = 40;
  ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#262c37";
  ctx.beginPath(); ctx.moveTo(left, half); ctx.lineTo(W, half); ctx.stroke();
  ctx.fillStyle = "#98a2b3"; ctx.font = "11px sans-serif";
  const maxMean = Math.max(0.05, ...S.traces.flatMap((r) => Object.values(r.means)));
  ctx.fillText(`${maxMean.toFixed(2)}`, 4, 12); ctx.fillText("0", 4, half - 4);
  ctx.fillText("1", 4, half + 12); ctx.fillText("0", 4, H - 4);
  const x = (i) => left + (i / (TRACE_LEN - 1)) * (W - left - 6);
  const line = (values, top, height, max, colour) => {
    ctx.strokeStyle = colour; ctx.beginPath();
    values.forEach((v, i) => {
      const y = top + height - (v / max) * (height - 6) - 3;
      i ? ctx.lineTo(x(i), y) : ctx.moveTo(x(i), y);
    });
    ctx.stroke();
  };
  for (const key of ["vp", "cx", "dn"]) line(S.traces.map((r) => r.means[key]), 0, half, maxMean, POP_COLORS[key]);
  const labels = (S.state && S.state.actions) || [];
  const nActions = S.traces.length ? S.traces[0].probs.length : labels.length;
  for (let a = 0; a < nActions; a++) line(S.traces.map((r) => r.probs[a]), half, half, 1, PROB_COLORS[a]);
  const legend = $("trace-legend");
  legend.innerHTML =
    ["vp", "cx", "dn"].map((k) => `<span><i style="background:${POP_COLORS[k]}"></i>mean ${{ vp: "visual projection", cx: "CX intrinsic", dn: "descending" }[k]} spike fraction</span>`).join("") +
    Array.from({ length: nActions }, (_, a) => `<span><i style="background:${PROB_COLORS[a]}"></i>P(${labels[a] || a})</span>`).join("");
}

init().catch((error) => showError(error.message));
