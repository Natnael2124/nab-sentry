/* NAB Find UI: plain JS, no build step. Talks to the local server with the per-launch token. */
"use strict";

const TOKEN = new URLSearchParams(location.hash.slice(1)).get("t") || "";
const $view = document.getElementById("view");
const $crumb = document.getElementById("crumb");
const S = { cid: null, case: null, tab: "search", poll: null, results: null, query: "", filters: { kind: "any" },
            ai: null, aiJob: null, seenEvents: 0, app: null, mode: "results", similarOf: null };

// ------------------------------------------------------------------ helpers
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;  // only for fixed, trusted markup (icons)
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k instanceof Node ? k : String(k));
  return n;
}
/** replaceChildren that skips null/false (optional parts of a view). */
function put(node, ...kids) {
  node.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  return node;
}
const ICON = {
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  folder: '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>',
  film: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M17 9h4M3 15h4M17 15h4"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  person: '<svg viewBox="0 0 24 24"><circle cx="12" cy="7" r="3.5"/><path d="M5 21a7 7 0 0 1 14 0"/></svg>',
};
const icon = (name) => el("span", { html: ICON[name], style: "display:inline-flex" });

async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || "GET",
    headers: { "X-Token": TOKEN, ...(opts.body ? { "Content-Type": "application/json" } : {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `something went wrong (${res.status})`);
  return data;
}
const media = (path) => `${path}${path.includes("?") ? "&" : "?"}t=${encodeURIComponent(TOKEN)}`;

const pad = (n) => String(n).padStart(2, "0");
const d = (ts) => new Date(ts * 1000);
const fmtClock = (ts) => { const x = d(ts); return `${pad(x.getHours())}:${pad(x.getMinutes())}:${pad(x.getSeconds())}`; };
const fmtDay = (ts) => d(ts).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
const fmtWhen = (ts) => `${fmtDay(ts)} · ${fmtClock(ts)}`;
function fmtDur(s) {
  s = Math.round(s || 0);
  if (s < 60) return `${s} s`;
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  if (h === 0) return `${m} min`;
  return m ? `${h} h ${m} min` : `${h} h`;
}
function fmtEta(s) {
  if (s === null || s === undefined || !isFinite(s)) return "";
  if (s < 50) return "under a minute";
  if (s < 3600) return `about ${Math.max(1, Math.round(s / 60))} min`;
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60 / 5) * 5;
  return m ? `about ${h} h ${m} min` : `about ${h} h`;
}
function toInput(ts) {
  const x = d(ts);
  return `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}T${pad(x.getHours())}:${pad(x.getMinutes())}`;
}
const fromInput = (v) => (v ? new Date(v).getTime() / 1000 : null);

function toast(msg, kind, action) {
  const t = el("div", { class: `toast ${kind || ""}` }, msg);
  if (action) t.append(" ", el("button", { class: "btn ghost small", style: "color:inherit;min-height:0;padding:2px 6px;text-decoration:underline", onclick: action.fn }, action.label));
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), action ? 9000 : 5000);
}
function chime() {
  try {
    const ctx = new AudioContext();
    [660, 880].forEach((f, i) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = f; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.0001, ctx.currentTime + i * 0.18);
      g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + i * 0.18 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + i * 0.18 + 0.35);
      o.start(ctx.currentTime + i * 0.18); o.stop(ctx.currentTime + i * 0.18 + 0.4);
    });
  } catch (_) { /* no audio */ }
}
const native = () => window.pywebview && window.pywebview.api;

function modal(build) {
  const dlg = document.getElementById("modal");
  dlg.replaceChildren();
  const close = () => dlg.close();
  dlg.append(build(close));
  dlg.showModal();
  return close;
}

// ------------------------------------------------------------------ status words
const STATUS = {
  new: ["Not started", ""], ready: ["Ready to start", "info"], queued: ["Waiting to start", "info"],
  processing: ["Getting ready", "info"], paused: ["Paused", "warn"], done: ["Ready to search", "good"],
  stopped: ["Stopped", "warn"], error: ["Problem", "bad"],
};
const statusChip = (s) => { const [t, k] = STATUS[s] || [s, ""]; return el("span", { class: `chip ${k}` }, t); };

// ------------------------------------------------------------------ app start
async function boot() {
  try {
    S.app = await api("/api/state");
  } catch (e) {
    $view.replaceChildren(el("div", { class: "empty" }, "NAB Find could not start: ", e.message));
    return;
  }
  updateDevice();
  document.getElementById("go-home").onclick = () => showHome();
  document.getElementById("open-settings").onclick = () => openSettings();
  showHome();
  setInterval(async () => {
    if (S.app && S.app.models_ready) return;
    S.app = await api("/api/state").catch(() => S.app);
    updateDevice();
  }, 2000);
}
function updateDevice() {
  const n = document.getElementById("device");
  if (!S.app) return;
  if (S.app.models_error) { n.textContent = "AI models failed to load"; n.title = S.app.models_error; return; }
  n.textContent = S.app.models_ready ? (S.app.device === "GPU" ? "Using graphics chip" : "Using processor") : "Starting up…";
}

// ------------------------------------------------------------------ home
async function showHome() {
  stopPoll();
  S.cid = null; S.case = null;
  $crumb.textContent = "";
  const cases = await api("/api/cases").catch((e) => { toast(e.message, "bad"); return []; });
  const list = cases.length
    ? el("div", { class: "cases" }, cases.map(caseCard))
    : el("div", { class: "empty panel" }, el("h2", {}, "No searches yet"),
        el("p", {}, "Start a new search, add the video files from your recorder, and ask what you are looking for."));
  $view.replaceChildren(
    el("section", { class: "hero panel" },
      el("div", { style: "flex:1" },
        el("h1", {}, "Find the moment in your recordings"),
        el("p", { class: "muted" }, "Add the videos from your camera recorder, say roughly when it happened, and search in your own words — “man in a red jacket”, “white van”, “someone behind the counter”. Everything stays on this computer.")),
      el("button", { class: "btn primary big", onclick: newCase }, icon("plus"), "New search")),
    el("h2", { style: "margin:4px 0 12px" }, "Your searches"),
    list);
}
function caseCard(c) {
  const span = c.first_ts ? `${fmtWhen(c.first_ts)} → ${fmtClock(c.last_ts)}` : "No videos yet";
  return el("button", { class: "case-card", onclick: () => openCase(c.id) },
    el("div", { class: "row" }, el("h3", { style: "flex:1" }, c.name), statusChip(c.status)),
    el("div", { class: "meta" }, span),
    el("div", { class: "meta" }, `${c.videos} video${c.videos === 1 ? "" : "s"} · ${fmtDur(c.footage_s)}${c.cameras.length ? " · " + c.cameras.join(", ") : ""}`));
}
async function newCase(paths) {
  try {
    const c = await api("/api/cases", { method: "POST", body: {} });
    await openCase(c.id);
    if (Array.isArray(paths) && paths.length) await addPaths(paths);
  } catch (e) { toast(e.message, "bad"); }
}

// ------------------------------------------------------------------ case router
async function openCase(cid) {
  stopPoll();
  S.cid = cid; S.results = null; S.ai = null; S.aiJob = null; S.query = ""; S.mode = "results"; S.similarOf = null;
  await refreshCase();
  S.seenEvents = (S.case.events || []).length;
  const started = S.case.counts.frames > 0 || ["processing", "queued", "paused", "done", "stopped"].includes(S.case.status);
  if (started) showCase(); else showSetup();
}
async function refreshCase() {
  S.case = await api(`/api/cases/${S.cid}`);
  $crumb.textContent = S.case.name;
  return S.case;
}

// ------------------------------------------------------------------ setup (wizard)
function showSetup() {
  stopPoll();
  const c = S.case;
  const videos = c.videos_list;
  const ok = videos.filter((v) => v.status !== "error");
  const step1 = el("section", { class: "panel" },
    el("div", { class: "step-head" }, el("span", { class: "step-num" }, "1"),
      el("div", {}, el("h2", {}, "Add your recordings"),
        el("div", { class: "muted small" }, "The video files you exported from the recorder (USB stick, phone, or a folder). They are not copied or changed."))),
    dropzone(),
    videos.length ? videoTable(videos) : null);

  const first = Math.min(...ok.map((v) => v.start_ts)), last = Math.max(...ok.map((v) => v.end_ts));
  const hasRange = !!c.range;
  const rStart = el("input", { type: "datetime-local", value: ok.length ? toInput(hasRange ? c.range[0] : first) : "" });
  const rEnd = el("input", { type: "datetime-local", value: ok.length ? toInput(hasRange ? c.range[1] : last) : "" });
  const estimate = el("div", { class: "estimate" });
  const radioAll = el("input", { type: "radio", name: "rng", checked: !hasRange });
  const radioSome = el("input", { type: "radio", name: "rng", checked: hasRange });
  const rangeBox = el("div", { class: "range-grid", style: hasRange ? "" : "display:none" },
    el("label", { class: "field" }, "From", rStart), el("label", { class: "field" }, "To", rEnd));
  const upd = () => {
    rangeBox.style.display = radioSome.checked ? "" : "none";
    let secs = ok.reduce((a, v) => a + v.duration_s, 0);
    if (radioSome.checked) {
      const a = fromInput(rStart.value), b = fromInput(rEnd.value);
      secs = ok.reduce((acc, v) => acc + Math.max(0, Math.min(v.end_ts, b || 0) - Math.max(v.start_ts, a || 0)), 0);
    }
    const sp = (S.app.settings.speeds || {});
    const q = sp.quick || 60, dd = sp.detail || 10;
    estimate.replaceChildren(
      ok.length ? el("div", {}, el("b", {}, fmtDur(secs)), " of video to go through. ",
        `First search in ${fmtEta(secs / q)}, everything in ${fmtEta(secs / q + 0.4 * secs / dd)} on this computer (a guess until it starts).`)
        : el("span", { class: "muted" }, "Add videos first."));
    startBtn.disabled = !ok.length || secs <= 0;
  };
  const startBtn = el("button", { class: "btn primary big", onclick: async () => {
    try {
      startBtn.disabled = true;
      const body = radioSome.checked ? { start_ts: fromInput(rStart.value), end_ts: fromInput(rEnd.value) } : {};
      await api(`/api/cases/${S.cid}/range`, { method: "PUT", body });
      await api(`/api/cases/${S.cid}/run/start`, { method: "POST" });
      await refreshCase();
      showCase();
    } catch (e) { toast(e.message, "bad"); startBtn.disabled = false; }
  } }, icon("search"), "Start");
  [rStart, rEnd, radioAll, radioSome].forEach((n) => n.addEventListener("change", upd));

  const step2 = el("section", { class: "panel" },
    el("div", { class: "step-head" }, el("span", { class: "step-num" }, "2"),
      el("div", {}, el("h2", {}, "When did it happen?"),
        el("div", { class: "muted small" }, "A shorter time window is faster. You can search as soon as the first part is ready."))),
    el("div", { class: "radio-row" },
      el("label", { class: "radio-card" }, radioAll, el("div", {}, el("b", {}, "Go through everything"), el("div", { class: "muted small" }, ok.length ? `${fmtWhen(first)} → ${fmtWhen(last)}` : ""))),
      el("label", { class: "radio-card" }, radioSome, el("div", {}, el("b", {}, "Only a time window"), el("div", { class: "muted small" }, "e.g. yesterday 14:00 to 17:00")))),
    el("div", { style: "margin-top:14px" }, rangeBox),
    el("div", { style: "margin-top:14px" }, estimate),
    el("div", { class: "row", style: "margin-top:16px" },
      el("input", { class: "input", value: c.name, title: "Name of this search", style: "flex:1;max-width:360px",
        onchange: async (e) => { await api(`/api/cases/${S.cid}`, { method: "PATCH", body: { name: e.target.value } }); $crumb.textContent = e.target.value; } }),
      el("span", { class: "spacer" }),
      el("button", { class: "btn ghost danger", onclick: deleteCase }, "Delete"),
      startBtn));
  put($view, el("div", { class: "steps" }, step1, step2));
  upd();
}

function dropzone() {
  return el("div", { class: "dropzone", id: "dropzone" },
    el("div", { html: ICON.film }),
    el("div", { style: "font-weight:650;margin:6px 0" }, "Drag video files or a folder here"),
    el("div", { class: "row", style: "justify-content:center;margin-top:10px" },
      el("button", { class: "btn primary", onclick: async () => { const n = native(); if (!n) return toast("File picker only works in the app window", "bad"); addPaths(await n.pick_files()); } }, icon("film"), "Choose files"),
      el("button", { class: "btn", onclick: async () => { const n = native(); if (!n) return toast("Folder picker only works in the app window", "bad"); addPaths(await n.pick_folder()); } }, icon("folder"), "Choose a folder")));
}

async function addPaths(paths) {
  if (!paths || !paths.length) return;
  const dz = document.getElementById("dropzone");
  if (dz) dz.replaceChildren(el("div", {}, el("span", { class: "spinner" }), "  Reading your videos… (long videos take a few seconds each)"));
  try {
    const r = await api(`/api/cases/${S.cid}/videos`, { method: "POST", body: { paths } });
    const bad = r.added.filter((a) => a.error);
    const good = r.added.filter((a) => !a.error);
    if (good.length) toast(`Added ${good.length} video${good.length === 1 ? "" : "s"}`);
    bad.forEach((b) => toast(`${b.path.split(/[\\/]/).pop()}: ${b.error}`, "bad"));
    if (!r.added.length) toast("No new videos found there");
  } catch (e) { toast(e.message, "bad"); }
  await refreshCase();
  showSetup();
}

function videoTable(videos) {
  const byCam = {};
  videos.forEach((v) => (byCam[v.camera] = byCam[v.camera] || []).push(v));
  const out = el("div", {});
  for (const [cam, list] of Object.entries(byCam)) {
    list.sort((a, b) => a.start_ts - b.start_ts);
    out.append(el("div", { class: "cam-group" }, cam, el("span", { class: "muted small" }, `  ${list.length} video${list.length === 1 ? "" : "s"}`)));
    const rows = list.map((v) => {
      const start = el("input", { type: "datetime-local", step: "1", value: toInputSec(v.start_ts), title: "When this video starts" });
      start.addEventListener("change", async () => {
        const ts = fromInput(start.value);
        const siblings = list.length > 1 && confirm("Move the other videos of this camera by the same amount?");
        try { await api(`/api/cases/${S.cid}/videos/${v.video_id}`, { method: "PATCH", body: { start_ts: ts, shift_siblings: siblings } }); }
        catch (e) { toast(e.message, "bad"); }
        await refreshCase(); showSetup();
      });
      const camIn = el("input", { value: v.camera, style: "width:130px", title: "Camera name" });
      camIn.addEventListener("change", async () => {
        await api(`/api/cases/${S.cid}/videos/${v.video_id}`, { method: "PATCH", body: { camera: camIn.value } }).catch((e) => toast(e.message, "bad"));
        await refreshCase(); showSetup();
      });
      return el("tr", {},
        el("td", { class: "name", title: v.path }, v.name),
        el("td", {}, camIn),
        el("td", {}, v.status === "error" ? el("span", { class: "chip bad", title: v.error || "" }, "Can't read") : start,
          v.start_estimated && v.status !== "error" ? el("div", { class: "small", style: "color:var(--warn)" }, "Guessed from the file date — please check") : null),
        el("td", {}, v.status === "error" ? el("span", { class: "small muted" }, v.error || "") : fmtDur(v.duration_s)),
        el("td", {}, el("button", { class: "btn ghost small", title: "Remove from this search", onclick: async () => {
          await api(`/api/cases/${S.cid}/videos/${v.video_id}`, { method: "DELETE" }).catch((e) => toast(e.message, "bad"));
          await refreshCase(); showSetup();
        } }, "Remove")));
    });
    out.append(el("table", { class: "vlist" },
      el("thead", {}, el("tr", {}, el("th", {}, "File"), el("th", {}, "Camera"), el("th", {}, "Starts at"), el("th", {}, "Length"), el("th", {}, ""))),
      el("tbody", {}, rows)));
  }
  return out;
}
function toInputSec(ts) { const x = d(ts); return `${toInput(ts)}:${pad(x.getSeconds())}`; }

async function deleteCase() {
  modal((close) => el("div", {},
    el("div", { class: "modal-body" }, el("h2", {}, "Delete this search?"),
      el("p", { class: "muted" }, "Your original video files are not touched. Only what NAB Find made from them (thumbnails and search data) is removed.")),
    el("div", { class: "modal-foot" },
      el("button", { class: "btn", onclick: close }, "Keep it"),
      el("button", { class: "btn primary", onclick: async () => {
        close();
        await api(`/api/cases/${S.cid}`, { method: "DELETE" }).catch((e) => toast(e.message, "bad"));
        showHome();
      } }, "Delete"))));
}

// ------------------------------------------------------------------ case view
function showCase() {
  const c = S.case;
  const head = el("div", { class: "case-head" },
    el("div", { class: "title" }, el("h1", {}, c.name),
      el("div", { class: "muted" }, c.first_ts ? `${c.cameras.join(", ")} · ${fmtWhen(c.range ? c.range[0] : c.first_ts)} → ${fmtWhen(c.range ? c.range[1] : c.last_ts)}` : "")),
    el("button", { class: "btn", id: "change-btn", onclick: async () => {
      if (["processing", "queued", "paused"].includes(S.case.status)) return toast("Stop the processing first to change videos or time");
      stopPoll(); showSetup();
    } }, "Change videos or time"));
  const progress = el("section", { class: "panel progress-card", id: "progress" });
  const tabs = el("div", { class: "tabs" },
    el("button", { class: `tab ${S.tab === "search" ? "active" : ""}`, onclick: () => { S.tab = "search"; showCase(); } }, "Search"),
    el("button", { class: `tab ${S.tab === "visits" ? "active" : ""}`, onclick: () => { S.tab = "visits"; showCase(); } }, "People & vehicles"));
  const body = el("div", { id: "tab-body" });
  put($view, head, progress, tabs, body);
  renderProgress();
  if (S.tab === "search") renderSearch(body); else renderVisits(body);
  startPoll();
}

function startPoll() {
  stopPoll();
  S.poll = setInterval(async () => {
    if (!S.cid) return stopPoll();
    try {
      const p = await api(`/api/cases/${S.cid}/progress`);
      Object.assign(S.case, p);
      const evs = p.events || [];
      for (const ev of evs.slice(S.seenEvents)) {
        if (ev.event === "quick_done") { chime(); toast("Quick search is ready — you can search all of it now. Still looking closer at the busy parts."); }
        if (ev.event === "done") { chime(); toast("All done. Everything is ready to search."); }
      }
      S.seenEvents = evs.length;
      renderProgress();
    } catch (_) { /* keep polling */ }
  }, 1000);
}
function stopPoll() { if (S.poll) clearInterval(S.poll); S.poll = null; }

function renderProgress() {
  const box = document.getElementById("progress");
  if (!box) return;
  const c = S.case, p = c.progress;
  const running = ["processing", "queued", "paused"].includes(c.status);
  let title, sub = "", pct = 100;
  if (c.status === "queued") { title = "Waiting to start…"; pct = 0; }
  else if (p && running) {
    pct = p.percent;
    if (p.pass === "quick") {
      title = `Getting your video ready — ${Math.floor(pct)}%`;
      sub = (p.quick_done_s > 0 ? `You can already search ${fmtDur(p.quick_done_s)} of ${fmtDur(p.quick_total_s)}. ` : `Starting on ${fmtDur(p.quick_total_s)} of video. `)
        + `Search ready in ${fmtEta(p.quick_eta_s)} · all done in ${fmtEta(p.full_eta_s)}.`;
    } else {
      title = `Search is ready · looking closer — ${Math.floor(pct)}%`;
      sub = `Finding each person and vehicle in the busy parts. All done in ${fmtEta(p.full_eta_s)}.`;
    }
    if (c.status === "paused") title = `Paused at ${Math.floor(pct)}%`;
  } else if (c.status === "done") {
    title = "Ready to search";
    sub = `${fmtDur(c.footage_s)} of video · ${c.counts.visits} people and vehicles found`;
  } else if (c.status === "stopped") {
    title = "Stopped"; sub = "Search works on the part that is done. Continue any time."; pct = p ? p.percent : 0;
  } else if (c.status === "error") {
    title = "Something went wrong while processing"; sub = "Try Continue. If it happens again, the log file in Settings helps us fix it.";
  } else { title = running ? "Starting…" : "Not started"; pct = 0; }

  const actions = el("div", { class: "row" });
  if (c.status === "processing") actions.append(
    el("button", { class: "btn", onclick: () => control("pause") }, "Pause"),
    el("button", { class: "btn ghost", onclick: () => control("cancel") }, "Stop"));
  if (c.status === "paused") actions.append(el("button", { class: "btn primary", onclick: () => control("resume") }, "Continue"),
    el("button", { class: "btn ghost", onclick: () => control("cancel") }, "Stop"));
  if (["stopped", "error"].includes(c.status)) actions.append(el("button", { class: "btn primary", onclick: () => control("start") }, "Continue"));

  put(box,
    el("div", {}, el("div", { class: "progress-text" }, title), sub ? el("div", { class: "muted small" }, sub) : null,
      p && running && p.speed_x && (p.quick_done_s > 0 || p.pass === "detail") ? el("div", { class: "muted small" }, `Working ${Math.round(p.speed_x)}× faster than real time on this computer`) : null),
    actions,
    el("div", { class: `bar ${c.status === "done" ? "done" : ""}` }, el("div", { style: `width:${pct}%` })),
    timeline());
}
function timeline() {
  const c = S.case;
  if (!c.coverage || !c.videos_list || !c.videos_list.length) return null;
  const a = c.range ? c.range[0] : c.first_ts, b = c.range ? c.range[1] : c.last_ts;
  if (!(b > a)) return null;
  const bar = el("div", { class: "timeline", title: "Light: searchable · dark: looked at closely" });
  for (const v of c.videos_list) {
    const cov = c.coverage[v.video_id];
    if (!cov) continue;
    for (const [cls, spans] of [["q", cov.quick], ["d", cov.detail]]) {
      for (const [s, e] of spans) {
        const L = Math.max(0, (v.start_ts + s - a) / (b - a)), R = Math.min(1, (v.start_ts + e - a) / (b - a));
        if (R > L) bar.append(el("span", { class: cls, style: `left:${L * 100}%;width:${(R - L) * 100}%` }));
      }
    }
  }
  return bar;
}
async function control(action) {
  try { await api(`/api/cases/${S.cid}/run/${action}`, { method: "POST" }); await refreshCase(); renderProgress(); }
  catch (e) { toast(e.message, "bad"); }
}

// ------------------------------------------------------------------ search tab
const EXAMPLES = ["person in a red jacket", "white van", "someone carrying a box", "person behind the counter", "motorbike", "two people talking"];

function renderSearch(body) {
  const input = el("input", { type: "search", placeholder: "Describe what you are looking for…", value: S.query, "aria-label": "What are you looking for" });
  const run = () => doSearch(input.value);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  const cams = S.case.cameras || [];
  const kindSeg = el("div", { class: "seg" }, [["any", "Anything"], ["people", "People"], ["vehicles", "Vehicles"]].map(([k, t]) =>
    el("button", { class: S.filters.kind === k ? "on" : "", onclick: () => { S.filters.kind = k; if (S.query) doSearch(S.query); else renderSearch(document.getElementById("tab-body")); } }, t)));
  const camSel = cams.length > 1 ? el("select", { onchange: (e) => { S.filters.camera = e.target.value || null; if (S.query) doSearch(S.query); } },
    el("option", { value: "" }, "All cameras"), cams.map((c) => el("option", { value: c, selected: S.filters.camera === c }, c))) : null;
  const from = el("input", { type: "datetime-local", value: S.filters.from ? toInput(S.filters.from) : "", title: "From" });
  const to = el("input", { type: "datetime-local", value: S.filters.to ? toInput(S.filters.to) : "", title: "To" });
  [from, to].forEach((n) => n.addEventListener("change", () => { S.filters.from = fromInput(from.value); S.filters.to = fromInput(to.value); if (S.query) doSearch(S.query); }));
  put(body,
    el("div", { class: "searchbar" }, input, el("button", { class: "btn primary big", onclick: run }, icon("search"), "Search")),
    el("div", { class: "examples" }, el("span", { class: "muted small", style: "align-self:center" }, "Try:"),
      EXAMPLES.map((x) => el("button", { class: "example", onclick: () => { input.value = x; run(); } }, x))),
    el("div", { class: "filters" }, kindSeg, camSel, el("span", { class: "muted small" }, "Between"), from, el("span", { class: "muted small" }, "and"), to),
    el("div", { id: "results" }));
  renderResults();
  setTimeout(() => input.focus(), 0);
}

async function doSearch(text) {
  text = (text || "").trim();
  if (!text) return;
  S.query = text; S.mode = "results"; S.similarOf = null; S.ai = null; S.aiJob = null;
  const box = document.getElementById("results");
  if (box) box.replaceChildren(el("div", { class: "empty" }, el("span", { class: "spinner" }), "  Searching…"));
  const qs = new URLSearchParams({ q: text, kind: S.filters.kind || "any" });
  if (S.filters.camera) qs.set("camera", S.filters.camera);
  if (S.filters.from) qs.set("from", S.filters.from);
  if (S.filters.to) qs.set("to", S.filters.to);
  try {
    const r = await api(`/api/cases/${S.cid}/search?${qs}`);
    S.results = r.moments;
  } catch (e) { S.results = []; toast(e.message, "bad"); }
  renderResults();
}

function momentKey(m) { return `${m.video_id}:${m.best_ts}`; }

function renderResults() {
  const box = document.getElementById("results");
  if (!box) return;
  if (S.results === null) {
    const c = S.case;
    box.replaceChildren(el("div", { class: "empty" }, c.counts.vectors ? "Type what you are looking for and press Search." : "Search becomes possible as soon as the first minutes of video are ready."));
    return;
  }
  let list = S.results;
  const checks = {};
  if (S.ai) {
    S.ai.items.forEach((it) => { if (it.check) checks[momentKey(it.moment)] = it.check; });
    list = S.ai.items.map((it) => it.moment).concat(list.filter((m) => !S.ai.items.some((it) => momentKey(it.moment) === momentKey(m))));
  }
  const top = list.length ? Math.max(...list.map((m) => m.score)) : 1;
  const low = list.length ? Math.min(...list.map((m) => m.score)) : 0;
  const head = el("div", { class: "results-head" },
    el("h2", { style: "flex:1" }, S.mode === "similar" ? "Looks like the same person or vehicle" : `${list.length} moment${list.length === 1 ? "" : "s"} for “${S.query}”`),
    S.mode === "similar" ? el("span", { class: "muted small" }, "Compares appearance only, not identity.") : null,
    S.mode === "results" && list.length ? aiButton() : null);
  const aiInfo = S.ai ? aiSummary() : null;
  const grid = el("div", { class: "grid" }, list.map((m) => resultCard(m, checks[momentKey(m)], (m.score - low) / Math.max(1e-6, top - low))));
  put(box, head, aiInfo, list.length ? grid : el("div", { class: "empty" }, "Nothing found. Try other words, or a wider time window."));
}

function resultCard(m, check, rel) {
  const badge = check ? el("span", { class: `chip badge ${check.match === "yes" ? "good" : check.match === "no" ? "bad" : "warn"}` },
    check.match === "yes" ? "AI: match" : check.match === "no" ? "AI: not it" : "AI: unsure") : null;
  return el("button", { class: `result ${check && check.match === "no" ? "no" : ""}`, onclick: () => openPlayer(m, check) },
    el("div", { class: "thumb", style: `background-image:url('${media(`/api/cases/${S.cid}/thumb/${encodeURIComponent(m.thumb)}`)}')` }, badge),
    el("div", { class: "body" },
      el("div", { class: "time" }, fmtClock(m.best_ts)),
      el("div", { class: "muted small" }, `${fmtDay(m.best_ts)} · ${m.camera}`),
      m.label ? el("div", { class: "small" }, m.label) : null,
      check && check.description ? el("div", { class: "ai" }, check.description) : null,
      el("div", { class: "rel", title: "How well it matches your words" }, el("div", { style: `width:${Math.round(15 + 85 * Math.max(0, Math.min(1, rel)))}%` }))));
}

// ------------------------------------------------------------------ AI check
function aiButton() {
  const busy = S.aiJob && !(S.ai && S.ai.done);
  return el("button", { class: "btn", disabled: busy, onclick: aiCheck }, busy ? el("span", { class: "spinner" }) : icon("spark"),
    busy ? `Checking… ${S.ai ? S.ai.progress[0] + "/" + S.ai.progress[1] : ""}` : "Check top results with AI");
}
function aiSummary() {
  const a = S.ai;
  const yes = a.items.filter((it) => it.check && it.check.match === "yes").length;
  const msg = a.done
    ? (yes ? `The AI thinks ${yes} of these show what you are looking for — they are now first.` : "The AI did not find a clear match among the top results. Look through them yourself, or try other words.")
    : "Checking the top results…";
  return el("div", { class: "panel small", style: "padding:12px 16px;margin-bottom:12px" }, msg,
    a.done && a.usd ? el("span", { class: "muted" }, `  (cost about $${a.usd.toFixed(3)})`) : null,
    a.errors && a.errors.length ? el("div", { class: "note", style: "margin-top:8px" }, a.errors.join(" · ")) : null,
    el("div", { class: "muted", style: "margin-top:4px" }, "AI can be wrong. Always watch the moment yourself."));
}
async function aiCheck() {
  const moments = S.results.slice(0, (S.app.settings.ai_moments || 20));
  let est;
  try { est = await api(`/api/cases/${S.cid}/aicheck/estimate`, { method: "POST", body: { moments } }); }
  catch (e) { return toast(e.message, "bad"); }
  if (!est.key_set) {
    return modal((close) => el("div", {},
      el("div", { class: "modal-body" }, el("h2", {}, "AI check needs a key"),
        el("p", { class: "muted" }, "The AI check uses Google Gemini. Add your Gemini API key in Settings once; it is stored encrypted on this computer.")),
      el("div", { class: "modal-foot" }, el("button", { class: "btn", onclick: close }, "Not now"),
        el("button", { class: "btn primary", onclick: () => { close(); openSettings(); } }, "Open Settings"))));
  }
  modal((close) => el("div", {},
    el("div", { class: "modal-body stack" }, el("h2", {}, "Check the top results with AI?"),
      el("p", {}, `${est.moments} moments — ${est.images} small pictures from them — will be sent to Google Gemini, which says whether each one shows “${S.query}”.`),
      el("p", { class: "muted small" }, `Whole videos never leave this computer. Estimated cost about $${Math.max(0.001, est.usd).toFixed(3)} on your key (${est.model}). Takes about half a minute.`)),
    el("div", { class: "modal-foot" }, el("button", { class: "btn", onclick: close }, "Cancel"),
      el("button", { class: "btn primary", onclick: async () => {
        close();
        try {
          const r = await api(`/api/cases/${S.cid}/aicheck`, { method: "POST", body: { q: S.query, moments } });
          S.aiJob = r.job; S.ai = null; renderResults(); pollAi(r.job);
        } catch (e) { toast(e.message, "bad"); }
      } }, "Send and check"))));
}
async function pollAi(job) {
  while (S.aiJob === job) {
    try {
      S.ai = await api(`/api/aicheck/${job}`);
      renderResults();
      if (S.ai.done) { if (S.ai.errors.length) toast(S.ai.errors[0], "bad"); break; }
    } catch (e) { toast(e.message, "bad"); break; }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

// ------------------------------------------------------------------ visits tab
async function renderVisits(body) {
  const st = S.visitOpts = S.visitOpts || { kind: "any", sort: "time", min: 0 };
  const box = el("div", { class: "grid" });
  const reload = async () => {
    const qs = new URLSearchParams({ kind: st.kind, sort: st.sort, min_s: st.min });
    box.replaceChildren(el("div", { class: "empty" }, el("span", { class: "spinner" })));
    const r = await api(`/api/cases/${S.cid}/visits?${qs}`).catch((e) => { toast(e.message, "bad"); return { visits: [] }; });
    count.textContent = `${r.visits.length} found`;
    box.replaceChildren(...(r.visits.length ? r.visits.map(visitCard) : [el("div", { class: "empty" }, "Nobody found yet. People and vehicles appear here as the video is looked at closely.")]));
  };
  const count = el("span", { class: "muted small" });
  const seg = (opts, key) => el("div", { class: "seg" }, opts.map(([k, t]) => el("button", { class: st[key] === k ? "on" : "", onclick: () => { st[key] = k; renderVisits(body); } }, t)));
  body.replaceChildren(
    el("div", { class: "filters" },
      seg([["any", "Everyone"], ["people", "People"], ["vehicles", "Vehicles"]], "kind"),
      el("span", { class: "muted small" }, "Sort:"),
      seg([["time", "By time"], ["longest", "Stayed longest"], ["fastest", "Moved fastest"]], "sort"),
      el("label", { class: "row small muted" }, "Stayed at least",
        el("select", { onchange: (e) => { st.min = +e.target.value; renderVisits(body); } },
          [[0, "any time"], [10, "10 s"], [60, "1 min"], [300, "5 min"], [600, "10 min"]].map(([v, t]) => el("option", { value: v, selected: st.min === v }, t)))),
      el("span", { class: "spacer" }), count),
    box);
  reload();
}
function visitCard(v) {
  const m = { video_id: v.video_id, camera: v.camera, best_ts: v.start_ts, offset_s: v.offset_s, visit_id: v.visit_id, thumb: v.thumb, label: v.label, score: 0 };
  return el("button", { class: "result", onclick: () => openPlayer(m) },
    el("div", { class: "thumb", style: `background-image:url('${media(`/api/cases/${S.cid}/thumb/${encodeURIComponent(v.thumb)}`)}')` }),
    el("div", { class: "body" },
      el("div", { class: "time" }, `${fmtClock(v.start_ts)} → ${fmtClock(v.end_ts)}`),
      el("div", { class: "muted small" }, `${fmtDay(v.start_ts)} · ${v.camera} · ${fmtDur(v.duration_s)}`),
      el("div", { class: "small" }, v.label)));
}

// ------------------------------------------------------------------ player
const PRE = 8;
function openPlayer(m, check) {
  const t = m.offset_s, clipStart = Math.max(0, t - PRE), videoStart = m.best_ts - m.offset_s;
  const src = media(`/api/cases/${S.cid}/preview?video=${m.video_id}&at=${t.toFixed(2)}`);
  const clock = el("div", { class: "clock" }, fmtClock(m.best_ts));
  const vid = el("video", { controls: true, autoplay: true, preload: "auto" });
  const stage = el("div", { class: "loading" }, el("div", {}, el("span", { class: "spinner" }), "  Preparing the clip…"));
  vid.addEventListener("loadedmetadata", () => { vid.currentTime = Math.min(t - clipStart, Math.max(0, vid.duration - 0.1)); stage.replaceWith(vid); });
  vid.addEventListener("timeupdate", () => { clock.textContent = fmtClock(videoStart + clipStart + vid.currentTime); });
  vid.addEventListener("error", () => { stage.replaceChildren(el("div", {}, "This clip could not be played. The video file may have been moved.")); });
  vid.src = src;
  const wrap = el("div", { class: "player-wrap", onclick: (e) => { if (e.target === wrap) close(); } });
  const close = () => { vid.pause(); vid.removeAttribute("src"); vid.load(); wrap.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  const jump = (dt) => { close(); openPlayer({ ...m, offset_s: Math.max(0, m.offset_s + dt), best_ts: m.best_ts + dt, visit_id: null, label: null }, null); };
  const exportBtn = el("button", { class: "btn", onclick: async () => {
    exportBtn.disabled = true; exportBtn.replaceChildren(el("span", { class: "spinner" }), " Saving clip…");
    try {
      const curT = clipStart + (vid.currentTime || (t - clipStart));
      const r = await api(`/api/cases/${S.cid}/export`, { method: "POST", body: { video: m.video_id, t: curT } });
      toast(`Saved ${r.path.split(/[\\/]/).pop()}`, "", native() ? { label: "Show in folder", fn: () => native().open_folder(r.path) } : null);
    } catch (e) { toast(e.message, "bad"); }
    exportBtn.disabled = false; exportBtn.replaceChildren(icon("download"), "Save a clip");
  } }, icon("download"), "Save a clip");
  wrap.append(el("div", { class: "player" }, stage,
    el("div", { class: "controls" },
      el("div", { class: "row" }, el("div", {}, clock, el("div", { class: "muted small" }, `${fmtDay(m.best_ts)} · ${m.camera}${m.label ? " · " + m.label : ""}`)),
        el("span", { class: "spacer" }),
        el("button", { class: "btn ghost", onclick: () => jump(-20) }, "◀ 20 s earlier"),
        el("button", { class: "btn ghost", onclick: () => jump(20) }, "20 s later ▶")),
      check && check.description ? el("div", { class: "ai-note" }, el("b", {}, check.match === "yes" ? "AI: looks like a match. " : check.match === "no" ? "AI: probably not it. " : "AI: not sure. "), check.description) : null,
      el("div", { class: "row" },
        m.visit_id ? el("button", { class: "btn primary", onclick: () => { close(); findAgain(m); } }, icon("person"), "Find this one again") : null,
        exportBtn,
        el("span", { class: "spacer" }),
        el("span", { class: "muted small" }, "The clip starts a few seconds before the moment. Saved clips have the camera and time written on them."),
        el("button", { class: "btn", onclick: close }, "Close")))));
  document.body.append(wrap);
}

async function findAgain(m) {
  S.tab = "search";
  if (!document.getElementById("results")) showCase();
  const box = document.getElementById("results");
  box.replaceChildren(el("div", { class: "empty" }, el("span", { class: "spinner" }), "  Looking for the same person or vehicle…"));
  try {
    const r = await api(`/api/cases/${S.cid}/similar/${m.visit_id}`);
    S.results = r.moments; S.mode = "similar"; S.similarOf = m; S.ai = null; S.aiJob = null;
  } catch (e) { toast(e.message, "bad"); }
  renderResults();
}

// ------------------------------------------------------------------ settings
async function openSettings() {
  const s = await api("/api/settings").catch((e) => { toast(e.message, "bad"); return null; });
  if (!s) return;
  const key = el("input", { class: "input", type: "password", placeholder: s.gemini_key_set ? "A key is saved — type a new one to replace it" : "Paste your Gemini API key", autocomplete: "off" });
  const model = el("select", { class: "input" }, Object.entries(s.models).map(([id, m]) => el("option", { value: id, selected: s.gemini_model === id }, m.label)));
  const n = el("select", { class: "input" }, [10, 20, 30, 50].map((x) => el("option", { value: x, selected: s.ai_moments === x }, `${x} results`)));
  const dev = el("select", { class: "input" }, [["auto", "Automatic (graphics chip if it works)"], ["CPU", "Processor only"], ["GPU", "Graphics chip"]].map(([v, t]) => el("option", { value: v, selected: s.device === v }, t)));
  modal((close) => el("div", {},
    el("div", { class: "modal-body stack" }, el("h2", {}, "Settings"),
      el("label", { class: "field" }, "AI check key (Google Gemini)", key,
        el("span", { class: "small" }, "Optional. Stored encrypted for your Windows account. Only small pictures of the results you choose to check are sent, never whole videos.")),
      s.gemini_key_set ? el("button", { class: "btn ghost danger small", style: "width:fit-content", onclick: async () => { await api("/api/settings", { method: "PUT", body: { gemini_key: "" } }); toast("Key removed"); close(); } }, "Remove saved key") : null,
      el("div", { class: "range-grid" },
        el("label", { class: "field" }, "AI model", model),
        el("label", { class: "field" }, "How many results to check", n)),
      el("label", { class: "field" }, "Where the AI models run (takes effect after restart)", dev),
      el("details", {}, el("summary", {}, "More"),
        el("p", { class: "small muted" }, `Your searches are kept in ${s.data_dir}. Logs are in its “logs” folder.`),
        el("p", { class: "small muted" }, `NAB Find ${S.app.version}.`))),
    el("div", { class: "modal-foot" }, el("button", { class: "btn", onclick: close }, "Cancel"),
      el("button", { class: "btn primary", onclick: async () => {
        const body = { gemini_model: model.value, ai_moments: +n.value, device: dev.value };
        if (key.value.trim()) body.gemini_key = key.value.trim();
        try { const r = await api("/api/settings", { method: "PUT", body }); S.app.settings = r; toast("Settings saved"); close(); }
        catch (e) { toast(e.message, "bad"); }
      } }, "Save"))));
}

// ------------------------------------------------------------------ drag and drop (paths come from the app)
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) document.body.classList.remove("dragging"); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => { e.preventDefault(); dragDepth = 0; document.body.classList.remove("dragging"); });
window.NF = {
  onDrop(paths) {
    document.body.classList.remove("dragging");
    if (!S.cid) return newCase(paths);
    if (["processing", "queued", "paused"].includes(S.case.status)) return toast("Stop the processing first to add videos");
    showSetup();
    addPaths(paths);
  },
};

boot();
