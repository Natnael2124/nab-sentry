// NAB Sentry Operator Console (Requirement 12).
// Plain ES2020 module, no dependencies, CSP-safe: no eval, no inline handlers, and no
// innerHTML — every piece of server data is written via textContent / attributes.

export const REQUEST_TIMEOUT_MS = 30000;
export const MAX_QUERY_LEN = 256;

export const MSG_NO_RESULTS = "No matching incidents found";
export const MSG_LOADING = "Searching…";
export const MSG_QUERY_INVALID = "Enter a search query of 1 to 256 characters.";
export const MSG_CAMERAS_FAILED = "The camera list could not be loaded. You can still search all cameras.";
export const MSG_TIMEOUT = "The server did not respond in time. Please try again.";
export const MSG_UNREACHABLE = "The server could not be reached. Please try again.";
export const MSG_VIDEO_FAILED = "The video could not be loaded.";

/** Raised by fetchWithTimeout when the request is aborted after the timeout. */
export class TimeoutError extends Error {
  constructor(message = MSG_TIMEOUT) {
    super(message);
    this.name = "TimeoutError";
  }
}

/** fetch() that aborts after `ms` milliseconds (12.6, 12.14). */
export async function fetchWithTimeout(url, ms = REQUEST_TIMEOUT_MS, init = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, ms);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (timedOut) throw new TimeoutError();
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** Number of Unicode code points (matches the server's Python len()). */
function codePointLength(s) {
  let n = 0;
  for (const _ of s) n += 1; // eslint-disable-line no-unused-vars
  return n;
}

/**
 * Client-side query validation (12.13). Returns { ok: true, q } with the trimmed query, or
 * { ok: false, message } when the query is empty, whitespace-only, or longer than 256.
 */
export function validateQuery(raw) {
  const text = typeof raw === "string" ? raw : "";
  const q = text.trim();
  if (q.length === 0 || codePointLength(text) > MAX_QUERY_LEN) {
    return { ok: false, message: MSG_QUERY_INVALID };
  }
  return { ok: true, q };
}

/**
 * Build the /api/search URL (12.3): trimmed q plus only the filters that are set.
 * `values` is { q, camera, start, end, cls } (strings). datetime-local values carry no
 * offset and are sent as-is; the server interprets naive times in its local zone.
 */
export function buildSearchUrl(values, base = "/api/search") {
  const params = new URLSearchParams();
  params.set("q", String(values.q ?? "").trim());
  for (const key of ["camera", "start", "end", "cls"]) {
    const v = values[key];
    if (typeof v === "string" && v !== "") params.set(key, v);
  }
  return `${base}?${params.toString()}`;
}

/** "2024-05-01T13:04:05.250+10:00" -> "2024-05-01 13:04:05" (camera-local wall clock, 12.4). */
export function formatTs(iso) {
  const s = String(iso ?? "");
  return `${s.slice(0, 10)} ${s.slice(11, 19)}`;
}

/** Event_Score rounded to 3 decimals (12.4). */
export function formatScore(score) {
  return Number(score).toFixed(3);
}

/** Timeline span as percentages of the clip duration, clamped to [0, 100] (12.8). */
export function spanPercent(startS, endS, duration) {
  if (!Number.isFinite(duration) || duration <= 0) return null;
  const clamp = (x) => Math.min(100, Math.max(0, x));
  const left = clamp((startS / duration) * 100);
  const right = clamp((endS / duration) * 100);
  return { left, width: Math.max(0, right - left) };
}

/** Enabled state of Previous / Replay / Next for an active index (12.10). */
export function navState(activeIndex, count) {
  const loaded = activeIndex >= 0 && activeIndex < count;
  return {
    prev: loaded && activeIndex > 0,
    replay: loaded,
    next: loaded && activeIndex < count - 1,
  };
}

/** Extract the API error message from a 4xx/5xx JSON body, or null. */
export async function readApiError(response) {
  try {
    const body = await response.json();
    const msg = body && body.error && body.error.message;
    return typeof msg === "string" && msg !== "" ? msg : null;
  } catch {
    return null;
  }
}

function requestErrorMessage(err) {
  return err instanceof TimeoutError ? MSG_TIMEOUT : MSG_UNREACHABLE;
}

// ---------------------------------------------------------------------------------------
// DOM wiring
// ---------------------------------------------------------------------------------------

function init() {
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("search-form"),
    q: $("q"),
    qError: $("q-error"),
    camera: $("camera"),
    cameraError: $("camera-error"),
    start: $("start"),
    end: $("end"),
    cls: $("cls"),
    searchBtn: $("search-btn"),
    status: $("status"),
    results: $("results"),
    video: $("video"),
    span: $("timeline-span"),
    replay: $("replay"),
    prev: $("prev"),
    next: $("next"),
    playerError: $("player-error"),
  };

  const state = {
    events: [],
    cards: [],
    active: -1,
    loadToken: 0, // guards against stale loadedmetadata/error after switching events
    searching: false,
  };

  // ---- cameras (12.2, 12.14) ----
  async function loadCameras() {
    try {
      const resp = await fetchWithTimeout("/api/cameras");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const cams = await resp.json();
      if (!Array.isArray(cams)) throw new Error("bad camera list");
      const frag = document.createDocumentFragment();
      for (const cam of cams) {
        if (!cam || typeof cam.camera_id !== "string") continue;
        const opt = document.createElement("option");
        opt.value = cam.camera_id;
        opt.textContent = typeof cam.label === "string" && cam.label ? cam.label : cam.camera_id;
        frag.appendChild(opt);
      }
      el.camera.appendChild(frag);
      el.cameraError.textContent = "";
    } catch {
      // Only "All cameras" remains; search stays available.
      while (el.camera.options.length > 1) el.camera.remove(1);
      el.camera.value = "";
      el.cameraError.textContent = MSG_CAMERAS_FAILED;
    }
  }

  // ---- player ----
  function updateNav() {
    const s = navState(state.active, state.events.length);
    el.prev.disabled = !s.prev;
    el.replay.disabled = !s.replay;
    el.next.disabled = !s.next;
  }

  function hideSpan() {
    el.span.hidden = true;
    el.span.style.removeProperty("--span-left");
    el.span.style.removeProperty("--span-width");
  }

  function showSpan(ev, duration) {
    const p = spanPercent(Number(ev.start_offset_s), Number(ev.end_offset_s), duration);
    if (!p) {
      hideSpan();
      return;
    }
    el.span.style.setProperty("--span-left", `${p.left}%`);
    el.span.style.setProperty("--span-width", `${p.width}%`);
    el.span.hidden = false;
  }

  function markActive(index) {
    state.cards.forEach((card, i) => {
      if (i === index) card.setAttribute("aria-current", "true");
      else card.removeAttribute("aria-current");
    });
  }

  function resetPlayer() {
    state.loadToken += 1;
    state.active = -1;
    el.video.pause();
    el.video.removeAttribute("src");
    el.video.load();
    el.playerError.textContent = "";
    hideSpan();
    updateNav();
  }

  function activate(index) {
    if (index < 0 || index >= state.events.length) return;
    const ev = state.events[index];
    const token = ++state.loadToken;
    state.active = index;
    markActive(index);
    updateNav();
    el.playerError.textContent = "";
    hideSpan();

    const onMeta = () => {
      if (token !== state.loadToken) return;
      el.video.currentTime = Number(ev.start_offset_s) || 0;
      showSpan(ev, el.video.duration);
    };
    el.video.addEventListener("loadedmetadata", onMeta, { once: true });
    el.video.src = ev.video_url;
    el.video.load();
  }

  // Video load failure (12.15): message in player area; list and active card untouched.
  el.video.addEventListener("error", () => {
    if (state.active < 0 || !el.video.getAttribute("src")) return;
    hideSpan();
    el.playerError.textContent = MSG_VIDEO_FAILED;
  });

  el.replay.addEventListener("click", () => {
    if (state.active < 0) return;
    el.video.currentTime = Number(state.events[state.active].start_offset_s) || 0;
  });
  el.prev.addEventListener("click", () => {
    if (state.active > 0) activate(state.active - 1);
  });
  el.next.addEventListener("click", () => {
    if (state.active >= 0 && state.active < state.events.length - 1) activate(state.active + 1);
  });

  // ---- results (12.4, 12.5, 12.12) ----
  function buildCard(ev, index) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "card";

    const label = String(ev.camera_label ?? ev.camera_id ?? "");
    const start = formatTs(ev.start_time);
    const end = formatTs(ev.end_time);

    const img = document.createElement("img");
    img.className = "card-thumb";
    img.src = ev.thumbnail_url;
    img.alt = `${label}, ${start}`;
    img.loading = "lazy";

    const cam = document.createElement("span");
    cam.className = "card-camera";
    cam.textContent = label;

    const time = document.createElement("span");
    time.className = "card-time";
    time.textContent = `${start} – ${end}`;

    const score = document.createElement("span");
    score.className = "card-score";
    score.textContent = `Score ${formatScore(ev.score)}`;

    btn.append(img, cam, time, score);
    btn.addEventListener("click", () => activate(index));
    li.appendChild(btn);
    return { li, btn };
  }

  function renderResults(events) {
    resetPlayer();
    state.events = events;
    state.cards = [];
    const frag = document.createDocumentFragment();
    events.forEach((ev, i) => {
      const { li, btn } = buildCard(ev, i);
      state.cards.push(btn);
      frag.appendChild(li);
    });
    el.results.replaceChildren(frag);
    updateNav();
    el.status.textContent = events.length === 0 ? MSG_NO_RESULTS : "";
  }

  // ---- search (12.3, 12.6, 12.13) ----
  async function runSearch() {
    if (state.searching) return;
    const check = validateQuery(el.q.value);
    if (!check.ok) {
      el.qError.textContent = check.message;
      return; // inputs left untouched, no request sent
    }
    el.qError.textContent = "";

    const url = buildSearchUrl({
      q: el.q.value,
      camera: el.camera.value,
      start: el.start.value,
      end: el.end.value,
      cls: el.cls.value,
    });

    state.searching = true;
    el.searchBtn.disabled = true;
    el.form.setAttribute("aria-busy", "true");
    el.status.textContent = MSG_LOADING;
    try {
      const resp = await fetchWithTimeout(url);
      if (!resp.ok) {
        el.status.textContent = (await readApiError(resp)) ?? MSG_UNREACHABLE;
        return;
      }
      let events;
      try {
        events = await resp.json();
      } catch {
        events = null;
      }
      if (!Array.isArray(events)) {
        el.status.textContent = MSG_UNREACHABLE;
        return;
      }
      renderResults(events);
    } catch (err) {
      el.status.textContent = requestErrorMessage(err);
    } finally {
      state.searching = false;
      el.searchBtn.disabled = false;
      el.form.removeAttribute("aria-busy");
    }
  }

  el.form.addEventListener("submit", (e) => {
    e.preventDefault();
    runSearch();
  });

  updateNav();
  loadCameras();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
}
