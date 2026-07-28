"use strict";

const REFRESH_MS = 15000;
let timer = null;
let currentDay = null;
const seenSymbols = new Set();   // symbols already alerted (per day) — avoid re-beeping
let firstRender = true;          // skip alarm on the first population of a day
let alarmReady = false;          // becomes true after first user interaction
let audioCtx = null;

const $ = (id) => document.getElementById(id);

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtFloat(v) {
  if (v === null || v === undefined) return "—";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return String(v);
}

function fmtTime(v) {
  if (!v) return "—";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function scoreColor(score) {
  if (score === null || score === undefined) return "#30363d";
  // 0 -> red, 50 -> amber, 100 -> green
  const hue = Math.max(0, Math.min(120, (score / 100) * 120));
  return `hsl(${hue}, 70%, 55%)`;
}

function setStatus(text) {
  $("status").textContent = text;
}

// Loud 2-second beeping alarm using the Web Audio API (no audio file needed).
// Browsers block audio until the user interacts, so we unlock it on interaction.
function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  // resume() is async; return the promise so callers can await it.
  const p = audioCtx.state === "suspended" ? audioCtx.resume() : Promise.resolve();
  alarmReady = true;
  return p;
}

async function soundAlarm(durationMs = 2000) {
  if (!alarmReady) {
    console.warn("[alarm] audio not unlocked yet — click the page once.");
    return;
  }
  // Make sure the context is actually running before scheduling beeps,
  // otherwise currentTime is frozen and nothing plays.
  await ensureAudio();
  if (audioCtx.state !== "running") {
    console.warn("[alarm] AudioContext not running:", audioCtx.state);
    return;
  }
  const beepLen = 0.25;    // seconds per beep
  const gap = 0.15;        // seconds between beeps
  const step = beepLen + gap;
  const count = Math.floor(durationMs / 1000 / step);
  const now = audioCtx.currentTime + 0.05;  // small lookahead
  for (let i = 0; i < count; i++) {
    const t = now + i * step;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;            // loud, attention-grabbing tone
    gain.gain.setValueAtTime(0.9, t);     // near-max volume
    gain.gain.setValueAtTime(0.0001, t + beepLen);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + beepLen);
  }
}

async function loadDays() {
  const res = await fetch("/api/days");
  const days = await res.json();
  const sel = $("day");
  sel.innerHTML = "";
  if (days.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "(no data)";
    sel.appendChild(opt);
    return null;
  }
  for (const d of days) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    sel.appendChild(opt);
  }
  return days[0];
}

async function loadList() {
  const day = $("day").value;
  // Reset the alarm baseline when switching days (don't beep for old rows).
  if (day !== currentDay) {
    seenSymbols.clear();
    firstRender = true;
    currentDay = day;
  }
  const qs = day && day !== "(no data)" ? `?date=${encodeURIComponent(day)}` : "";
  let data;
  try {
    const res = await fetch(`/api/lists${qs}`);
    data = await res.json();
  } catch (err) {
    setStatus("fetch error");
    return;
  }

  let sections;
  try {
    sections = buildSections(data);
  } catch (err) {
    console.error("[render] failed to build sections", err);
    setStatus("render error");
    return;
  }

  // Collect all symbols across every section to drive the alarm check.
  const allRows = [
    ...sections.top_gainers,
    ...sections.small_cap,
    ...sections.low_float,
  ];
  _checkAlarm(allRows);

  try {
    renderSection("top_gainers", sections.top_gainers);
    renderSection("small_cap", sections.small_cap);
    renderSection("low_float", sections.low_float);
  } catch (err) {
    console.error("[render] failed to paint sections", err);
    setStatus("render error");
    return;
  }
  setStatus(`updated ${new Date().toLocaleTimeString()} · ${data.date || "—"}`);
}

function buildSections(data) {
  const alerts = Array.isArray(data.alerts) ? data.alerts : [];
  if (!alerts.length) {
    return {
      top_gainers: data["top_gainers"] || [],
      small_cap: data["small_cap"] || [],
      low_float: data["low_float"] || [],
    };
  }

  const sectionRows = {
    top_gainers: [],
    small_cap: [],
    low_float: [],
  };
  const seen = {
    top_gainers: new Set(),
    small_cap: new Set(),
    low_float: new Set(),
  };

  for (const row of alerts) {
    const cats = Array.isArray(row.categories) ? row.categories : [];
    for (const cat of ["top_gainers", "small_cap", "low_float"]) {
      if (!cats.includes(cat)) continue;
      const key = `${row.symbol || ""}|${row.triggered_at || ""}`;
      if (seen[cat].has(key)) continue;
      seen[cat].add(key);
      sectionRows[cat].push(row);
    }
  }

  // Fall back per-section if alerts exist but a specific category is missing.
  return {
    top_gainers: sectionRows.top_gainers.length ? sectionRows.top_gainers : (data["top_gainers"] || []),
    small_cap: sectionRows.small_cap.length ? sectionRows.small_cap : (data["small_cap"] || []),
    low_float: sectionRows.low_float.length ? sectionRows.low_float : (data["low_float"] || []),
  };
}

function _checkAlarm(rows) {
  let newHit = false;
  for (const r of rows) {
    if (r.symbol && !seenSymbols.has(r.symbol)) {
      seenSymbols.add(r.symbol);
      newHit = true;
    }
  }
  if (newHit && !firstRender) soundAlarm(2000);
  firstRender = false;
}

function renderSection(category, rows) {
  rows = rows.slice().sort((a, b) => (b.momentum_score || 0) - (a.momentum_score || 0));
  const tbody = $(`rows-${category}`);
  const empty = $(`empty-${category}`);
  tbody.innerHTML = "";
  empty.hidden = rows.length > 0;

  for (const r of rows) {
    const tr = document.createElement("tr");
    const chgClass = (r.change_pct || 0) >= 0 ? "pos" : "neg";
    const tags = (r.categories || [])
      .map((c) => `<span class="tag">${c}</span>`)
      .join("");
    const scoreVal = r.momentum_score;
    const scoreCell =
      scoreVal === null || scoreVal === undefined
        ? "—"
        : `<span class="score-pill" style="background:${scoreColor(scoreVal)}">${Math.round(scoreVal)}</span>`;

    tr.innerHTML = `
      <td class="time">${fmtTime(r.triggered_at)}</td>
      <td class="num">${scoreCell}</td>
      <td class="symbol">${r.symbol || "—"}</td>
      <td>${r.company_name || ""}</td>
      <td class="num">$${fmtNum(r.price)}</td>
      <td class="num ${chgClass}">${fmtNum(r.change_pct, 1)}%</td>
      <td class="num">${fmtNum(r.relative_volume, 1)}x</td>
      <td class="num">${fmtFloat(r.float_shares)}</td>
      <td>${r.headline || ""}</td>
      <td>${tags}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function runScan() {
  setStatus("scanning…");
  try {
    const res = await fetch("/api/scan", { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      setStatus(`scan failed: ${err.detail || res.status}`);
      return;
    }
    const out = await res.json();
    setStatus(`scan done · ${out.hits} hit(s)`);
    await loadDays();
    await loadList();
  } catch (err) {
    setStatus("scan error");
  }
}

function scheduleAuto() {
  if (timer) clearInterval(timer);
  if ($("auto").checked) timer = setInterval(loadList, REFRESH_MS);
}

async function init() {
  // Unlock audio on any user gesture so the alarm can sound later. We keep the
  // listeners active (no { once }) so a re-suspended context gets resumed again.
  const unlock = () => ensureAudio();
  document.body.addEventListener("click", unlock);
  document.body.addEventListener("keydown", unlock);
  // Browsers may suspend the context when the tab is hidden; resume on return.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
  });

  $("refresh").addEventListener("click", loadList);
  $("scan").addEventListener("click", runScan);
  $("test-alarm").addEventListener("click", () => soundAlarm(2000));
  $("day").addEventListener("change", loadList);
  $("auto").addEventListener("change", scheduleAuto);

  try {
    currentDay = await loadDays();
    await loadList();
    scheduleAuto();
  } catch (err) {
    console.error("[init] failed", err);
    setStatus("init error");
  }
}

init();
