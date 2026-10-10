/* Jadal QA PoC dashboard — plain JS, no CDN, polls every 3 s. */
"use strict";

const $ = (id) => document.getElementById(id);
const PER_PAGE = 24;
const POLL_MS = 3000;

const state = {
  route: "overview",
  summary: null,
  lastN: -1,
  page: 1,
  pitch: false,
  facetsLoaded: false,
};

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

/* ── routing ─────────────────────────────────────────────────────────── */
const ROUTES = ["overview", "gallery", "live", "about"];
function route() {
  const h = (location.hash || "#/overview").replace("#/", "");
  state.route = ROUTES.includes(h) ? h : "overview";
  for (const name of ROUTES) {
    const tab = $("tab-" + name), view = $("view-" + name);
    const on = name === state.route;
    tab.setAttribute("aria-selected", on ? "true" : "false");
    if (on) tab.removeAttribute("tabindex"); else tab.setAttribute("tabindex", "-1");
    view.hidden = !on;
  }
  if (state.route === "gallery") loadGallery();
  if (state.route === "live") loadLive();
}
window.addEventListener("hashchange", route);
for (const name of ROUTES) {
  $("tab-" + name).addEventListener("click", () => { location.hash = "#/" + name; });
}

/* ── summary + overview ──────────────────────────────────────────────── */
async function loadSummary() {
  let s;
  try {
    s = await getJSON("/api/summary");
  } catch (e) {
    $("foot-source").textContent = "No data yet — the run worker has not written summary.json.";
    return;
  }
  state.summary = s;
  $("partial-pill").hidden = !s.partial;
  $("demo-pill").hidden = s.source !== "demo-fixture";
  $("run-meta").textContent = `${s.n_windows} windows · ${s.hours} h video`;
  $("foot-source").textContent =
    `Source: ${s.source === "demo-fixture" ? "demo fixture (tests/fixtures/poc)" : "poc_data (live run)"}` +
    ` · generated ${s.generated_utc}${s.partial ? " · partial run, more windows arriving" : ""}`;
  $("overview-sub").textContent =
    `${s.n_windows} windows · ${s.hours} h of video${s.partial ? " · partial run" : ""}`;
  renderTiles(s);
  renderReasonChart(s);
  renderFactoryChart(s);
  renderStageChart(s);
  if (s.n_windows !== state.lastN) {
    const grew = state.lastN !== -1;
    state.lastN = s.n_windows;
    if (grew && state.route === "gallery" && !state.pitch) loadGallery();
  }
}

function tile(num, label) {
  return `<div class="tile"><div class="t-num">${esc(num)}</div><div class="t-label">${esc(label)}</div></div>`;
}

function renderTiles(s) {
  const n = s.n_windows || 0;
  const bv = s.by_verdict || {};
  const pct = (v) => n ? `${(100 * (v || 0) / n).toFixed(1)}%` : "—";
  const tp = s.throughput || {};
  $("tiles").innerHTML =
    tile(n, "windows (10 s each)") +
    tile(s.hours, "hours of video") +
    tile(`${bv.accept || 0} · ${pct(bv.accept)}`, "accept") +
    tile(`${bv.review || 0} · ${pct(bv.review)}`, "review (S2, handed on)") +
    tile(`${bv.reject || 0} · ${pct(bv.reject)}`, "reject") +
    tile(tp.hours_of_video_per_wall_hour != null ? tp.hours_of_video_per_wall_hour + "×" : "—",
         "hours of video per wall-hour") +
    tile(tp.gpu || "—", "measured on");
}

function hbarChart(rows, maxV) {
  // rows: [label, value, cls]
  const W = 560, rowH = 30, labelW = 150, valW = 60;
  const H = rows.length * rowH + 8;
  let out = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
  rows.forEach(([label, v, cls], i) => {
    const y = i * rowH + 4;
    const bw = maxV ? ((W - labelW - valW) * v / maxV) : 0;
    out += `<text x="0" y="${y + 19}" font-size="12">${esc(label)}</text>` +
      `<rect class="${cls || "bar"}" x="${labelW}" y="${y + 4}" width="${bw.toFixed(1)}" height="18" rx="3"/>` +
      `<text x="${W - valW}" y="${y + 19}" font-size="12">${esc(v)}</text>`;
  });
  return out + "</svg>";
}

function renderReasonChart(s) {
  const by = s.by_reason || {};
  const keys = Object.keys(by).sort((a, b) => by[b] - by[a]);
  $("chart-reason").innerHTML = keys.length
    ? hbarChart(keys.map((k) => [k, by[k]]), Math.max(...keys.map((k) => by[k])))
    : `<p class="micro">No rejects yet.</p>`;
}

function renderFactoryChart(s) {
  const bf = s.by_factory || {};
  const facs = Object.keys(bf).sort();
  if (!facs.length) { $("chart-factory").innerHTML = `<p class="micro">No factories yet.</p>`; return; }
  const W = 560, rowH = 34, labelW = 120, valW = 44;
  const H = facs.length * rowH + 8;
  let out = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Per-factory stacked verdict shares">`;
  facs.forEach((f, i) => {
    const d = bf[f], n = d.n || 1, y = i * rowH + 4;
    const bw = W - labelW - valW;
    const wa = bw * (d.accept || 0) / n, wr = bw * (d.review || 0) / n, wj = bw * (d.reject || 0) / n;
    out += `<text x="0" y="${y + 20}" font-size="12">${esc(f)}</text>` +
      `<rect class="seg-a" x="${labelW}" y="${y + 4}" width="${wa.toFixed(1)}" height="18"/>` +
      `<rect class="seg-r" x="${labelW + wa}" y="${y + 4}" width="${wr.toFixed(1)}" height="18"/>` +
      `<rect class="seg-j" x="${labelW + wa + wr}" y="${y + 4}" width="${wj.toFixed(1)}" height="18" rx="0"/>` +
      `<text x="${W - valW}" y="${y + 20}" font-size="12">n=${d.n}</text>`;
  });
  out += "</svg>";
  out += `<p class="micro"><span class="chip chip-accept">accept</span><span class="chip chip-review">review</span><span class="chip chip-reject">reject</span></p>`;
  $("chart-factory").innerHTML = out;
}

function renderStageChart(s) {
  const live = s.live || {}, bs = live.by_stage || { S0: 0, S1: 0, S2: 0 };
  const total = (bs.S0 || 0) + (bs.S1 || 0) + (bs.S2 || 0);
  const share = total ? `S0 rejected ${(100 * (live.early_reject_share || 0)).toFixed(1)}% of all rejects without full-window scoring.` : "";
  $("chart-stage").innerHTML =
    hbarChart([["S0 · early rules", bs.S0 || 0], ["S1 · full window", bs.S1 || 0], ["S2 · handed on", bs.S2 || 0]],
              Math.max(1, bs.S0 || 0, bs.S1 || 0, bs.S2 || 0)) +
    (share ? `<p class="micro">${esc(share)}</p>` : "");
}

/* ── gallery ─────────────────────────────────────────────────────────── */
function fillSelect(el, values, seen) {
  for (const v of values) {
    if (seen.has(v)) continue;
    seen.add(v);
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    el.appendChild(o);
  }
}

function galleryQuery(extra) {
  const [sort, order] = $("f-sort").value.split("-");
  const p = new URLSearchParams({
    verdict: $("f-verdict").value, reason: $("f-reason").value,
    factory: $("f-factory").value, worker: $("f-worker").value,
    stage: $("f-stage").value, q: $("f-q").value.trim(),
    sort, order: order || "desc", page: state.page, per_page: PER_PAGE,
    ...(extra || {}),
  });
  return p.toString();
}

/* Every window on this dashboard is Egocentric-10K, where the cascade is not
   validated (human-labelled check, 60 windows). Verdict chips carry this tag. */
const OOD_TITLE = "Human-labelled check on 60 Egocentric-10K windows (51 good, 9 bad): " +
  "score AUROC 0.529 (chance), 21 of 51 good windows falsely rejected (41%), " +
  "5 of 9 bad rejected, balanced accuracy 0.51. Verdict rates here are descriptive, not quality.";
function oodTag() {
  return `<span class="chip chip-ood" title="${esc(OOD_TITLE)}">unvalidated on this data type</span>`;
}

function evidenceTitle(reasons) {
  return reasons.map((r) => {
    const e = r.evidence || {};
    return `${r.code}: ${e.feature} ${e.value} ${e.op} ${e.threshold}`;
  }).join("\n");
}

function cardHTML(w) {
  const thumbs = (w.thumbs || []).slice(0, 3).map((t) =>
    `<img src="/${esc(t)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">`).join("");
  const vchip = `<span class="chip chip-${w.verdict === "review" ? "review" : w.verdict}">${esc(w.verdict)} · ${esc(w.stage || "")}</span> ${oodTag()}`;
  const rchips = (w.reasons || []).map((r) =>
    `<span class="chip chip-reason" title="${esc(evidenceTitle([r]))}">${esc(r.code)}</span>`).join("");
  const score = w.score == null ? "score —" : `score ${Number(w.score).toFixed(2)}`;
  return `<button class="gcard" data-wid="${esc(w.window_id)}">` +
    `<span class="strip">${thumbs}</span>` +
    `<span class="gbody"><span class="gid">${esc(w.window_id)}</span><br>${vchip}<br>${rchips}` +
    `<span class="micro">${esc(w.factory || "")} · ${esc(score)}</span></span></button>`;
}

function wireCards() {
  document.querySelectorAll(".gcard").forEach((c) =>
    c.addEventListener("click", () => openDrawer(c.dataset.wid)));
}

async function loadGallery() {
  state.pitch = false;
  let d;
  try {
    d = await getJSON("/api/windows?" + galleryQuery());
  } catch (e) {
    $("gallery").innerHTML = `<div class="empty">Could not load windows (${esc(e.message)}).</div>`;
    return;
  }
  if (d.facets && !state.facetsLoaded) {
    state.facetsLoaded = true;
    fillSelect($("f-reason"), d.facets.reasons, new Set([""]));
    fillSelect($("f-factory"), d.facets.factories, new Set([""]));
    fillSelect($("f-worker"), d.facets.workers, new Set([""]));
  } else if (d.facets) {
    fillSelect($("f-reason"), d.facets.reasons, new Set([...$("f-reason").options].map((o) => o.value)));
    fillSelect($("f-factory"), d.facets.factories, new Set([...$("f-factory").options].map((o) => o.value)));
    fillSelect($("f-worker"), d.facets.workers, new Set([...$("f-worker").options].map((o) => o.value)));
  }
  const pages = Math.max(1, Math.ceil(d.total / d.per_page));
  $("gallery-meta").textContent = `${d.total} window${d.total === 1 ? "" : "s"} · page ${d.page}/${pages}`;
  $("gallery-sub").textContent = `${d.total} windows match — hover a reason chip for its evidence.`;
  $("gallery").innerHTML = d.windows.length
    ? d.windows.map(cardHTML).join("")
    : `<div class="empty">No windows match these filters.</div>`;
  wireCards();
  $("page-info").textContent = `page ${d.page} of ${pages}`;
  $("prev-page").disabled = d.page <= 1;
  $("next-page").disabled = d.page >= pages;
  // Deep links (?verdict=reject, ?wid=<id>): preset filter once, open drawer.
  const q = new URLSearchParams(location.search);
  if (!state.deepLinked) {
    state.deepLinked = true;
    const v = q.get("verdict");
    if (v && [...$("f-verdict").options].some((o) => o.value === v)) {
      $("f-verdict").value = v;
      state.page = 1;
      await loadGallery();
      return;
    }
  }
  const wid = q.get("wid");
  if (wid && !$("drawer")?.open && $("drawer").hidden) openDrawer(wid);
}

/* pitch mode: 12 accepted + 12 rejected per reason */
async function pitchMode() {
  state.pitch = true;
  $("gallery").innerHTML = `<div class="empty">Collecting pitch examples…</div>`;
  try {
    const acc = await getJSON("/api/windows?verdict=accept&per_page=12&sort=score&order=desc");
    const reasons = (state.summary && state.summary.by_reason)
      ? Object.keys(state.summary.by_reason) : [];
    let html = `<h2 class="view-title" style="font-size:1.1rem">Accepted (top 12 by score)</h2>`;
    html += `<div class="grid" style="margin:var(--s3) 0 var(--s5)">` + acc.windows.map(cardHTML).join("") + `</div>`;
    for (const r of reasons) {
      const d = await getJSON(`/api/windows?verdict=reject&reason=${encodeURIComponent(r)}&per_page=12`);
      html += `<h2 class="view-title" style="font-size:1.1rem">Rejected · ${esc(r)} (${d.total})</h2>`;
      html += `<div class="grid" style="margin:var(--s3) 0 var(--s5)">` + d.windows.map(cardHTML).join("") + `</div>`;
    }
    $("gallery").innerHTML = html;
    $("gallery-meta").textContent = "Pitch set: 12 accepted + 12 rejected per reason. Change any filter to exit.";
    wireCards();
  } catch (e) {
    $("gallery").innerHTML = `<div class="empty">Could not build pitch set (${esc(e.message)}).</div>`;
  }
}

/* ── drawer ──────────────────────────────────────────────────────────── */
async function openDrawer(wid) {
  $("drawer").hidden = false;
  $("drawer-veil").hidden = false;
  $("drawer-title").textContent = wid;
  $("drawer-sub").textContent = "loading…";
  $("drawer-body").innerHTML = "";
  $("drawer-close").focus();
  let w;
  try {
    w = await getJSON("/api/windows/" + encodeURIComponent(wid));
  } catch (e) {
    $("drawer-body").innerHTML = `<div class="empty">Not found (${esc(e.message)}).</div>`;
    return;
  }
  $("drawer-sub").textContent = `${w.factory} · ${w.worker} · t=${w.start_s}–${w.end_s}s · ${w.verdict} · ${w.stage}`;
  const big = (w.thumbs || []).map((t) =>
    `<img class="big" src="/${esc(t)}" alt="frame from ${esc(wid)}" onerror="this.remove()">`).join("");
  const evRows = (w.reasons || []).map((r) => {
    const e = r.evidence || {};
    return `<tr><td>${esc(r.code)}</td><td class="mono">${esc(e.feature)}</td><td class="mono">${esc(e.value)}</td><td class="mono">${esc(e.op)} ${esc(e.threshold)}</td></tr>`;
  }).join("") || `<tr><td colspan="4">No reject reasons — accepted or handed on.</td></tr>`;
  const feats = Object.entries(w.features || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  const vers = Object.entries(w.versions || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  const lat = w.latency_ms ? Object.entries(w.latency_ms).map(([k, v]) => `${k} ${v} ms`).join(" · ") : "—";
  $("drawer-body").innerHTML =
    big +
    `<p><span class="chip chip-${w.verdict === "review" ? "review" : w.verdict}">${esc(w.verdict)}</span> ${oodTag()}` +
    `<span class="micro">score ${w.score == null ? "—" : esc(w.score)} · latency ${esc(lat)}</span></p>` +
    `<h3>Evidence</h3><table class="ev"><thead><tr><th>Reason</th><th>Feature</th><th>Value</th><th>Op threshold</th></tr></thead><tbody>${evRows}</tbody></table>` +
    `<h3>Features</h3><dl class="kv">${feats}</dl>` +
    `<h3>Receipt</h3><dl class="kv"><dt>receipt_sha256</dt><dd>${esc(w.receipt_sha256)}</dd><dt>features_digest</dt><dd>${esc(w.features_digest)}</dd></dl>` +
    `<h3>Versions</h3><dl class="kv">${vers}</dl>` +
    `<h3>Re-verify</h3><p class="micro">Runs <span class="mono">scripts/poc_run.py --verify ${esc(wid)}</span> and compares the receipt.</p>` +
    `<p style="margin:var(--s2) 0"><button class="btn" id="reverify-btn" type="button">Re-verify this window</button></p>` +
    `<div id="reverify-out" aria-live="polite"></div>`;
  $("reverify-btn").addEventListener("click", () => reverify(wid));
}
function closeDrawer() {
  $("drawer").hidden = true;
  $("drawer-veil").hidden = true;
}
async function reverify(wid) {
  const out = $("reverify-out");
  out.innerHTML = `<p class="micro">Running…</p>`;
  try {
    const r = await fetch("/api/reverify/" + encodeURIComponent(wid), { method: "POST" });
    const body = await r.json();
    if (body.status === "unavailable") {
      out.innerHTML = `<div class="empty">Re-verify not available yet — the run worker has not provided scripts/poc_run.py.</div>`;
    } else {
      out.innerHTML = `<p><span class="chip ${body.status === "ok" ? "chip-accept" : "chip-reject"}">${esc(body.status)}</span></p>` +
        (body.stdout ? `<pre class="out">${esc(body.stdout)}</pre>` : "") +
        (body.stderr ? `<pre class="out">${esc(body.stderr)}</pre>` : "");
    }
  } catch (e) {
    out.innerHTML = `<div class="empty">Request failed (${esc(e.message)}).</div>`;
  }
}

/* ── live ────────────────────────────────────────────────────────────── */
function reasonText(r) {
  if (r == null || r === "") return "";
  if (typeof r === "string") return r;
  try {
    const s = r.s0 || r;
    if (s && typeof s === "object" && (s.rule || s.feature))
      return `${s.rule || ""}${s.feature ? ": " + s.feature : ""}` +
        (s.value != null ? ` ${s.value} ${s.op || ""} ${s.threshold ?? ""}` : "");
    return JSON.stringify(r);
  } catch (e) { return String(r); }
}
function sparkline(events) {
  const W = 560, H = 90, pad = 8;
  const n = events.length;
  if (!n) return "";
  const xs = (i) => pad + (W - 2 * pad) * (n === 1 ? 0.5 : i / (n - 1));
  const ys = (v) => H - pad - (H - 2 * pad) * Math.max(0, Math.min(1, v));
  const pts = events.map((e, i) => `${xs(i).toFixed(1)},${ys(e.running_score).toFixed(1)}`).join(" ");
  let dots = "";
  events.forEach((e, i) => {
    if (e.decision === "early_reject")
      dots += `<circle class="rej" cx="${xs(i)}" cy="${ys(e.running_score)}" r="5"><title>early reject at t=${e.t}s${e.reason ? " · " + esc(reasonText(e.reason)) : ""}</title></circle>`;
    else if (e.decision === "accept")
      dots += `<circle class="acc" cx="${xs(i)}" cy="${ys(e.running_score)}" r="4"><title>accept at t=${e.t}s</title></circle>`;
  });
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Running score over 2-second chunks">` +
    `<polyline class="line" points="${pts}"/>${dots}</svg>`;
}

async function loadLive() {
  let d;
  try {
    d = await getJSON("/api/stream");
  } catch (e) {
    $("live-body").innerHTML = `<div class="empty">Stream unavailable (${esc(e.message)}).</div>`;
    return;
  }
  if (!d.available || !d.events.length) {
    $("live-body").innerHTML = `<div class="empty">Live replay placeholder — no <span class="mono">stream_events.jsonl</span> yet. ` +
      `The demo worker writes 2-second-chunk replay events here during the pitch; this panel then shows the running-score sparkline and the moment of early reject.</div>`;
    $("live-sub").textContent = "Waiting for the demo worker's replay events.";
    return;
  }
  const byWin = {};
  for (const e of d.events) (byWin[e.window_id] = byWin[e.window_id] || []).push(e);
  $("live-sub").textContent = `${d.events.length} replay events across ${Object.keys(byWin).length} stream(s).`;
  $("live-body").innerHTML = Object.entries(byWin).map(([wid, ev]) => {
    const last = ev[ev.length - 1];
    const badge = last.decision === "early_reject"
      ? `<span class="chip chip-reject">early reject @ t=${esc(last.t)}s${last.reason ? " · " + esc(reasonText(last.reason)) : ""}</span>`
      : last.decision === "accept" ? `<span class="chip chip-accept">accept</span>`
      : `<span class="chip chip-review">in progress · chunk ${esc(last.chunk)}</span>`;
    return `<div class="stream"><h3 class="mono">${esc(wid)}</h3><p>${badge}</p>${sparkline(ev)}</div>`;
  }).join("");
}

/* ── wiring ──────────────────────────────────────────────────────────── */
document.querySelector('.honesty a[data-goto="claims"]').addEventListener("click", () => {
  // hash change renders the About view; scroll to the claims table once shown
  setTimeout(() => $("claims").scrollIntoView({ behavior: "smooth", block: "start" }), 60);
});
$("overview-refresh").addEventListener("click", loadSummary);
$("gallery-refresh").addEventListener("click", loadGallery);
$("live-refresh").addEventListener("click", loadLive);
$("pitch-btn").addEventListener("click", pitchMode);
$("prev-page").addEventListener("click", () => { if (state.page > 1) { state.page--; loadGallery(); } });
$("next-page").addEventListener("click", () => { state.page++; loadGallery(); });
for (const id of ["f-verdict", "f-reason", "f-factory", "f-worker", "f-stage", "f-sort"]) {
  $(id).addEventListener("change", () => { state.page = 1; loadGallery(); });
}
let qTimer = null;
$("f-q").addEventListener("input", () => {
  clearTimeout(qTimer);
  qTimer = setTimeout(() => { state.page = 1; loadGallery(); }, 300);
});
$("drawer-close").addEventListener("click", closeDrawer);
$("drawer-veil").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

route();
loadSummary();
setInterval(() => {
  loadSummary();
  if (state.route === "live") loadLive();
}, POLL_MS);
