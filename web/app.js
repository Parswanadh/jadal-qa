/* Jadal QA dashboard v2 — vanilla JS, no build step, no CDN.
 *
 * Five screens in one page: Batches (filterable list), QA queue (review
 * pile), Segment (receipt tree + re-verify + step timeline), Video search
 * (hub clips), Insights (risk/coverage + SOP gaps).
 * A global filter bar of real <select>s + search; ALL state lives in the
 * URL hash (see JadalApi.parseHash / buildHash) — queue reason as `qr`,
 * hub search as `hq`/`hsite`/`hverdict`/`hreason`, so filtered queues and
 * hub searches deep-link and survive reload.
 *
 * Field access rule: every payload field is read through web/api.js
 * (`JadalApi`). app.js never touches raw service keys directly.
 *
 * Live against the service (docs/SPEC.md), or against web/mock/*.json
 * with ?mock=1.
 *
 * NOT DONE:
 * - Per-head risk curves and per-segment worker/clip/latency are not
 *   exposed by the API; the UI labels the fallbacks instead of inventing
 *   data (see the notes rendered on screen).
 * - No retry/backoff jitter tuning beyond a fixed schedule.
 */

(function () {
'use strict';

var A = window.JadalApi;
if (!A) throw new Error('web/api.js must load before web/app.js');

/* ============================ basics ============================ */

var $ = function (sel, root) { return (root || document).querySelector(sel); };
var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

function esc(v) {
  return String(v === null || v === undefined ? '' : v).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

function el(html) {
  var t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

/* ---- config ---- */
var params = new URLSearchParams(location.search);
var MOCK = params.get('mock') === '1';
var API = (params.get('api') || '').replace(/\/+$/, ''); // explicit base override, default same-origin

var PICO_POLL_MS = 5000;
var WS_RECONNECT = [1000, 2000, 4000, 8000];
var SEARCH_DEBOUNCE_MS = 250;

var clientOpts = { mock: MOCK, apiBase: API, forceMismatch: false };
var client = A.createClient(clientOpts);

var state = {
  route: 'batches',       // 'batches' | 'queue' | 'segment' | 'videos' | 'insights'
  segmentId: null,
  filters: Object.assign({}, A.FILTER_DEFAULTS),
  lang: A.LANGS[0],
  head: '',
  batches: null,          // normalized batches, or null while loading
  batchesError: null,
  receipt: null,          // normalized receipt (JadalApi.normReceipt)
  receiptError: null,
  receiptId: null,        // the receipt actually displayed
  requestedId: null,      // the id the route asked for (differs on fallback)
  segBatch: null,         // batch detail the displayed receipt was opened from
  rv: { status: 'idle', result: null, error: null }, // re-verify
  rvSeq: 0,               // invalidates stale re-verify responses
  hubSeq: 0,              // invalidates stale hub-search responses
  risk: null, riskError: null,
  gaps: null, gapsError: null,
  pico: null,
  live: false, lastEvent: null,
  mockMismatch: false,
  queueReason: '',
  hub: { status: 'idle', items: [], total: undefined, mock: false, error: null },
  hubQ: '', hubSite: '', hubVerdict: '', hubReason: '',
  batchDetailCache: {},   // batch_id -> batch detail payload
  receiptCache: {},       // receipt_id -> receipt body
  applyingHash: false,    // guard: control events while syncing from hash
  searchTimer: null,
};

function wsURL() {
  return A.wsUrl(API, location.protocol, location.host);
}

/* ============================ formatting ============================ */

function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }

function num(v, d) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(d === undefined ? 4 : d);
}

function fmtTime(iso) {
  if (!iso) return '—';
  var date = new Date(iso);
  if (isNaN(date.getTime())) return String(iso);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

function fullTime(iso) {
  var date = new Date(String(iso || ''));
  return isNaN(date.getTime()) ? String(iso || '') : date.toLocaleString();
}

function verdictClass(v) {
  v = String(v || '').toLowerCase();
  if (v === 'accept') return 'verdict-accept';
  if (v === 'review') return 'verdict-review';
  if (v === 'reject') return 'verdict-reject';
  return '';
}

function verdictChip(v, extra) {
  return '<span class="verdict ' + verdictClass(v) + (extra ? ' ' + extra : '') + '">' + esc(v || '—') + '</span>';
}

/* ============================ data loading ============================ */

async function loadBatches(soft) {
  if (!soft) {
    state.batches = null;
    state.batchesError = null;
    renderBatches();
  }
  try {
    var raw = await client.batches();
    var list = A.normBatchList(raw);
    list.sort(function (a, b) {
      return String(b.created_utc || '').localeCompare(String(a.created_utc || ''));
    });
    state.batches = list;
    state.batchesError = null;
    populateFilterSelects();
    if (MOCK) loadPico();
  } catch (e) {
    if (!soft) state.batches = null;
    state.batchesError = e;
  }
  renderBatches();
}

async function getBatchDetail(id) {
  if (state.batchDetailCache[id]) return state.batchDetailCache[id];
  var detail = await client.batchDetail(id);
  state.batchDetailCache[id] = detail;
  return detail;
}

async function getReceiptBody(id) {
  if (state.receiptCache[id]) return state.receiptCache[id];
  var r = await client.receipt(id);
  state.receiptCache[r.id] = r.body;
  return r.body;
}

// Segment rows link by RECEIPT id only. Resolve segment -> receipt through
// the batch detail's receipt_ids + each receipt's verdict.segment_id.
async function openReceipt(batchId, segmentId, btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Opening…'; }
  try {
    var detail = await getBatchDetail(batchId);
    var ids = detail.receipt_ids || detail.receiptIds || [];
    var bodies = {};
    for (var i = 0; i < ids.length; i++) bodies[ids[i]] = await getReceiptBody(ids[i]);
    var rid = A.findReceiptForSegment(detail, bodies, segmentId);
    if (!rid) throw new Error('no receipt found for segment ' + segmentId);
    state.segBatch = detail;
    location.hash = A.buildHash('segment', rid, null, withLang(null));
  } catch (e) {
    if (btn) { btn.disabled = true; btn.textContent = 'Unavailable'; btn.title = String((e && e.message) || e); }
  }
}

async function loadReceipt(id) {
  state.receipt = null;
  state.receiptError = null;
  state.receiptId = null;
  state.requestedId = id;
  state.segBatch = null;
  state.rvSeq++;
  state.rv = { status: 'idle', result: null, error: null };
  renderSegment();
  try {
    var r = await client.receipt(id);
    if (state.requestedId !== id) return; // user moved on
    state.receipt = A.normReceipt(r.body);
    state.receiptId = r.id;
    state.receiptFallback = r.fallback || null;
    state.receiptCache[r.id] = r.body;
    collectHeadOptions(state.receipt);
  } catch (e) {
    if (state.requestedId !== id) return;
    state.receipt = null;
    state.receiptError = e;
  }
  renderSegment();
}

async function runReverify() {
  if (!state.receipt || !state.receipt.receipt_sha256) return;
  var my = ++state.rvSeq;
  var id = state.receiptId;
  state.rv = { status: 'busy', result: null, error: null };
  renderSegment();
  try {
    clientOpts.forceMismatch = state.mockMismatch;
    var res = await client.reverify(id, state.receipt.receipt_sha256);
    if (my !== state.rvSeq || id !== state.receiptId) return;
    state.rv = { status: 'done', result: res, error: null };
  } catch (e) {
    if (my !== state.rvSeq || id !== state.receiptId) return;
    state.rv = { status: 'done', result: null, error: e };
  }
  renderSegment();
}

async function loadInsights() {
  if (state.risk === null && state.riskError === null) renderInsights();
  try {
    state.risk = A.normRisk(await client.risk());
    state.riskError = null;
  } catch (e) {
    state.risk = null;
    state.riskError = e;
  }
  try {
    state.gaps = A.normGaps(await client.gaps());
    state.gapsError = null;
  } catch (e) {
    state.gaps = null;
    state.gapsError = e;
  }
  renderInsights();
}

async function loadPico() {
  try {
    state.pico = A.normPico(await client.pico());
  } catch (e) {
    state.pico = { connected: false, led: 'OFF', ageMs: null, offlineReason: String((e && e.message) || e || 'unreachable') };
  }
  renderPico();
  renderSystem();
}

/* ============================ filter bar ============================ */

function setSelectOptions(sel, items, allLabel) {
  var cur = sel.value;
  sel.innerHTML = '';
  sel.appendChild(el('<option value="">' + esc(allLabel) + '</option>'));
  items.forEach(function (v) {
    var o = document.createElement('option');
    o.value = v;
    o.textContent = v;
    sel.appendChild(o);
  });
  // keep the selection if it still exists, else fall back to All
  if (cur && items.indexOf(cur) !== -1) sel.value = cur;
}

function populateFilterSelects() {
  if (!state.batches) return;
  var opts = A.filterOptions(state.batches);
  state.applyingHash = true;
  try {
    setSelectOptions($('#f-site'), opts.sites, 'All sites');
    var wsel = $('#f-worker');
    setSelectOptions(wsel, opts.workers, 'All workers');
    wsel.disabled = !opts.workers.length;
    wsel.title = opts.workers.length ? '' : 'Worker ids are not exposed by the API yet';
    setSelectOptions($('#f-batch'), opts.batches, 'All batches');
    setSelectOptions($('#f-reason'), opts.reasons, 'All reasons');
  } finally {
    state.applyingHash = false;
  }
  syncControlsFromState();
}

function syncControlsFromState() {
  state.applyingHash = true;
  try {
    $('#f-site').value = state.filters.site || '';
    var w = $('#f-worker');
    if (!w.disabled) w.value = state.filters.worker || '';
    $('#f-batch').value = state.filters.batch || '';
    $('#f-verdict').value = state.filters.verdict || '';
    $('#f-reason').value = state.filters.reason || '';
    $('#f-lang').value = state.lang;
    if (document.activeElement !== $('#f-q') || $('#f-q').value !== state.filters.q) {
      $('#f-q').value = state.filters.q || '';
    }
  } finally {
    state.applyingHash = false;
  }
}

function langExtra() {
  return state.lang !== A.LANGS[0] ? { lang: state.lang } : null;
}

// Merge the shared language extra into a route's own extras, dropping
// empties the way buildHash does. Every hash built in this file goes
// through here so tab switches and filter reroutes never silently reset
// the language (or the queue/hub state they carry).
function withLang(extra) {
  var out = {};
  [extra, langExtra()].forEach(function (e) {
    if (!e) return;
    Object.keys(e).forEach(function (k) { if (e[k]) out[k] = e[k]; });
  });
  return Object.keys(out).length ? out : null;
}

function queueExtra() {
  return state.queueReason ? { qr: state.queueReason } : null;
}

function hubExtra() {
  var hx = {};
  if (state.hubQ) hx.hq = state.hubQ;
  if (state.hubSite) hx.hsite = state.hubSite;
  if (state.hubVerdict) hx.hverdict = state.hubVerdict;
  if (state.hubReason) hx.hreason = state.hubReason;
  return Object.keys(hx).length ? hx : null;
}

function pushHash() {
  var h;
  if (state.route === 'segment' && state.segmentId) {
    h = A.buildHash('segment', state.segmentId, null, withLang(null));
  } else if (state.route === 'insights') {
    h = A.buildHash('insights', null, null, withLang(state.head ? { head: state.head } : null));
  } else if (state.route === 'queue') {
    h = A.buildHash('queue', null, null, withLang(queueExtra()));
  } else if (state.route === 'videos') {
    h = A.buildHash('videos', null, null, withLang(hubExtra()));
  } else {
    h = A.buildHash('batches', null, state.filters, withLang(null));
  }
  if (location.hash !== h) location.hash = h;
  else routeFromHash();
}

function hubKey() {
  return [state.hubQ, state.hubSite, state.hubVerdict, state.hubReason].join('|');
}

function bindFilterbar() {
  ['f-site', 'f-worker', 'f-batch', 'f-verdict', 'f-reason', 'f-lang'].forEach(function (id) {
    $('#' + id).addEventListener('change', function () {
      if (state.applyingHash) return;
      state.filters.site = $('#f-site').value;
      state.filters.worker = $('#f-worker').disabled ? '' : $('#f-worker').value;
      state.filters.batch = $('#f-batch').value;
      state.filters.verdict = $('#f-verdict').value;
      state.filters.reason = $('#f-reason').value;
      state.lang = $('#f-lang').value;
      if (state.route === 'batches') pushHash();
      else location.hash = A.buildHash('batches', null, state.filters, withLang(null));
      renderBatches();
    });
  });
  $('#f-q').addEventListener('input', function () {
    if (state.applyingHash) return;
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(function () {
      state.filters.q = $('#f-q').value.trim();
      if (state.route !== 'batches') {
        location.hash = A.buildHash('batches', null, state.filters, withLang(null));
      } else pushHash();
      renderBatches();
    }, SEARCH_DEBOUNCE_MS);
  });
  $('#filterbar').addEventListener('submit', function (ev) {
    ev.preventDefault();
    clearTimeout(state.searchTimer);
    state.filters.q = $('#f-q').value.trim();
    if (state.route !== 'batches') location.hash = A.buildHash('batches', null, state.filters, withLang(null));
    else pushHash();
    renderBatches();
  });
  $('#f-reset').addEventListener('click', function () {
    state.filters = Object.assign({}, A.FILTER_DEFAULTS);
    syncControlsFromState();
    // Filters are just-cleared, so this yields a bare #/batches — plus the
    // non-default language, exactly like the submit sibling above.
    if (state.route !== 'batches') location.hash = A.buildHash('batches', null, state.filters, withLang(null));
    else pushHash();
    renderBatches();
  });
}

/* ============================ shared states ============================ */

function skeletonCards(n) {
  var html = '';
  for (var i = 0; i < n; i++) html += '<div class="skel skel-card" aria-hidden="true"></div>';
  return html;
}

function errorState(msg, retry) {
  return el(
    '<div class="state is-error" role="alert">' +
      '<p class="state-title">Couldn’t load this screen</p>' +
      '<p class="state-note">' + esc(msg) + '</p>' +
      '<button class="btn" type="button" data-retry="' + retry + '">Retry</button>' +
    '</div>');
}

/* ============================ batches screen ============================ */

function activeFilterCount() {
  var n = 0;
  ['site', 'worker', 'batch', 'verdict', 'reason', 'q'].forEach(function (k) {
    if (state.filters[k]) n++;
  });
  return n;
}

function renderBatches() {
  var body = $('#batches-body');
  var meta = $('#batches-meta');
  body.innerHTML = '';
  body.setAttribute('aria-busy', 'true');

  if (state.batchesError && !state.batches) {
    meta.hidden = true;
    body.appendChild(errorState(String(state.batchesError.message || state.batchesError), 'batches'));
    body.setAttribute('aria-busy', 'false');
    return;
  }
  if (state.batches === null) {
    meta.hidden = true;
    body.innerHTML = skeletonCards(3);
    return;
  }

  var items = A.applyFilters(state.batches, state.filters);
  var segTotal = items.reduce(function (n, x) { return n + x.rows.length; }, 0);
  var nf = activeFilterCount();

  meta.hidden = false;
  meta.textContent = items.length + (items.length === 1 ? ' batch' : ' batches') +
    ' · ' + segTotal + ' segments' +
    (nf ? ' · ' + nf + ' filter' + (nf === 1 ? '' : 's') + ' on' : '') +
    (state.batchesError ? ' · refresh failed, showing cached list' : '') +
    (MOCK ? ' · mock' : ' · live');

  $('#batches-sub').textContent = MOCK
    ? 'Fixture list from web/mock/batches.json. Switch off ?mock=1 for the live service.'
    : 'Every batch decision with its per-segment verdicts, newest first.';

  if (!items.length) {
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">No batches match</p>' +
        '<p class="state-note">Nothing ingested yet, or the filters hide everything. ' +
        '<button class="btn btn-ghost" type="button" id="empty-reset">Clear filters</button></p>' +
      '</div>'));
    var rb = $('#empty-reset');
    if (rb) rb.addEventListener('click', function () { $('#f-reset').click(); });
    body.setAttribute('aria-busy', 'false');
    return;
  }

  // One-line funnel summary with a segmented bar, ahead of the cards.
  var allRows = [];
  items.forEach(function (x) { x.rows.forEach(function (r) { allRows.push(r); }); });
  body.appendChild(funnelBar(A.funnelCounts(allRows)));

  items.forEach(function (x, i) {
    // newest card starts expanded so the list never looks empty
    body.appendChild(batchCard(x.batch, x.rows, i === 0 || nf > 0));
  });
  body.setAttribute('aria-busy', 'false');
}

function funnelBar(c) {
  function pct(n) { return c.total ? (100 * n / c.total) : 0; }
  var bar = el(
    '<div class="funnel" role="img" aria-label="Funnel: ' + c.accept + ' accept, ' +
      c.reject + ' reject, ' + c.review + ' review, ' + c.other + ' other out of ' +
      c.total + ' segments">' +
      '<p class="funnel-line">' +
        '<b>' + c.total + '</b> segments — ' +
        '<span class="verdict verdict-accept">' + c.accept + ' accept</span> · ' +
        '<span class="verdict verdict-reject">' + c.reject + ' reject</span> · ' +
        '<span class="verdict verdict-review">' + c.review + ' review</span>' +
        (c.other ? ' · <span class="other-count">' + c.other + ' other</span>' : '') +
      '</p>' +
      '<div class="funnel-track">' +
        '<span class="funnel-seg is-accept" style="width:' + pct(c.accept).toFixed(1) + '%"></span>' +
        '<span class="funnel-seg is-reject" style="width:' + pct(c.reject).toFixed(1) + '%"></span>' +
        '<span class="funnel-seg is-review" style="width:' + pct(c.review).toFixed(1) + '%"></span>' +
        (c.other ? '<span class="funnel-seg is-other" style="width:' + pct(c.other).toFixed(1) + '%"></span>' : '') +
      '</div>' +
    '</div>');
  return bar;
}

function batchCard(b, rows, open) {
  var card = el('<article class="card batch" aria-label="Batch ' + esc(b.batch_id) + '"></article>');
  var rule = b.rule ? esc(b.rule) :
    (b.k !== undefined ? 'rule: reject if rejected_count &gt; ' + esc(b.k) : 'fixed rule: reject if rejected_count &gt; k');

  card.appendChild(el(
    '<div class="batch-top">' +
      '<span class="batch-id">' + esc(b.batch_id) + '</span>' +
      verdictChip(b.decision, 'verdict-batch') +
      '<span class="batch-meta">site <b>' + esc(b.site_id) + '</b></span>' +
      '<span class="batch-meta" title="' + esc(fullTime(b.created_utc)) + '">' + esc(fmtTime(b.created_utc)) + '</span>' +
      '<span class="batch-spacer"></span>' +
      '<span class="batch-meta">' + esc(b.rejected_count) + ' rejected / ' + esc(b.n) + '</span>' +
    '</div>' +
    '<p class="batch-rule">' + rule + ' · n = ' + esc(b.n) + '</p>'));

  var det = document.createElement('details');
  det.className = 'segdetails';
  if (open) det.setAttribute('open', '');
  var sum = document.createElement('summary');
  sum.className = 'segdetails-sum';
  sum.innerHTML = '<span>' + esc(rows.length) + (rows.length === 1 ? ' segment' : ' segments') + '</span>' +
    '<span class="segdetails-caret" aria-hidden="true">▾</span>';
  det.appendChild(sum);

  var list = el('<div class="seglist"></div>');
  if (!rows.length) {
    list.appendChild(el('<p class="state-note" style="padding: var(--s3) var(--s4);">No segments match the filters in this batch.</p>'));
  }
  rows.forEach(function (r) {
    var reasons = (r.reasons && r.reasons.length)
      ? r.reasons.map(function (c) { return '<span class="reason">' + esc(c) + '</span>'; }).join('')
      : '<span class="reason-none">no flags</span>';
    var row = el(
      '<div class="seg-row">' +
        '<div class="seg-main">' +
          '<span class="seg-id">' + esc(r.segment_id) + '</span>' +
          verdictChip(r.verdict) +
        '</div>' +
        '<div class="seg-reasons">' + reasons + '</div>' +
        '<div class="seg-meta">' +
          '<button class="btn btn-open" type="button">Open receipt →</button>' +
        '</div>' +
      '</div>');
    var btn = $('.btn-open', row);
    btn.setAttribute('data-batch', b.batch_id);
    btn.setAttribute('data-segment', r.segment_id);
    btn.setAttribute('aria-label', 'Open receipt for segment ' + r.segment_id);
    list.appendChild(row);
  });
  det.appendChild(list);
  card.appendChild(det);
  return card;
}

/* ============================ segment screen ============================ */

function segHeadRow(label, value) {
  return '<div class="kv-row"><span class="kv-key">' + esc(label) +
    '</span><span class="kv-val">' + esc(value === undefined || value === null ? '—' : value) + '</span></div>';
}

function renderSegment() {
  var body = $('#segment-body');
  var actions = $('#segment-actions');
  body.innerHTML = '';
  actions.innerHTML = '';
  body.setAttribute('aria-busy', 'true');

  actions.appendChild(el(
    '<form class="voice-row" id="open-form" role="search" aria-label="Open receipt by id">' +
      '<div class="field">' +
        '<label class="field-label" for="open-id">Receipt id</label>' +
        '<input class="mono" id="open-id" name="id" inputmode="text" autocomplete="off" spellcheck="false"' +
          ' placeholder="paste a receipt id" style="font-family:var(--mono);font-size:.8125rem;padding:var(--s2) var(--s3);' +
          'border:1px solid var(--rule-2);border-radius:var(--radius);background:var(--surface);color:var(--ink);max-width:14rem;">' +
      '</div>' +
      '<button class="btn" type="submit">Open</button>' +
    '</form>'));
  $('#open-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var v = $('#open-id').value.trim();
    if (v) location.hash = A.buildHash('segment', v, null, withLang(null));
  });

  if (!state.segmentId) {
    $('#segment-title').textContent = 'Segment';
    $('#segment-sub').textContent = 'Pick a segment from the Batches screen to see its receipt.';
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">No segment selected</p>' +
        '<p class="state-note">Open any segment from the batch list, or paste a receipt id above.</p>' +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }

  if (state.receiptError && !state.receipt) {
    $('#segment-title').textContent = 'Segment';
    body.appendChild(errorState(String(state.receiptError.message || state.receiptError), 'segment'));
    body.setAttribute('aria-busy', 'false');
    return;
  }
  if (!state.receipt) {
    $('#segment-title').textContent = 'Segment';
    body.innerHTML = '<div class="skel skel-card"></div><div class="skel skel-line" style="width:60%"></div><div class="skel skel-line" style="width:40%"></div>';
    return;
  }

  renderSegmentReady(body);
  body.setAttribute('aria-busy', 'false');
}

function renderSegmentReady(body) {
  var r = state.receipt;
  var raw = r.raw;
  // All named fields come from the api.js normalizer (r.* / A.first),
  // never from raw service keys. `raw` below is only handed to the
  // generic receipt tree, which renders whatever keys exist.
  var sig = r.signals || {};
  var reasons = r.reasons;

  var segLabel = r.segment_id || state.requestedId || state.receiptId;
  var batchId = (state.segBatch && A.first(state.segBatch, ['batch_id', 'batchId'])) || null;
  var siteId = (state.segBatch && A.first(state.segBatch, ['site_id', 'siteId'])) || '—';

  $('#segment-title').textContent = 'Segment';
  $('#segment-sub').textContent = 'Receipt for ' + segLabel + (batchId ? ' · from batch ' + batchId : '');

  // ---- top: one summary card (verdict + reasons + snapshot), then the
  // Re-verify result, then a compact step strip — all above the fold.
  // The receipt tree follows immediately; Segment facts, voice, hashes
  // and signals live collapsed in <details> below it. ----
  var confTop = Number(r.answer_conf);
  var summary = el(
    '<section class="card seg-summary" aria-label="Segment summary">' +
      '<div class="card-body">' +
        '<div class="seg-head">' +
          verdictChip(r.verdict) +
          '<span class="seg-head-id">' + esc(segLabel) + '</span>' +
        '</div>' +
        '<div class="seg-reasons" style="margin-bottom:var(--s2);">' +
          (reasons.length
            ? reasons.map(function (c) { return '<span class="reason">' + esc(c) + '</span>'; }).join('')
            : '<span class="reason-none">no flags</span>') +
        '</div>' +
        '<dl class="sumline">' +
          '<div><dt>confidence</dt><dd>' + esc(isFinite(confTop) ? num(confTop) : '—') + '</dd></div>' +
          '<div><dt>threshold</dt><dd>' + esc(r.threshold !== undefined && r.threshold !== null ? r.threshold : '—') + '</dd></div>' +
          '<div><dt>site</dt><dd>' + esc(siteId) + '</dd></div>' +
          (batchId ? '<div><dt>batch</dt><dd>' + esc(batchId) + '</dd></div>' : '') +
        '</dl>' +
      '</div>' +
    '</section>');
  body.appendChild(summary);

  if (state.receiptFallback) {
    body.appendChild(el(
      '<div class="state" role="note" style="margin-bottom:var(--s4);">' +
        '<p class="state-note"><b>mock:</b> id “' + esc(state.receiptFallback) +
        '” is not in web/mock/receipts.json; showing the first fixture receipt instead (' +
        esc(state.receiptId) + ').</p>' +
      '</div>'));
  }

  // ---- top: re-verify first so verdict + reasons + result sit above
  // the fold; the receipt tree and step timeline follow immediately.
  // Segment facts, hashes, signals and voice are details, below. ----
  var hasHash = !!r.receipt_sha256;
  var rvCard = el(
    '<section class="card" aria-label="Re-verify receipt">' +
      '<div class="card-head"><h2 class="card-title">Re-verify</h2>' +
      '<span class="card-note">re-runs the pipeline, compares hashes</span></div>' +
      '<div class="card-body">' +
        '<div class="rv-head">' +
          '<button class="btn btn-primary" type="button" id="rv-run"' +
            ((state.rv.status === 'busy' || !hasHash) ? ' disabled' : '') +
            (!hasHash ? ' title="receipt has no receipt_sha256 to compare"' : '') + '>Re-verify this receipt</button>' +
          (MOCK
            ? '<label class="card-note" style="display:flex;gap:var(--s1);align-items:center;">' +
              '<input type="checkbox" id="mock-mismatch"' + (state.mockMismatch ? ' checked' : '') + '> mock: force mismatch</label>'
            : '') +
        '</div>' +
        '<div id="rv-out"></div>' +
      '</div>' +
    '</section>');
  body.appendChild(rvCard);

  renderReverify($('#rv-out'));
  $('#rv-run').addEventListener('click', runReverify);
  var mm = $('#mock-mismatch');
  if (mm) mm.addEventListener('change', function () { state.mockMismatch = mm.checked; });

  // ---- step timeline (above the fold) ----
  body.appendChild(stepTimelineCard(r));

  // ---- collapsible receipt tree (above the fold) ----
  var treeCard = el(
    '<section class="card" aria-label="Receipt detail">' +
      '<div class="card-head"><h2 class="card-title">Receipt detail</h2>' +
      '<span class="card-note">expand objects, every hash copiable</span></div>' +
      '<div class="card-body"><div class="rtree" id="receipt-tree"></div></div>' +
    '</section>');
  body.appendChild(treeCard);
  $('#receipt-tree').appendChild(receiptTree(raw, []));

  // ---- collapsed below the fold: Segment facts, voice, hashes, signals ----
  var conf = Number(r.answer_conf);
  var facts = moreDetails('Segment', fullTime(r.created_utc));
  facts.body.innerHTML =
    '<div class="kv">' +
      segHeadRow('snapshot', 'answer_conf ' + (isFinite(conf) ? num(conf) : '—') + ' · threshold ' + (r.threshold !== undefined && r.threshold !== null ? r.threshold : '—')) +
      segHeadRow('site', siteId) +
      segHeadRow('worker', '—') +
      segHeadRow('source clip', '—') +
      segHeadRow('latency', '—') +
    '</div>' +
    '<p class="state-note" style="margin-top:var(--s3);">Worker, clip path and latency are not exposed by the API yet.</p>';
  body.appendChild(facts.det);

  // ---- voice ----
  var voice = moreDetails('Voice message',
    MOCK ? 'mock placeholder tone' : 'GET /voice/{code}/{lang}');
  voice.body.innerHTML =
    '<div class="voice-row">' +
      '<div class="field"><span class="field-label">Reason code</span>' +
        '<span class="mono" style="font-size:.8125rem;">' + esc(reasons[0] || '—') + '</span></div>' +
      '<div class="field"><label class="field-label" for="voice-lang">Language</label>' +
        '<select id="voice-lang">' + A.LANGS.map(function (l) {
          return '<option value="' + l + '"' + (l === state.lang ? ' selected' : '') + '>' + l + '</option>';
        }).join('') + '</select></div>' +
      '<button class="btn" type="button" id="voice-play"' + (reasons.length ? '' : ' disabled') + '>▶ Play message</button>' +
    '</div>' +
    '<p class="audio-note" id="voice-note" role="status">' +
      (reasons.length ? 'Reads the supervisor message for the first reason code (' + esc(reasons[0]) + ').' : 'No reason codes on this segment, so there is no voice message.') +
    '</p>';
  body.appendChild(voice.det);
  $('#voice-lang').addEventListener('change', function () {
    state.lang = this.value;
    $('#f-lang').value = state.lang;
    pushHash();
  });
  $('#voice-play').addEventListener('click', function () { playVoice(reasons[0]); });

  // ---- hashes ----
  var hashes = moreDetails('Hashes', 'per docs/SPEC.md · click a value for the full string');
  hashes.body.innerHTML =
    hashRow('ID', r.receipt_id, 'id') +
    hashRow('receipt SHA-256', r.receipt_sha256, 'sha') +
    hashRow('segment SHA-256', r.segment_sha256, 'sha') +
    (r.blurred_sha256 ? hashRow('blurred file SHA-256', r.blurred_sha256, 'sha') : '');
  body.appendChild(hashes.det);
  $$('[data-copy]', hashes.det).forEach(function (b) {
    b.addEventListener('click', function () { copyText(b.getAttribute('data-copy'), 'copied', b); });
  });
  $$('.hash-val[data-full]', hashes.det).forEach(function (s) {
    function toggle() {
      var full = s.getAttribute('data-full');
      var expanded = s.getAttribute('data-expanded') === '1';
      s.textContent = expanded ? s.getAttribute('data-short') : full;
      s.setAttribute('data-expanded', expanded ? '0' : '1');
    }
    s.addEventListener('click', toggle);
    s.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggle(); }
    });
  });

  // ---- signals (compact, collapsed) ----
  var sigWrap = moreDetails('Signals', 'exact, per-segment (L0)');
  sigWrap.body.appendChild(signalsCardBody(sig));
  body.appendChild(sigWrap.det);
}

/* ---- collapsed "more" sections: long lists hide behind <details> ---- */

function moreDetails(label, note) {
  var det = el(
    '<details class="card more">' +
      '<summary class="more-sum"><span class="card-title">' + esc(label) + '</span>' +
      '<span class="card-note">' + esc(note || 'collapsed') + '</span></summary>' +
      '<div class="more-body"></div>' +
    '</details>');
  return { det: det, body: $('.more-body', det) };
}

function endsHash(v) {
  // First 8 + last 6 so the receipt ID and the SHA-256 never look
  // identical at a glance; full value is one click/hover away.
  var s = String(v || '');
  if (s.length <= 16) return s;
  return s.slice(0, 8) + '…' + s.slice(-6);
}

function hashRow(label, value, kind) {
  var v = value || '—';
  var isId = kind === 'id';
  var short = isId ? endsHash(v) : A.shortHash(v, 16);
  var cls = isId ? 'hash-val hash-id' : 'hash-val hash-sha';
  return '<div class="hash-row' + (isId ? ' is-id' : '') + '"><span class="hash-key">' + esc(label) + '</span>' +
    '<span class="' + cls + '" title="' + esc(v) + ' — click to ' +
    'toggle full value" data-full="' + esc(v) + '" data-short="' + esc(short) +
    '" data-expanded="0" role="button" tabindex="0">' + esc(short) + '</span>' +
    (value ? '<button class="btn btn-ghost btn-copy" type="button" data-copy="' + esc(v) +
      '" aria-label="Copy ' + esc(label) + '">copy</button>' : '') +
    '</div>';
}

function renderReverify(out) {
  out.innerHTML = '';
  if (state.rv.status === 'busy') {
    out.appendChild(el('<p class="rv-busy" role="status">Re-running the pipeline on the stored segment… this can take a while.</p>'));
    return;
  }
  if (state.rv.error) {
    out.appendChild(el(
      '<div class="rv-banner" role="alert">' +
        '<p class="rv-verdict">Re-verify failed</p>' +
        '<p class="rv-detail">' + esc(String(state.rv.error.message || state.rv.error)) + '</p>' +
      '</div>'));
    return;
  }
  var res = state.rv.result;
  if (!res) return;
  var match = !!res.match;
  var banner = el(
    '<div class="rv-banner ' + (match ? 'is-match' : 'is-mismatch') + '" role="status">' +
      '<p class="rv-verdict">' +
        (match ? '✓ MATCH — receipt reproduces' : '✕ MISMATCH — receipt does not reproduce') + '</p>' +
      '<p class="rv-detail">expected <span class="mono">' + esc(res.expected === null || res.expected === undefined ? '—' : res.expected) + '</span><br>' +
        'actual&nbsp;&nbsp;&nbsp; <span class="mono">' + esc(res.actual === null || res.actual === undefined ? '—' : res.actual) + '</span></p>' +
      (((res.diff || []).length)
        ? '<ul class="rv-diff">' + res.diff.map(function (d) { return '<li>' + esc(d) + '</li>'; }).join('') + '</ul>'
        : '<p class="rv-detail">no differing fields</p>') +
      (res.version_match === undefined || res.version_match === null ? '' :
        '<p class="rv-detail">code versions ' + (res.version_match ? 'match' :
          'differ' + (((res.version_diff || []).length) ? ' (' + res.version_diff.map(esc).join(', ') + ')' : '')) + '</p>') +
    '</div>');
  out.appendChild(banner);
}

/* ---- compact step strip: wraps, never scrolls sideways ---- */

function stepTimelineCard(r) {
  var names = r.step_names.length ? r.step_names : ['—'];
  var probs = r.step_probs;
  var choice = r.step_choice;
  var conf = Number(r.step_conf);
  var thr = Number(r.threshold);
  var confTxt = isFinite(conf) ? conf.toFixed(2) : '—';

  // top candidate (for the "no step detected" note when choice is null)
  var topName = null, topP = -1;
  names.forEach(function (name) {
    var p = Number(probs[name]);
    if (isFinite(p) && p > topP) { topP = p; topName = name; }
  });
  topP = topP >= 0 ? topP.toFixed(2) : '—';

  var track = el('<ol class="timeline tl-compact" aria-label="SOP step timeline"></ol>');
  names.forEach(function (name) {
    var p = Number(probs[name]) || 0;
    var isTop = name === choice;
    var li = document.createElement('li');
    li.className = 'tl-chip' + (isTop ? ' is-detected ' + verdictClass(r.verdict) : '');
    if (isTop) li.setAttribute('aria-current', 'step');
    var dot = isTop && isFinite(conf) && isFinite(thr)
      ? (conf >= thr ? 'conf-high' : 'conf-low') : '';
    li.innerHTML =
      '<span class="tl-dot ' + dot + '" aria-hidden="true"></span>' +
      '<span class="tl-name">' + esc(name) + '</span>' +
      '<span class="tl-prob">' + (probs[name] !== undefined ? num(p, 2) : '—') + '</span>' +
      (isTop ? '<span class="tl-flag">detected</span>' : '');
    li.title = name + ': p = ' + (probs[name] !== undefined ? num(p) : '—') +
      (isTop ? ' (detected, confidence ' + confTxt + ')' : '');
    track.appendChild(li);
  });

  var card = el(
    '<section class="card" aria-label="Step timeline">' +
      '<div class="card-head"><h2 class="card-title">Steps</h2>' +
      '<span class="card-note">detected step marked · conf ' + esc(confTxt) +
      (isFinite(thr) ? ' · threshold ' + thr : '') + '</span></div>' +
      '<div class="card-body"><div class="tl-wrap"></div>' +
      (choice
        ? ''
        : '<p class="state-note" style="margin-bottom:var(--s2);">No step detected — top candidate ' +
          esc(topName) + ' (' + esc(topP) + ').</p>') +
      '<p class="state-note">Hands p(yes) ' +
        esc(r.hands_p_yes !== undefined && r.hands_p_yes !== null ? num(r.hands_p_yes) : '—') +
        ' · active-work p(yes) ' +
        esc(r.active_work_p_yes !== undefined && r.active_work_p_yes !== null ? num(r.active_work_p_yes) : '—') +
      '</p></div>' +
    '</section>');
  $('.tl-wrap', card).appendChild(track);
  return card;
}

/* ---- collapsible receipt tree ---- */

function isHashLike(key, value) {
  if (typeof value !== 'string') return false;
  key = String(key || '').toLowerCase();
  if (key.indexOf('sha256') !== -1 || key === 'phash' || key === 'receipt_id') return true;
  return /^[0-9a-f]{16,128}$/i.test(value) && value.length >= 16;
}

function leafRow(key, value) {
  var row = el('<div class="rt-leaf"></div>');
  var display, full = String(value);
  if (value === null) display = 'null';
  else if (typeof value === 'boolean') display = value ? 'yes' : 'no';
  else if (typeof value === 'number') display = num(value);
  else if (typeof value === 'string' && isHashLike(key, value)) display = A.shortHash(value);
  else display = String(value);
  row.innerHTML = '<span class="rt-key">' + esc(key) + '</span>' +
    '<span class="rt-val" title="' + esc(full) + '">' + esc(display) + '</span>';
  if (typeof value === 'string' && isHashLike(key, value)) {
    var b = el('<button class="btn btn-ghost btn-copy" type="button" aria-label="Copy ' + esc(key) + '">copy</button>');
    b.addEventListener('click', function (ev) { ev.stopPropagation(); copyText(full, 'copied', b); });
    row.appendChild(b);
  }
  return row;
}

function receiptTree(value, path) {
  if (value !== null && typeof value === 'object') {
    var keys = Array.isArray(value) ? value.map(function (_, i) { return i; }) : Object.keys(value).sort();
    var det = document.createElement('details');
    det.className = 'rt-node';
    if (path.length === 0) det.setAttribute('open', '');
    var label = path.length === 0 ? 'receipt' : String(path[path.length - 1]);
    var sum = document.createElement('summary');
    sum.className = 'rt-sum';
    sum.innerHTML = '<span class="rt-key">' + esc(label) + '</span>' +
      '<span class="rt-count">' + esc(keys.length) + (Array.isArray(value) ? ' items' : ' keys') + '</span>';
    det.appendChild(sum);
    var kids = el('<div class="rt-kids"></div>');
    keys.forEach(function (k) {
      var wrap = el('<div class="rt-item"></div>');
      if (value[k] !== null && typeof value[k] === 'object') wrap.appendChild(receiptTree(value[k], path.concat([k])));
      else wrap.appendChild(leafRow(String(k), value[k]));
      kids.appendChild(wrap);
    });
    det.appendChild(kids);
    return det;
  }
  return leafRow(String(path[path.length - 1] || 'value'), value);
}

/* ---- signals (compact stat grid, no walls of text) ---- */

function signalsCardBody(sig) {
  function stat(label, value, unit) {
    return '<div class="stat"><div class="stat-label">' + esc(label) + '</div>' +
      '<div class="stat-value">' + esc(value) + (unit ? '<span class="stat-unit">' + esc(unit) + '</span>' : '') + '</div></div>';
  }
  sig = sig || {};
  var grid = el('<div class="statgrid"></div>');
  grid.innerHTML =
    stat('decodable', sig.decodable === undefined ? '—' : (sig.decodable ? 'yes' : 'no')) +
    stat('blur variance', sig.blur_var === undefined ? '—' : num(sig.blur_var, 1)) +
    stat('brightness', sig.brightness_mean === undefined ? '—' : num(sig.brightness_mean, 1)) +
    stat('motion', sig.motion_mean === undefined ? '—' : num(sig.motion_mean, 2)) +
    stat('hands visible', sig.hands_visible_frac === undefined ? '—' : num(sig.hands_visible_frac, 2)) +
    stat('hand count', sig.hand_count_mode === undefined ? '—' : String(sig.hand_count_mode)) +
    stat('faces', (sig.faces_found === undefined ? '—' : sig.faces_found) +
      (sig.faces_blurred ? ' · ' + sig.faces_blurred + ' blurred' : '')) +
    stat('underexposed', sig.under_exposed_frac === undefined ? '—' : num(sig.under_exposed_frac, 2));
  return grid;
}

function copyText(text, okMsg, btn) {
  function done() {
    // Feedback appears next to the clicked button so the user sees it
    // where they acted — never in an unrelated card.
    if (btn && btn.parentElement) flashCopied(btn, okMsg || 'copied');
    else {
      var n = $('#voice-note');
      if (n) { n.textContent = okMsg; n.classList.remove('is-bad'); }
    }
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text) && done(); });
  } else {
    fallbackCopy(text) && done();
  }
}

// Small inline "copied" note inserted right after the clicked button,
// removed after ~1.5 s. Positioned inline so it never shifts layout much.
function flashCopied(btn, msg) {
  try {
    var parent = btn.parentElement;
    if (!parent) return;
    var old = parent.querySelector('.copy-toast');
    if (old) old.remove();
    var s = document.createElement('span');
    s.className = 'copy-toast';
    s.setAttribute('role', 'status');
    s.textContent = msg || 'copied';
    btn.after(s);
    setTimeout(function () { if (s.parentElement) s.remove(); }, 1500);
  } catch (e) { /* feedback is best-effort */ }
}

function fallbackCopy(text) {
  try {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch (e) { return false; }
}

/* ============================ voice ============================ */

var audioEl = null;
var audioUrl = null;

function playVoice(code) {
  var note = $('#voice-note');
  function say(msg, bad) {
    if (note) { note.textContent = msg; note.classList.toggle('is-bad', !!bad); }
  }
  if (!code) { say('No reason code on this segment.', true); return; }
  if (MOCK) {
    say('Mock mode: playing a placeholder tone. The real message comes from GET /voice/' + code + '/' + state.lang + '.');
    mockTone(code);
    return;
  }
  say('Loading voice message for ' + code + ' in ' + state.lang + '…');
  fetch(client.voiceUrl(code, state.lang), { headers: { Accept: 'audio/*' } }).then(function (res) {
    if (!res.ok) throw new Error('GET /voice/' + code + '/' + state.lang + ' → ' + res.status +
      (res.status === 404 ? ' (the voice file has not been generated yet)' : ' ' + res.statusText));
    return res.blob();
  }).then(function (blob) {
    if (!audioEl) audioEl = new Audio();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(blob);
    audioEl.src = audioUrl;
    audioEl.onended = function () { if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; } };
    return audioEl.play();
  }).then(function () {
    say('Playing the supervisor message for ' + code + ' (' + state.lang + ').');
  }).catch(function (e) {
    say('Couldn’t play the voice message: ' + (e.message || e), true);
  });
}

// clearly a placeholder: distinct pitch per reason code, 0.4 s
function mockTone(code) {
  try {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    var ctx = mockTone.ctx || (mockTone.ctx = new Ctx());
    var idx = Math.max(0, A.REASON_CODES.indexOf(code));
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 330 + idx * 28;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.22, ctx.currentTime + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.42);
  } catch (e) { /* audio is best-effort in mock mode */ }
}

/* ============================ insights ============================ */

var headOptions = [];

function collectHeadOptions(receipt) {
  Object.keys(receipt.model_versions || {}).forEach(function (h) {
    if (headOptions.indexOf(h) === -1) headOptions.push(h);
  });
  headOptions.sort();
  var sel = $('#f-head');
  var cur = state.head;
  sel.innerHTML = '<option value="">aggregate</option>';
  headOptions.forEach(function (h) {
    var o = document.createElement('option');
    o.value = h;
    o.textContent = h;
    sel.appendChild(o);
  });
  sel.value = cur;
}

function renderInsights() {
  var body = $('#insights-body');
  body.innerHTML = '';
  body.setAttribute('aria-busy', 'true');

  $('#insights-mock-pill').hidden = !MOCK;

  var riskSec = el(
    '<section class="card" aria-label="Risk coverage curve">' +
      '<div class="card-head"><h2 class="card-title">Risk / coverage</h2>' +
      '<span class="card-note">GET /stats/risk_coverage' +
      (state.head ? ' · head: ' + esc(state.head) + ' (aggregate curve — per-head curves not exposed yet)' : '') +
      '</span></div>' +
      '<div class="card-body"><div id="risk-out"></div></div>' +
    '</section>');
  body.appendChild(riskSec);

  var rOut = $('#risk-out');
  if (state.riskError && !state.risk) {
    rOut.appendChild(errorState(String(state.riskError.message || state.riskError), 'insights'));
  } else if (!state.risk) {
    rOut.innerHTML = '<div class="skel skel-card"></div>';
  } else {
    rOut.appendChild(riskStatsRow(state.risk));
    var chart = el('<div class="chart" role="img"></div>');
    chart.appendChild(riskChart(state.risk));
    rOut.appendChild(chart);
    // The bound line is drawn only for a finite number (see riskChart /
    // JadalApi.isBound). Absent and explicit-null each get their own
    // caption so a missing bound never reads as a zero bound, and an
    // explicit null never reads as "not reported".
    var cap = A.boundCaption(state.risk.error_bound);
    if (cap) {
      rOut.appendChild(el('<p class="state-note">' + esc(cap) + '</p>'));
    }
    rOut.appendChild(riskLegend(state.risk));
  }

  var gapSec = el(
    '<section class="card" aria-label="SOP gaps">' +
      '<div class="card-head"><h2 class="card-title">Where the SOP leaks</h2>' +
      '<span class="card-note">GET /stats/sop_gaps</span></div>' +
      '<div class="card-body"><div id="gaps-out"></div></div>' +
    '</section>');
  body.appendChild(gapSec);

  var gOut = $('#gaps-out');
  if (state.gapsError && !state.gaps) {
    gOut.appendChild(errorState(String(state.gapsError.message || state.gapsError), 'insights'));
  } else if (!state.gaps) {
    gOut.innerHTML = '<div class="skel skel-card"></div>';
  } else {
    gOut.appendChild(gapsBars(state.gaps));
  }

  body.setAttribute('aria-busy', 'false');
}

function riskStatsRow(d) {
  function stat(label, value, note) {
    return '<div class="stat"><div class="stat-label">' + esc(label) + '</div>' +
      '<div class="stat-value">' + esc(value) + '</div>' +
      (note ? '<div class="stat-note">' + esc(note) + '</div>' : '') + '</div>';
  }
  var row = el('<div class="statgrid" style="margin-bottom:var(--s4);"></div>');
  // chosen_threshold has three states: a number (certified), an explicit
  // null (the API says "not certified"), and absent/undefined (the API
  // said nothing — a neutral "not reported", never a verdict).
  var ct = d.chosen_threshold;
  var ctValue, ctNote;
  if (ct === undefined) {
    ctValue = 'not reported';
    ctNote = 'no certified threshold reported by the API';
  } else if (ct === null) {
    ctValue = 'not certified';
    ctNote = 'the API returned null — nothing certified yet';
  } else {
    ctValue = num(ct);
    ctNote = (d.min_accepted !== undefined ? '≥ ' + d.min_accepted + ' accepted' : '');
  }
  var ebNote, ebValue;
  if (A.isBound(d.error_bound)) {
    ebValue = num(d.error_bound);
    ebNote = (d.delta !== undefined ? 'exact binomial, δ = ' + d.delta : 'exact binomial upper bound');
  } else {
    // Same shared caption as under the chart, so the two can never
    // disagree — including for unparseable non-null values.
    ebValue = d.error_bound === null ? 'not certified' : '—';
    ebNote = A.boundCaption(d.error_bound);
  }
  row.innerHTML =
    stat('error bound', ebValue, ebNote) +
    stat('target error', d.target_error === undefined ? '—' : num(d.target_error)) +
    stat('errors / n', (d.errors !== undefined && d.n !== undefined) ? d.errors + ' / ' + d.n : '—') +
    stat('chosen threshold', ctValue, ctNote);
  return row;
}

function riskLegend(d) {
  var html = '<div class="legend">' +
    '<span class="legend-item"><span class="legend-swatch"></span> error vs coverage</span>';
  if (A.isBound(d.error_bound)) {
    html += '<span class="legend-item"><span class="legend-swatch accent dashed"></span> error bound (exact binomial)</span>';
  }
  if (d.target_error !== undefined && isFinite(d.target_error)) {
    html += '<span class="legend-item"><span class="legend-swatch dashed"></span> target error</span>';
  }
  return el(html + '</div>');
}

function riskChart(d) {
  var NS = 'http://www.w3.org/2000/svg';
  var pts = (d.points || []).slice().sort(function (a, b) { return a.coverage - b.coverage; });
  if (!pts.length) return el('<p class="state-note">No risk/coverage points were returned.</p>');

  // Readable at 390 px with no horizontal scroll: the viewBox is narrow
  // (480 wide, so a ~360 px card shrinks it only ~0.75×) and type is set
  // large enough to survive the shrink (16-unit ticks land at ~12 px
  // effective). Only 2 + 2 ticks — endpoints only — so labels never crowd.
  // Desktop caps the width in CSS (.chart svg max-width) instead of
  // stretching the artboard, so the same type reads well at both sizes.
  var W = 480, H = 308, L = 56, R = 12, T = 26, B = 56;
  var iw = W - L - R, ih = H - T - B;
  var maxErr = 0;
  pts.forEach(function (p) { maxErr = Math.max(maxErr, p.error); });
  var hasBound = A.isBound(d.error_bound);
  var hasTarget = d.target_error !== undefined && isFinite(d.target_error);
  var hasChosen = d.chosen_threshold !== undefined && d.chosen_threshold !== null && isFinite(d.chosen_threshold);
  if (hasBound) maxErr = Math.max(maxErr, d.error_bound);
  if (hasTarget) maxErr = Math.max(maxErr, d.target_error);
  var yMax = Math.max(maxErr * 1.18, 0.01);

  function x(c) { return L + iw * clamp(c, 0, 1); }
  function y(e) { return T + ih * (1 - clamp(e / yMax, 0, 1)); }

  var svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('role', 'presentation');
  function node(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    svg.appendChild(n);
    return n;
  }

  // Endpoints only: 2 coverage ticks, 2 error ticks.
  [0, 1].forEach(function (c) {
    node('line', { x1: x(c), y1: T, x2: x(c), y2: T + ih, 'class': 'grid-line' });
    node('text', { x: x(c), y: T + ih + 22, 'text-anchor': c === 0 ? 'start' : 'end', 'class': 'tick-t' }, Math.round(c * 100) + '%');
  });
  [0, yMax].forEach(function (ev) {
    node('line', { x1: L, y1: y(ev), x2: L + iw, y2: y(ev), 'class': 'grid-line' });
    node('text', { x: L - 8, y: y(ev) + 5, 'text-anchor': 'end', 'class': 'tick-t' }, ev.toFixed(2));
  });
  node('line', { x1: L, y1: T, x2: L, y2: T + ih, 'class': 'axis-line' });
  node('line', { x1: L, y1: T + ih, x2: L + iw, y2: T + ih, 'class': 'axis-line' });
  node('text', { x: L + iw / 2, y: T + ih + 44, 'text-anchor': 'middle', 'class': 'label-t' }, 'coverage');
  node('text', { x: L, y: 12, 'text-anchor': 'start', 'class': 'label-t' }, 'error');

  var dAttr = pts.map(function (p, k) { return (k ? 'L' : 'M') + x(p.coverage).toFixed(1) + ' ' + y(p.error).toFixed(1); }).join(' ');
  node('path', { d: dAttr + ' L' + x(pts[pts.length - 1].coverage).toFixed(1) + ' ' + (T + ih) + ' L' + x(pts[0].coverage).toFixed(1) + ' ' + (T + ih) + ' Z', 'class': 'curve-area' });
  node('path', { d: dAttr, 'class': 'curve' });

  if (hasBound) {
    var yb = y(d.error_bound);
    node('line', { x1: L, y1: yb, x2: L + iw, y2: yb, 'class': 'bound' });
    node('text', { x: L + iw - 4, y: yb - 6, 'text-anchor': 'end', 'class': 'label-t' },
      'bound ' + num(d.error_bound));
  }
  if (hasTarget) {
    var yt = y(d.target_error);
    node('line', { x1: L, y1: yt, x2: L + iw, y2: yt, 'class': 'target' });
    node('text', { x: L + 4, y: yt - 6, 'class': 'label-t' }, 'target ' + num(d.target_error));
  }

  var pick = null;
  if (hasChosen) {
    pts.forEach(function (p) {
      if (!isFinite(p.threshold)) return;
      if (!pick || Math.abs(p.threshold - d.chosen_threshold) < Math.abs(pick.threshold - d.chosen_threshold)) pick = p;
    });
  }
  pts.forEach(function (p) {
    node('circle', { cx: x(p.coverage), cy: y(p.error), r: p === pick ? 5 : 2.5, 'class': 'dot' + (p === pick ? ' is-pick' : '') });
  });
  if (pick) {
    node('text', { x: clamp(x(pick.coverage) + 10, L + 4, L + iw - 120), y: clamp(y(pick.error) - 8, T + 8, T + ih), 'class': 'label-t' },
      'operating θ=' + num(pick.threshold, 2));
  }

  var noBound = A.boundCaption(d.error_bound);
  svg.setAttribute('aria-label', 'Risk/coverage curve with ' + pts.length + ' points. ' +
    (hasBound ? 'Error bound ' + num(d.error_bound) + '. '
      : (noBound.charAt(0).toUpperCase() + noBound.slice(1) + '. ')) +
    (hasTarget ? 'Target error ' + num(d.target_error) + '.' : ''));
  return svg;
}

function gapsBars(g) {
  var gaps = (g.gaps || []).slice().sort(function (a, b) { return b.count - a.count; });
  if (!gaps.length) return el('<p class="state-note">No SOP gaps were returned.</p>');
  var max = Math.max.apply(null, gaps.map(function (x) { return x.count; })) || 1;
  var wrap = el('<div class="gaps"></div>');
  gaps.forEach(function (x, i) {
    var pct = (x.count / max) * 100;
    var share = x.share !== undefined && x.share !== null
      ? (x.share <= 1 ? Math.round(x.share * 100) + '%' : x.share)
      : null;
    wrap.appendChild(el(
      '<div class="gap-row">' +
        '<div class="gap-label">' + esc(x.code) +
          (x.section ? '<span class="gap-sub">SOP ' + esc(x.section) + '</span>' : '') +
        '</div>' +
        '<div class="gap-track" role="img" aria-label="' + esc(x.code) + ': ' + esc(x.count) + ' segments">' +
          '<div class="gap-fill' + (i === 0 ? ' is-top' : '') + '" style="width:' + pct.toFixed(1) + '%"></div>' +
        '</div>' +
        '<div class="gap-value">' + esc(x.count) + (share !== null ? ' · ' + esc(share) : '') + '</div>' +
      '</div>'));
  });
  var note = 'Across ' + (g.total !== undefined ? esc(g.total) + ' segments' : 'the reporting window') +
    (g.window ? ' · ' + esc(g.window) : '') + '.';
  wrap.appendChild(el('<p class="state-note">' + note + '</p>'));
  return wrap;
}

/* ============================ QA queue ============================ */

function renderQueue() {
  var body = $('#queue-body');
  body.innerHTML = '';
  body.setAttribute('aria-busy', 'true');

  $('#queue-mock-pill').hidden = !MOCK;

  if (state.batchesError && !state.batches) {
    $('#queue-meta').hidden = true;
    body.appendChild(errorState(String(state.batchesError.message || state.batchesError), 'queue'));
    body.setAttribute('aria-busy', 'false');
    return;
  }
  if (state.batches === null) {
    $('#queue-meta').hidden = true;
    body.innerHTML = skeletonCards(2);
    return;
  }

  var rows = A.queueRows(state.batches);
  var reasons = A.queueReasons(rows);
  var sel = $('#q-reason');
  var cur = state.queueReason || '';
  sel.innerHTML = '<option value="">All reasons</option>' +
    reasons.map(function (c) {
      return '<option value="' + esc(c) + '"' + (c === cur ? ' selected' : '') + '>' + esc(c) + '</option>';
    }).join('');
  if (reasons.indexOf(cur) === -1) {
    // Unknown ?qr=... (stale link, data changed): coerce to All and push
    // so the URL stops advertising a filter the UI is not showing.
    if (cur) { state.queueReason = ''; pushHash(); }
    sel.value = '';
  }

  var list = state.queueReason
    ? rows.filter(function (r) { return (r.reasons || []).indexOf(state.queueReason) !== -1; })
    : rows;

  $('#queue-meta').hidden = false;
  $('#queue-meta').textContent = list.length + (list.length === 1 ? ' segment' : ' segments') +
    ' awaiting review' + (state.queueReason ? ' · reason ' + state.queueReason : '') +
    (MOCK ? ' · mock' : ' · live');

  if (!list.length) {
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">Queue is clear</p>' +
        '<p class="state-note">No review-routed segments' +
        (state.queueReason ? ' with reason ' + esc(state.queueReason) : '') + '.</p>' +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }

  var wrap = el('<div class="card"><div class="seglist"></div></div>');
  var inner = $('.seglist', wrap);
  list.forEach(function (r) {
    var chips = (r.reasons && r.reasons.length)
      ? r.reasons.map(function (c) { return '<span class="reason">' + esc(c) + '</span>'; }).join('')
      : '<span class="reason-none">no flags</span>';
    var row = el(
      '<div class="seg-row q-row">' +
        '<div class="seg-main">' +
          '<span class="q-prio" title="queue priority — “—” means the field is absent">' +
            esc(r.queue_priority === null || r.queue_priority === undefined ? '—' : r.queue_priority) + '</span>' +
          '<span class="seg-id">' + esc(r.segment_id) + '</span>' +
          verdictChip(r.verdict) +
        '</div>' +
        '<div class="seg-reasons">' + chips + '</div>' +
        '<div class="seg-meta"><span>' + esc(r.site_id) + ' · ' + esc(r.batch_id) + '</span>' +
          '<button class="btn btn-open" type="button">Open segment →</button>' +
        '</div>' +
      '</div>');
    var btn = $('.btn-open', row);
    btn.setAttribute('data-batch', r.batch_id);
    btn.setAttribute('data-segment', r.segment_id);
    btn.setAttribute('aria-label', 'Open segment ' + r.segment_id);
    inner.appendChild(row);
  });
  body.appendChild(wrap);
  body.setAttribute('aria-busy', 'false');
}

/* ============================ video search (mock placeholder) ============================ */

function syncHubControls() {
  $('#v-q').value = state.hubQ || '';
  var opts = state.batches ? A.filterOptions(state.batches) : null;
  var sites = opts ? opts.sites : [];
  var ssel = $('#v-site');
  var curS = state.hubSite || '';
  ssel.innerHTML = '<option value="">All sites</option>' +
    sites.map(function (s) {
      return '<option value="' + esc(s) + '"' + (s === curS ? ' selected' : '') + '>' + esc(s) + '</option>';
    }).join('');
  if (sites.indexOf(curS) === -1 && curS) { state.hubSite = ''; ssel.value = ''; }
  $('#v-verdict').value = state.hubVerdict || '';
  var rsel = $('#v-reason');
  var reasons = opts ? opts.reasons : A.REASON_CODES;
  var curR = state.hubReason || '';
  rsel.innerHTML = '<option value="">All reasons</option>' +
    reasons.map(function (c) {
      return '<option value="' + esc(c) + '"' + (c === curR ? ' selected' : '') + '>' + esc(c) + '</option>';
    }).join('');
  if (reasons.indexOf(curR) === -1 && curR) { state.hubReason = ''; rsel.value = ''; }
}

function renderVideos() {
  var body = $('#videos-body');
  body.innerHTML = '';
  body.setAttribute('aria-busy', 'true');
  syncHubControls();
  $('#videos-mock-pill').hidden = !MOCK;

  var h = state.hub;
  if (!h || h.status === 'idle') {
    $('#videos-meta').hidden = true;
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">Search the video hub</p>' +
        '<p class="state-note">Type a query and press Search. ' +
        (MOCK ? 'Mock mode: results are placeholders projected from the batch fixtures — no footage.' : '') + '</p>' +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }
  if (h.status === 'busy') {
    $('#videos-meta').hidden = true;
    body.innerHTML = skeletonCards(2);
    return;
  }
  if (h.status === 'error') {
    $('#videos-meta').hidden = true;
    var msg = String((h.error && h.error.message) || h.error || 'search failed');
    // The adapter formats transport failures as "GET <path> → <status>",
    // so a 404 from the hub search endpoint lands here without app.js
    // naming any hub URL itself.
    var hubGone = msg.indexOf('→ 404') !== -1;
    body.appendChild(el(
      '<div class="state is-error" role="alert">' +
        '<p class="state-title">' +
          (hubGone ? 'Video hub not available in this deployment' : 'Couldn’t search the hub') + '</p>' +
        '<p class="state-note">' + esc(hubGone ? 'The hub search endpoint returned 404.' : msg) + '</p>' +
        (hubGone ? '' : '<button class="btn" type="button" data-retry="videos">Retry</button>') +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }

  var items = h.items || [];
  $('#videos-meta').hidden = false;
  $('#videos-meta').textContent = items.length + (items.length === 1 ? ' result' : ' results') +
    (h.total !== undefined && h.total !== items.length ? ' of ' + h.total : '') +
    (h.mock ? ' · mock placeholders' : ' · live');
  if (!items.length) {
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">No footage matches</p>' +
        '<p class="state-note">Try a shorter query or fewer filters.</p>' +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }

  var grid = el('<div class="video-grid"></div>');
  items.forEach(function (it) {
    var card = el(
      '<article class="card v-card">' +
        '<div class="v-thumb"></div>' +
        '<div class="v-meta">' +
          '<span class="seg-id">' + esc(it.segment_id) + '</span>' +
          verdictChip(it.verdict) +
          '<div class="seg-reasons">' +
            ((it.reasons && it.reasons.length)
              ? it.reasons.map(function (c) { return '<span class="reason">' + esc(c) + '</span>'; }).join('')
              : '<span class="reason-none">no flags</span>') +
          '</div>' +
          '<p class="state-note">' + esc(it.site_id) + '</p>' +
        '</div>' +
        '<div class="v-actions"></div>' +
        '<div class="v-player"></div>' +
      '</article>');
    var thumb = $('.v-thumb', card);
    var src = it.thumb_url || ((!h.mock && !MOCK) ? client.hubThumbUrl(it.segment_id) : null);
    if (src) {
      var img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = 'thumbnail for ' + it.segment_id;
      img.src = src;
      img.addEventListener('error', function () {
        thumb.innerHTML = '<div class="v-thumb-missing">no thumbnail</div>';
      });
      thumb.appendChild(img);
    } else {
      thumb.innerHTML = '<div class="v-thumb-missing">mock placeholder — no footage</div>';
    }
    var actions = $('.v-actions', card);
    if (h.mock || MOCK) {
      actions.appendChild(el('<p class="state-note">mock: playback disabled — live plays ' +
        '<span class="mono">/hub/video/' + esc(it.segment_id) + '.mp4</span>.</p>'));
    } else {
      var play = el('<button class="btn" type="button">▶ Play clip</button>');
      play.addEventListener('click', function () {
        var holder = $('.v-player', card);
        if (holder.firstChild) { holder.innerHTML = ''; play.textContent = '▶ Play clip'; return; }
        var v = document.createElement('video');
        v.controls = true;
        v.preload = 'metadata';
        v.src = it.video_url || client.hubVideoUrl(it.segment_id, it.start_s, it.end_s);
        holder.appendChild(v);
        play.textContent = '■ Close player';
      });
      actions.appendChild(play);
    }
    grid.appendChild(card);
  });
  body.appendChild(grid);
  body.setAttribute('aria-busy', 'false');
}

async function loadHub() {
  // Stale-response guard mirroring runReverify's rvSeq: a rapid second
  // search invalidates the first, so out-of-order resolves are dropped.
  var my = ++state.hubSeq;
  state.hubKeyFor = hubKey();
  state.hub = { status: 'busy', items: [], error: null };
  renderVideos();
  try {
    var res = await client.hubSearch(state.hubQ || '', {
      site: state.hubSite || '', verdict: state.hubVerdict || '', reason: state.hubReason || '',
    });
    if (my !== state.hubSeq) return;
    state.hub = { status: 'done', items: res.items || [], total: res.total, mock: !!res.mock, error: null };
  } catch (e) {
    if (my !== state.hubSeq) return;
    state.hub = { status: 'error', items: [], error: e };
  }
  renderVideos();
}

/* ============================ pico strip + system drawer ============================ */

function renderPico() {
  var led = $('#pico-led');
  var line = $('#pico-line');
  var p = state.pico;
  if (!p) {
    led.setAttribute('data-led', 'off');
    line.textContent = 'checking…';
    return;
  }
  led.setAttribute('data-led', p.led || 'off');
  if (!p.connected) {
    line.textContent = 'offline' + (p.offlineReason ? ' — ' + p.offlineReason : ' — no status light') + (MOCK ? ' · mock' : '');
    return;
  }
  var word = { G: 'green', A: 'amber', R: 'red', OFF: 'off' }[p.led] || 'off';
  var bits = ['pico ' + word];
  if (p.port) bits.push(p.port);
  bits.push(p.ageMs === null || p.ageMs === undefined ? 'heartbeat age unknown' : 'beat ' + fmtBeatAge(p.ageMs));
  line.textContent = bits.join(' · ') + (MOCK ? ' · mock' : '');
}

function fmtBeatAge(ms) {
  if (ms < 1500) return 'just now';
  if (ms < 60000) return Math.round(ms / 1000) + ' s ago';
  return Math.round(ms / 60000) + ' min ago';
}

// Collapsed-by-default drawer: Pico status, model versions, pipeline version.
function renderSystem() {
  var box = $('#sys-body');
  box.innerHTML = '';
  var p = state.pico;
  var ledWord = p ? ({ G: 'green', A: 'amber', R: 'red', OFF: 'off' })[p.led] || 'off' : 'unknown';

  box.appendChild(el(
    '<div class="kv">' +
      segHeadRow('pico', p && p.connected ? 'connected · LED ' + ledWord : 'offline') +
      segHeadRow('port', p && p.port ? p.port : '—') +
      segHeadRow('heartbeat age', p && p.ageMs !== null && p.ageMs !== undefined ? fmtBeatAge(p.ageMs) : '—') +
      segHeadRow('pico ticks', p && p.ticks !== undefined && p.ticks !== null ? String(p.ticks) : '—') +
    '</div>'));

  var r = state.receipt;
  if (r) {
    var mv = el('<div class="kv" style="margin-top:var(--s3);"></div>');
    var names = Object.keys(r.model_versions);
    mv.innerHTML = names.length
      ? names.map(function (m) {
          return '<div class="kv-row"><span class="kv-key">model · ' + esc(m) + '</span>' +
            '<span class="kv-val">' + esc(r.model_versions[m]) + '</span></div>';
        }).join('')
      : '<div class="kv-row"><span class="kv-key">models</span><span class="kv-val">—</span></div>';
    mv.innerHTML += '<div class="kv-row"><span class="kv-key">pipeline</span>' +
      '<span class="kv-val">' + esc(r.pipeline_version || '—') + '</span></div>';
    var vers = r.config_versions || {};
    ['python', 'torch', 'opencv'].forEach(function (k) {
      if (vers[k]) {
        mv.innerHTML += '<div class="kv-row"><span class="kv-key">' + esc(k) + '</span>' +
          '<span class="kv-val">' + esc(vers[k]) + '</span></div>';
      }
    });
    box.appendChild(mv);
  } else {
    box.appendChild(el('<p class="state-note" style="margin-top:var(--s3);">Open a segment to see model and pipeline versions.</p>'));
  }
}

/* ============================ live channel ============================ */

function onPushEvent(data) {
  state.lastEvent = Date.now();
  var live = $('#live-pill');
  live.hidden = false;
  live.textContent = 'live · event ' + ageText(state.lastEvent);
  try {
    var type = data && data.type;
    if (type === 'hello' || type === 'pong') return;
    if (data && (data.batch_id || data.batchId || data.id) && (data.verdicts || data.rows || data.segments)) {
      var nb = A.normBatch(data);
      var found = false;
      state.batches = (state.batches || []).map(function (b) {
        if (b.batch_id === nb.batch_id) { found = true; return nb; }
        return b;
      });
      if (!found) state.batches.unshift(nb);
      state.batches.sort(function (a, b) {
        return String(b.created_utc || '').localeCompare(String(a.created_utc || ''));
      });
      populateFilterSelects();
      if (state.route === 'batches') renderBatches();
    } else if (state.route === 'batches') {
      loadBatches(true);
    }
  } catch (e) { /* a bad push must never break the list */ }
}

function ageText(ts) {
  if (!ts) return '—';
  var s = Math.max(0, (Date.now() - ts) / 1000);
  if (s < 5) return 'just now';
  if (s < 60) return Math.floor(s) + ' s ago';
  var m = s / 60;
  if (m < 60) return Math.floor(m) + ' min ago';
  return Math.floor(m / 60) + ' h ' + Math.floor(m % 60) + ' min ago';
}

function connectWS() {
  if (MOCK) return;
  var attempt = 0;
  var pill = $('#live-pill');

  function open() {
    var ws;
    try {
      ws = new WebSocket(wsURL());
    } catch (e) {
      schedule();
      return;
    }
    ws.onopen = function () {
      attempt = 0;
      state.live = true;
      pill.hidden = false;
      pill.textContent = 'live';
    };
    ws.onmessage = function (ev) {
      var data = null;
      try { data = JSON.parse(ev.data); } catch (e) { data = ev.data; }
      onPushEvent(data);
    };
    ws.onclose = function () {
      state.live = false;
      schedule();
    };
    ws.onerror = function () {
      try { ws.close(); } catch (e) { /* noop */ }
    };
  }

  function schedule() {
    pill.hidden = false;
    pill.textContent = 'reconnecting…';
    var wait = WS_RECONNECT[Math.min(attempt++, WS_RECONNECT.length - 1)];
    setTimeout(open, wait);
  }

  open();
}

function tickLivePill() {
  if (state.route === 'batches' && state.live && state.lastEvent) {
    $('#live-pill').textContent = 'live · event ' + ageText(state.lastEvent);
  }
  if (state.pico && state.pico.connected) renderPico();
  setTimeout(tickLivePill, 5000);
}

/* ============================ router ============================ */

var TITLES = { batches: 'Batches', segment: 'Segment', insights: 'Insights', queue: 'QA queue', videos: 'Video search' };

var VIEWS = ['batches', 'segment', 'insights', 'queue', 'videos'];

function setRoute(name, id) {
  state.route = name;
  VIEWS.forEach(function (n) {
    var view = $('#view-' + n);
    var tab = $('#tab-' + n);
    if (!view || !tab) return;
    var on = n === name;
    view.hidden = !on;
    tab.classList.toggle('is-on', on);
    tab.setAttribute('aria-selected', on ? 'true' : 'false');
    tab.tabIndex = on ? 0 : -1;
  });
  document.title = 'Jadal QA — ' + (TITLES[name] || 'dashboard');

  if (name === 'batches') {
    if (state.batches === null || state.batchesError) loadBatches(false);
    else renderBatches();
  }
  if (name === 'segment') {
    if (!id) renderSegment();
    else if (id !== state.requestedId || (!state.receipt && !state.receiptError)) loadReceipt(id);
    else renderSegment();
  }
  if (name === 'insights' && (state.risk === null || state.gaps === null)) loadInsights();
  else if (name === 'insights') renderInsights();
  if (name === 'queue') {
    if (state.batches === null || state.batchesError) {
      loadBatches(false).then(function () { if (state.route === 'queue') renderQueue(); });
      renderQueue();
    } else renderQueue();
  }
  if (name === 'videos') {
    if (state.batches === null && !state.batchesError) {
      loadBatches(true).then(function () { if (state.route === 'videos') renderVideos(); });
    }
    // Hash-driven (deep link, back button) searches refetch; in-view
    // re-renders reuse the last result.
    if (!state.hub || state.hub.status === 'idle' || state.hubKeyFor !== hubKey()) loadHub();
    else renderVideos();
  }
}

function routeFromHash() {
  var parsed = A.parseHash(location.hash);
  state.filters = parsed.filters;
  state.lang = parsed.lang;
  // Never interpolate the hash into a selector: a value like head=a"b
  // would throw a SyntaxError and kill routing. Compare option values
  // directly, and reset unknown heads to the aggregate view.
  state.head = typeof parsed.head === 'string' ? parsed.head : '';
  var sel = $('#f-head');
  if (sel) {
    var has = Array.prototype.some.call(sel.options, function (o) { return o.value === state.head; });
    if (has) sel.value = state.head;
    else { sel.value = ''; state.head = ''; }
  }
  syncControlsFromState();
  state.segmentId = parsed.route === 'segment' ? parsed.id : null;
  // Queue + hub search state round-trips through the hash like everything
  // else, so filtered queues and hub searches are deep-linkable.
  state.queueReason = parsed.queueReason || '';
  state.hubQ = (parsed.hub && parsed.hub.q) || '';
  state.hubSite = (parsed.hub && parsed.hub.site) || '';
  state.hubVerdict = (parsed.hub && parsed.hub.verdict) || '';
  state.hubReason = (parsed.hub && parsed.hub.reason) || '';
  setRoute(parsed.route, state.segmentId);
}

/* ============================ boot ============================ */

function bindStatic() {
  $$('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      var route = tab.getAttribute('data-route');
      if (route === state.route) return;
      if (route === 'segment' && state.segmentId) {
        location.hash = A.buildHash('segment', state.segmentId, null, withLang(null));
      } else if (route === 'segment') {
        location.hash = A.buildHash('segment', null, null, withLang(null));
      } else if (route === 'insights') {
        location.hash = A.buildHash('insights', null, null, withLang(state.head ? { head: state.head } : null));
      } else if (route === 'queue') {
        location.hash = A.buildHash('queue', null, null, withLang(queueExtra()));
      } else if (route === 'videos') {
        location.hash = A.buildHash('videos', null, null, withLang(hubExtra()));
      } else {
        location.hash = A.buildHash('batches', null, state.filters, withLang(null));
      }
    });
    tab.addEventListener('keydown', function (ev) {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      var tabs = $$('.tab');
      var i = tabs.indexOf(tab);
      var next = ev.key === 'ArrowRight' ? (i + 1) % tabs.length : (i - 1 + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
      ev.preventDefault();
    });
  });

  bindFilterbar();

  $('#f-head').addEventListener('change', function () {
    state.head = this.value;
    pushHash();
    renderInsights();
  });

  $('#batches-refresh').addEventListener('click', function () { loadBatches(false); });
  $('#insights-refresh').addEventListener('click', function () {
    state.risk = null; state.riskError = null;
    state.gaps = null; state.gapsError = null;
    loadInsights();
  });

  document.addEventListener('click', function (ev) {
    var t = ev.target;
    if (t && t.getAttribute) {
      var retry = t.getAttribute('data-retry');
      if (retry === 'batches') { loadBatches(false); return; }
      if (retry === 'segment') { loadReceipt(state.segmentId); return; }
      if (retry === 'insights') { loadInsights(); return; }
      if (retry === 'queue') { loadBatches(false).then(function () { renderQueue(); }); return; }
      if (retry === 'videos') { loadHub(); return; }
      if (t.classList && t.classList.contains('btn-open')) {
        openReceipt(t.getAttribute('data-batch'), t.getAttribute('data-segment'), t);
        return;
      }
    }
  });

  $('#q-reason').addEventListener('change', function () {
    state.queueReason = this.value;
    pushHash();
    renderQueue();
  });

  $('#videos-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    state.hubQ = $('#v-q').value.trim();
    state.hubSite = $('#v-site').value;
    state.hubVerdict = $('#v-verdict').value;
    state.hubReason = $('#v-reason').value;
    pushHash();
    loadHub();
  });
  $('#videos-reset').addEventListener('click', function () {
    state.hubQ = ''; state.hubSite = ''; state.hubVerdict = ''; state.hubReason = '';
    syncHubControls();
    // Push the cleared state first (yields a clean #/videos, lang kept):
    // the follow-up hashchange finds hubKeyFor already on the cleared key,
    // so it re-renders without refetching.
    pushHash();
    loadHub();
  });

  window.addEventListener('hashchange', routeFromHash);

  var foot = $('#foot-source');
  foot.textContent = MOCK
    ? 'Source: mock fixtures in web/mock/*.json (?mock=1). Illustrative data for UI work — not measurements.'
    : 'Source: live service at ' + (API || location.origin) + ' · GET /batches · WS /ws · GET /stats/* · GET /pico/status.';
  if (MOCK) { var mp = $('#mock-pill'); if (mp) mp.hidden = false; }
}

function boot() {
  bindStatic();
  loadPico();
  setInterval(loadPico, PICO_POLL_MS);
  setTimeout(tickLivePill, 5000);
  connectWS();
  routeFromHash();
  if (state.route === 'batches') loadBatches(false);
  if (state.route === 'insights') loadInsights();
  if (state.route === 'segment' && state.segmentId) loadReceipt(state.segmentId);
  else if (state.route === 'segment') renderSegment();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

})();
