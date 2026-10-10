/* Jadal QA dashboard v2 — vanilla JS, no build step, no CDN.
 *
 * Three screens in one page: Batches (filterable list), Segment (receipt
 * tree + re-verify + step timeline), Insights (risk/coverage + SOP gaps).
 * A global filter bar of real <select>s + search; ALL state lives in the
 * URL hash (see JadalApi.parseHash / buildHash).
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
  route: 'batches',       // 'batches' | 'segment' | 'insights'
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
  risk: null, riskError: null,
  gaps: null, gapsError: null,
  pico: null,
  live: false, lastEvent: null,
  mockMismatch: false,
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
    location.hash = A.buildHash('segment', rid, null, { lang: state.lang });
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

function pushHash() {
  var extra = { lang: state.lang !== A.LANGS[0] ? state.lang : '' };
  if (state.route === 'insights' && state.head) extra.head = state.head;
  var h = state.route === 'segment' && state.segmentId
    ? A.buildHash('segment', state.segmentId, null, { lang: state.lang !== A.LANGS[0] ? state.lang : '' })
    : state.route === 'insights'
      ? A.buildHash('insights', null, null, extra.head ? { head: state.head } : null)
      : A.buildHash('batches', null, state.filters, state.lang !== A.LANGS[0] ? { lang: state.lang } : null);
  if (location.hash !== h) location.hash = h;
  else routeFromHash();
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
      else location.hash = A.buildHash('batches', null, state.filters, null);
      renderBatches();
    });
  });
  $('#f-q').addEventListener('input', function () {
    if (state.applyingHash) return;
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(function () {
      state.filters.q = $('#f-q').value.trim();
      if (state.route !== 'batches') {
        location.hash = A.buildHash('batches', null, state.filters, null);
      } else pushHash();
      renderBatches();
    }, SEARCH_DEBOUNCE_MS);
  });
  $('#filterbar').addEventListener('submit', function (ev) {
    ev.preventDefault();
    clearTimeout(state.searchTimer);
    state.filters.q = $('#f-q').value.trim();
    if (state.route !== 'batches') location.hash = A.buildHash('batches', null, state.filters, null);
    else pushHash();
    renderBatches();
  });
  $('#f-reset').addEventListener('click', function () {
    state.filters = Object.assign({}, A.FILTER_DEFAULTS);
    syncControlsFromState();
    if (state.route !== 'batches') location.hash = '#/batches';
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

  items.forEach(function (x, i) {
    // newest card starts expanded so the list never looks empty
    body.appendChild(batchCard(x.batch, x.rows, i === 0 || nf > 0));
  });
  body.setAttribute('aria-busy', 'false');
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
    if (v) location.hash = A.buildHash('segment', v, null, null);
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
  var sig = raw.signals || {};
  var reasons = r.reasons;

  var segLabel = r.segment_id || state.requestedId || state.receiptId;
  var batchId = (state.segBatch && (state.segBatch.batch_id || state.segBatch.batchId)) || null;
  var siteId = (state.segBatch && (state.segBatch.site_id || state.segBatch.siteId)) || '—';

  $('#segment-title').textContent = 'Segment';
  $('#segment-sub').textContent = 'Receipt for ' + segLabel + (batchId ? ' · from batch ' + batchId : '');

  var head = el(
    '<div class="seg-head">' +
      verdictChip(r.verdict) +
      '<span class="seg-head-id">' + esc(segLabel) + '</span>' +
    '</div>');
  body.appendChild(head);

  if (reasons.length) {
    var rc = el('<div class="seg-reasons" style="margin-bottom:var(--s4);"></div>');
    reasons.forEach(function (c) { rc.appendChild(el('<span class="reason">' + esc(c) + '</span>')); });
    body.appendChild(rc);
  }

  if (state.receiptFallback) {
    body.appendChild(el(
      '<div class="state" role="note" style="margin-bottom:var(--s4);">' +
        '<p class="state-note"><b>mock:</b> id “' + esc(state.receiptFallback) +
        '” is not in web/mock/receipts.json; showing the first fixture receipt instead (' +
        esc(state.receiptId) + ').</p>' +
      '</div>'));
  }

  // ---- meta + reverify grid ----
  var grid = el('<div class="panel-grid panel-grid-2"></div>');
  var conf = Number(r.answer_conf);
  grid.appendChild(el(
    '<section class="card" aria-label="Segment facts">' +
      '<div class="card-head"><h2 class="card-title">Segment</h2>' +
      '<span class="card-note">' + esc(fullTime(r.created_utc)) + '</span></div>' +
      '<div class="card-body"><div class="kv">' +
        segHeadRow('snapshot', 'answer_conf ' + (isFinite(conf) ? num(conf) : '—') + ' · threshold ' + (r.threshold !== undefined && r.threshold !== null ? r.threshold : '—')) +
        segHeadRow('site', siteId) +
        segHeadRow('worker', '—') +
        segHeadRow('source clip', '—') +
        segHeadRow('latency', '—') +
      '</div>' +
      '<p class="state-note" style="margin-top:var(--s3);">Worker, clip path and latency are not exposed by the API yet.</p>' +
      '</div>' +
    '</section>'));

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
  grid.appendChild(rvCard);
  body.appendChild(grid);

  renderReverify($('#rv-out'));
  $('#rv-run').addEventListener('click', runReverify);
  var mm = $('#mock-mismatch');
  if (mm) mm.addEventListener('change', function () { state.mockMismatch = mm.checked; });

  // ---- step timeline ----
  body.appendChild(stepTimelineCard(r));

  // ---- voice ----
  var voiceCard = el(
    '<section class="card" aria-label="Voice message">' +
      '<div class="card-head"><h2 class="card-title">Voice message</h2>' +
      '<span class="card-note">' + (MOCK ? 'mock: placeholder tone, real audio comes from the service' : 'GET /voice/{code}/{lang}') + '</span></div>' +
      '<div class="card-body">' +
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
        '</p>' +
      '</div>' +
    '</section>');
  body.appendChild(voiceCard);
  $('#voice-lang').addEventListener('change', function () {
    state.lang = this.value;
    $('#f-lang').value = state.lang;
    pushHash();
  });
  $('#voice-play').addEventListener('click', function () { playVoice(reasons[0]); });

  // ---- hashes ----
  var hashCard = el(
    '<section class="card" aria-label="Hashes">' +
      '<div class="card-head"><h2 class="card-title">Hashes</h2>' +
      '<span class="card-note">SHA-256, per docs/SPEC.md · hover for full value, button copies</span></div>' +
      '<div class="card-body">' +
        hashRow('receipt id', r.receipt_id) +
        hashRow('receipt sha256', r.receipt_sha256) +
        hashRow('segment sha256', r.segment_sha256) +
        (r.blurred_sha256 ? hashRow('blurred file sha256', r.blurred_sha256) : '') +
      '</div>' +
    '</section>');
  body.appendChild(hashCard);
  $$('[data-copy]', hashCard).forEach(function (b) {
    b.addEventListener('click', function () { copyText(b.getAttribute('data-copy'), 'Hash copied.'); });
  });

  // ---- signals (compact) ----
  body.appendChild(signalsCard(sig));

  // ---- collapsible receipt tree (instead of a JSON dump) ----
  var treeCard = el(
    '<section class="card" aria-label="Receipt detail">' +
      '<div class="card-head"><h2 class="card-title">Receipt detail</h2>' +
      '<span class="card-note">expand objects, every hash copiable</span></div>' +
      '<div class="card-body"><div class="rtree" id="receipt-tree"></div></div>' +
    '</section>');
  body.appendChild(treeCard);
  $('#receipt-tree').appendChild(receiptTree(raw, []));
}

function hashRow(label, value) {
  var v = value || '—';
  var short = A.shortHash(v);
  return '<div class="hash-row"><span class="hash-key">' + esc(label) + '</span>' +
    '<span class="hash-val" title="' + esc(v) + '">' + esc(short) + '</span>' +
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
    '</div>');
  out.appendChild(banner);
}

/* ---- step timeline: SOP steps left-to-right, detected step marked ---- */

function stepTimelineCard(r) {
  var names = r.step_names.length ? r.step_names : ['—'];
  var probs = r.step_probs;
  var choice = r.step_choice;
  var conf = Number(r.step_conf);
  var thr = Number(r.threshold);
  var confTxt = isFinite(conf) ? conf.toFixed(2) : '—';

  var track = el('<ol class="timeline" aria-label="SOP step timeline"></ol>');
  names.forEach(function (name) {
    var p = Number(probs[name]) || 0;
    var isTop = name === choice;
    var li = document.createElement('li');
    li.className = 'tl-step' + (isTop ? ' is-detected ' + verdictClass(r.verdict) : '');
    if (isTop) li.setAttribute('aria-current', 'step');
    var dot = isTop && isFinite(conf) && isFinite(thr)
      ? (conf >= thr ? 'conf-high' : 'conf-low') : '';
    li.innerHTML =
      '<span class="tl-dot ' + dot + '" aria-hidden="true"></span>' +
      '<span class="tl-name">' + esc(name) + '</span>' +
      '<span class="tl-prob">' + (probs[name] !== undefined ? num(p, 2) : '—') + '</span>' +
      (isTop ? '<span class="tl-flag">detected · conf ' + esc(confTxt) + '</span>' : '');
    li.title = name + ': p = ' + (probs[name] !== undefined ? num(p) : '—') +
      (isTop ? ' (detected, confidence ' + confTxt + ')' : '');
    track.appendChild(li);
  });

  var card = el(
    '<section class="card" aria-label="Step timeline">' +
      '<div class="card-head"><h2 class="card-title">Step timeline</h2>' +
      '<span class="card-note">SOP steps · detected step marked, shaded by confidence</span></div>' +
      '<div class="card-body"><div class="tl-scroll"></div>' +
      '<p class="state-note">Hands p(yes) ' +
        esc(r.raw.decisions && r.raw.decisions.hands_ok && r.raw.decisions.hands_ok.p_yes !== undefined ? num(r.raw.decisions.hands_ok.p_yes) : '—') +
        ' · active-work p(yes) ' +
        esc(r.raw.decisions && r.raw.decisions.active_work && r.raw.decisions.active_work.p_yes !== undefined ? num(r.raw.decisions.active_work.p_yes) : '—') +
      '</p></div>' +
    '</section>');
  $('.tl-scroll', card).appendChild(track);
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
    b.addEventListener('click', function (ev) { ev.stopPropagation(); copyText(full, 'Hash copied.'); });
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

function signalsCard(sig) {
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
  var card = el(
    '<section class="card" aria-label="Exact signals (L0)">' +
      '<div class="card-head"><h2 class="card-title">Signals</h2><span class="card-note">exact, per-segment (L0)</span></div>' +
    '</section>');
  card.appendChild(grid);
  return card;
}

function copyText(text, okMsg) {
  function done() {
    var n = $('#voice-note');
    if (n) { n.textContent = okMsg; n.classList.remove('is-bad'); }
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text) && done(); });
  } else {
    fallbackCopy(text) && done();
  }
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
  var certified = d.chosen_threshold !== undefined && d.chosen_threshold !== null;
  row.innerHTML =
    stat('error bound', d.error_bound === undefined ? '—' : num(d.error_bound),
      d.delta !== undefined ? 'exact binomial, δ = ' + d.delta : 'exact binomial upper bound') +
    stat('target error', d.target_error === undefined ? '—' : num(d.target_error)) +
    stat('errors / n', (d.errors !== undefined && d.n !== undefined) ? d.errors + ' / ' + d.n : '—') +
    stat('chosen threshold', certified ? num(d.chosen_threshold) : 'not certified',
      certified
        ? (d.min_accepted !== undefined ? '≥ ' + d.min_accepted + ' accepted' : '')
        : 'no certified threshold yet');
  return row;
}

function riskLegend(d) {
  var html = '<div class="legend">' +
    '<span class="legend-item"><span class="legend-swatch"></span> error vs coverage</span>';
  if (d.error_bound !== undefined && isFinite(d.error_bound)) {
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

  var W = 720, H = 308, L = 46, R = 16, T = 20, B = 52;
  var iw = W - L - R, ih = H - T - B;
  var maxErr = 0;
  pts.forEach(function (p) { maxErr = Math.max(maxErr, p.error); });
  var hasBound = d.error_bound !== undefined && isFinite(d.error_bound);
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

  for (var i = 0; i <= 4; i++) {
    var c = i / 4;
    node('line', { x1: x(c), y1: T, x2: x(c), y2: T + ih, 'class': 'grid-line' });
    node('text', { x: x(c), y: T + ih + 18, 'text-anchor': 'middle' }, Math.round(c * 100) + '%');
  }
  for (var j = 0; j <= 4; j++) {
    var ev = (yMax * j) / 4;
    node('line', { x1: L, y1: y(ev), x2: L + iw, y2: y(ev), 'class': 'grid-line' });
    node('text', { x: L - 7, y: y(ev) + 3.5, 'text-anchor': 'end' }, ev.toFixed(3));
  }
  node('line', { x1: L, y1: T, x2: L, y2: T + ih, 'class': 'axis-line' });
  node('line', { x1: L, y1: T + ih, x2: L + iw, y2: T + ih, 'class': 'axis-line' });
  node('text', { x: L + iw / 2, y: T + ih + 40, 'text-anchor': 'middle', 'class': 'label-t' }, 'coverage');
  node('text', { x: L, y: T - 7, 'class': 'label-t' }, 'error');

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

  svg.setAttribute('aria-label', 'Risk/coverage curve with ' + pts.length + ' points. ' +
    (hasBound ? 'Error bound ' + num(d.error_bound) + '. ' : 'No certified error bound yet. ') +
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

var TITLES = { batches: 'Batches', segment: 'Segment', insights: 'Insights' };

function setRoute(name, id) {
  state.route = name;
  ['batches', 'segment', 'insights'].forEach(function (n) {
    var view = $('#view-' + n);
    var tab = $('#tab-' + n);
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
}

function routeFromHash() {
  var parsed = A.parseHash(location.hash);
  state.filters = parsed.filters;
  state.lang = parsed.lang;
  state.head = parsed.head || '';
  var sel = $('#f-head');
  if (sel && sel.querySelector('option[value="' + (state.head || '') + '"]')) sel.value = state.head;
  syncControlsFromState();
  state.segmentId = parsed.route === 'segment' ? parsed.id : null;
  setRoute(parsed.route, state.segmentId);
}

/* ============================ boot ============================ */

function bindStatic() {
  $$('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      var route = tab.getAttribute('data-route');
      if (route === state.route) return;
      if (route === 'segment' && state.segmentId) {
        location.hash = A.buildHash('segment', state.segmentId, null, null);
      } else if (route === 'segment') {
        location.hash = '#/segment';
      } else if (route === 'insights') {
        location.hash = A.buildHash('insights', null, null, state.head ? { head: state.head } : null);
      } else {
        location.hash = A.buildHash('batches', null, state.filters, null);
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
      if (t.classList && t.classList.contains('btn-open')) {
        openReceipt(t.getAttribute('data-batch'), t.getAttribute('data-segment'), t);
        return;
      }
    }
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
