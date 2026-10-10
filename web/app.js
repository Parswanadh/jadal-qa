/* Jadal QA dashboard — vanilla JS, no build step, no CDN.
 *
 * Three screens in one page: Batches (live list), Segment (receipt + re-verify),
 * Insights (risk/coverage + SOP gaps). Live against the service (docs/SPEC.md),
 * or against web/mock/*.json with ?mock=1.
 *
 * NOT DONE:
 * - No retry/backoff jitter tuning beyond a fixed schedule.
 * - Mock mode fabricates only latency/labels, never payloads; payloads in
 *   mock mode come straight from web/mock/*.json.
 */

(function () {
'use strict';

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

var LANGS = ['hi-IN', 'kn-IN', 'ta-IN', 'te-IN', 'en-IN'];
var MOCK_DELAY = 260; // ms: simulate a network round-trip so loading states are real
var PICO_POLL_MS = 5000;
var WS_RECONNECT = [1000, 2000, 4000, 8000];

var state = {
  route: 'batches',       // 'batches' | 'segment' | 'insights'
  segmentId: null,
  filter: 'all',
  expanded: {},           // batch_id -> true (show all segments)
  batches: null,          // normalized batches, or null while loading
  batchesError: null,
  receipt: null,
  receiptError: null,
  receiptId: null,
  rv: { status: 'idle', result: null, error: null }, // re-verify
  risk: null, riskError: null,
  gaps: null, gapsError: null,
  lang: 'hi-IN',
  voiceNote: null, voiceError: null,
  pico: null,
  live: false, lastEvent: null,
  mockMismatch: false,
  known: {},              // receipt_id/segment_id -> { batch_id }
};

function wsURL() {
  if (API) {
    return API.replace(/^http/, 'ws') + '/ws';
  }
  return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
}

/* ============================ formatting ============================ */

function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }

function num(v, d) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  var s = Number(v).toFixed(d === undefined ? 4 : d);
  return s;
}

function fmtLatency(ms) {
  if (ms === null || ms === undefined || isNaN(ms)) return '—';
  if (ms >= 1000) return (ms / 1000).toFixed(1) + ' s';
  return Math.round(ms) + ' ms';
}

function fmtTime(iso) {
  if (!iso) return '—';
  var date = new Date(iso);
  if (isNaN(date.getTime())) return String(iso);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit'
  }).format(date);
}

function fullTime(iso) {
  var date = new Date(String(iso || ''));
  return isNaN(date.getTime()) ? String(iso || '') : date.toLocaleString();
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

/* ============================ normalizers ============================
 * The service responses are tolerated liberally: accept the SPEC shapes,
 * plus camelCase / wrapped variants, because sibling modules are built by
 * other agents in parallel.
 */

function first(obj, keys) {
  for (var i = 0; i < keys.length; i++) {
    if (obj && obj[keys[i]] !== undefined && obj[keys[i]] !== null) return obj[keys[i]];
  }
  return undefined;
}

function normSegments(raw) {
  // Returns merged segment rows from a batch's {segment_ids, segments, verdicts}.
  var out = [];
  var verdicts = first(raw, ['verdicts']) || [];
  var vById = {};
  verdicts.forEach(function (v) {
    if (v && (v.segment_id || v.segmentId)) vById[v.segment_id || v.segmentId] = v;
  });
  var segs = first(raw, ['segments']);
  var knownIds = {};
  (segs || []).forEach(function (s) {
    var id = s.segment_id || s.segmentId || s.id;
    if (!id) return;
    knownIds[id] = true;
    var v = vById[id] || {};
    var lat = first(s, ['latency_ms', 'latencyMs', 'processing_ms', 'proc_ms', 'duration_ms']);
    var lats = first(s, ['latency_s', 'latencyS']);
    if (lat === undefined && lats !== undefined) lat = lats * 1000;
    out.push({
      segment_id: id,
      receipt_id: first(s, ['receipt_id', 'receiptId']) || id,
      worker_id: first(s, ['worker_id', 'workerId', 'worker']) || first(v, ['worker_id']) || '—',
      site_id: first(s, ['site_id', 'siteId', 'site']) || first(raw, ['site_id', 'siteId']) || '—',
      verdict: first(v, ['verdict']) || '—',
      reasons: first(v, ['reasons']) || [],
      conf: first(v, ['answer_conf', 'conf']),
      latency_ms: lat === undefined ? null : lat,
      start_s: first(s, ['start_s', 'startS']),
      end_s: first(s, ['end_s', 'endS']),
      source_clip: first(s, ['source_clip', 'clip']),
    });
  });
  // segments known only via segment_ids / verdicts
  var ids = first(raw, ['segment_ids', 'segmentIds']) || [];
  ids.concat(Object.keys(vById)).forEach(function (id) {
    if (knownIds[id]) return;
    knownIds[id] = true;
    var v = vById[id] || {};
    out.push({
      segment_id: id,
      receipt_id: id,
      worker_id: '—',
      site_id: first(raw, ['site_id', 'siteId']) || '—',
      verdict: first(v, ['verdict']) || '—',
      reasons: first(v, ['reasons']) || [],
      conf: first(v, ['answer_conf', 'conf']),
      latency_ms: null, start_s: null, end_s: null, source_clip: null,
    });
  });
  return out;
}

function normBatch(raw) {
  var decision = first(raw, ['batch_decision', 'batchDecision', 'decision']) || '—';
  var rows = normSegments(raw);
  var rejected = rows.filter(function (r) { return r.verdict === 'reject'; }).length;
  return {
    batch_id: first(raw, ['batch_id', 'batchId', 'id']) || '—',
    site_id: first(raw, ['site_id', 'siteId']) || '—',
    decision: decision,
    rule: first(raw, ['rule']) || null,
    k: first(raw, ['k']),
    n: first(raw, ['n']) || rows.length,
    rejected_count: first(raw, ['rejected_count', 'rejectedCount']) !== undefined
      ? first(raw, ['rejected_count', 'rejectedCount']) : rejected,
    created_utc: first(raw, ['created_utc', 'createdUtc', 'created']) || null,
    rows: rows,
  };
}

function normBatchList(raw) {
  var arr = Array.isArray(raw) ? raw : (first(raw, ['batches', 'items', 'data', 'results']) || []);
  if (!Array.isArray(arr)) arr = [];
  return arr.map(normBatch);
}

function normRisk(raw) {
  var pts = first(raw, ['points', 'curve', 'data']);
  if (Array.isArray(raw)) pts = raw;
  pts = (pts || []).map(function (p) {
    return {
      threshold: Number(first(p, ['threshold', 't'])),
      coverage: Number(first(p, ['coverage', 'c', 'cov'])),
      error: Number(first(p, ['error', 'err', 'e'])),
    };
  }).filter(function (p) { return isFinite(p.coverage) && isFinite(p.error); });
  return {
    points: pts,
    error_bound: first(raw, ['error_bound', 'errorBound', 'bound']),
    chosen_threshold: first(raw, ['chosen_threshold', 'threshold', 'operating_threshold']),
    n: first(raw, ['n', 'n_val', 'samples']),
    errors: first(raw, ['errors']),
    target_error: first(raw, ['target_error', 'targetError', 'target']),
    delta: first(raw, ['delta']) !== undefined ? first(raw, ['delta']) : 0.1,
    min_accepted: first(raw, ['min_accepted', 'minAccepted']),
  };
}

function normGaps(raw) {
  var arr = Array.isArray(raw) ? raw : (first(raw, ['gaps', 'bars', 'items', 'data']) || []);
  return {
    gaps: (arr || []).map(function (g) {
      return {
        code: first(g, ['reason_code', 'reasonCode', 'code', 'reason']) || '—',
        count: Number(first(g, ['count', 'n', 'value'])) || 0,
        section: first(g, ['sop_section', 'section', 'sop', 'label']) || '',
        share: first(g, ['share', 'fraction', 'frac']),
      };
    }),
    total: first(raw, ['total_segments', 'total', 'n']),
    window: first(raw, ['window_utc', 'window']),
  };
}

function normPico(raw) {
  raw = raw || {};
  var led = String(first(raw, ['led']) || 'OFF').toUpperCase();
  if (['G', 'A', 'R', 'OFF'].indexOf(led) === -1) {
    var m = { GREEN: 'G', AMBER: 'A', RED: 'R', GRAY: 'OFF', OFF: 'OFF', ON: 'G' };
    led = m[led] || 'OFF';
  }
  var hb = first(raw, ['last_heartbeat_utc', 'lastHeartbeatUtc', 'last_heartbeat', 'heartbeat']);
  var ageMs = null;
  if (hb) {
    var t = new Date(hb).getTime();
    if (!isNaN(t)) ageMs = Date.now() - t;
  } else if (raw.heartbeat_age_s !== undefined || raw.heartbeatAgeS !== undefined) {
    ageMs = Number(first(raw, ['heartbeat_age_s', 'heartbeatAgeS'])) * 1000;
  }
  return {
    connected: first(raw, ['connected']) !== undefined ? !!first(raw, ['connected']) : (led !== 'OFF'),
    led: led,
    label: first(raw, ['led_meaning', 'label', 'meaning']) || null,
    ageMs: ageMs,
    ticks: first(raw, ['ticks_ms', 'ticks']),
    firmware: first(raw, ['firmware']),
    note: first(raw, ['_note']) || null,
  };
}

/* ============================ transport ============================ */

async function liveGet(path) {
  var ctl = new AbortController();
  var timer = setTimeout(function () { ctl.abort(); }, 12000);
  try {
    var res = await fetch(API + path, { headers: { Accept: 'application/json' }, signal: ctl.signal });
    if (!res.ok) throw new Error('GET ' + path + ' → ' + res.status + ' ' + res.statusText);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function livePost(path, body) {
  var ctl = new AbortController();
  var timer = setTimeout(function () { ctl.abort(); }, 60000);
  try {
    var res = await fetch(API + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body || {}),
      signal: ctl.signal,
    });
    if (!res.ok) throw new Error('POST ' + path + ' → ' + res.status + ' ' + res.statusText);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/* ---- mock transport: real payloads from web/mock/*.json ---- */
var mockCache = {};
async function mockFile(name) {
  if (mockCache[name]) return mockCache[name];
  await new Promise(function (r) { setTimeout(r, MOCK_DELAY); });
  var res = await fetch('mock/' + name + '.json', { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error('mock/' + name + '.json → ' + res.status);
  var data = await res.json();
  mockCache[name] = data;
  return data;
}

var mockApi = {
  async batches() { return mockFile('batches'); },
  async batch(id) {
    var all = await mockFile('batches');
    var b = all.filter(function (x) {
      return (x.batch_id || x.batchId || x.id) === id;
    })[0];
    if (!b) throw new Error('mock: batch ' + id + ' not in fixture');
    return b;
  },
  async receipt(id) {
    await new Promise(function (r) { setTimeout(r, MOCK_DELAY); });
    var all = await mockFile('receipts');
    var keys = Object.keys(all).filter(function (k) { return k.charAt(0) !== '_'; });
    if (all[id]) return { receipt: all[id], id: id, fallback: false };
    // unknown id → first fixture, labelled, never silent
    return { receipt: all[keys[0]], id: keys[0], fallback: id };
  },
  async reverify(id, expected) {
    await new Promise(function (r) { setTimeout(r, 900); }); // re-verify is slow: it re-runs the pipeline
    if (state.mockMismatch) {
      return { match: false, expected: expected, actual: 'm' + expected.slice(1), diff: ['decisions.step.answer_conf', '_mock.forced_mismatch'] };
    }
    return { match: true, expected: expected, actual: expected, diff: [] };
  },
  async risk() { return mockFile('risk_coverage'); },
  async gaps() { return mockFile('sop_gaps'); },
  async pico() {
    var p = await mockFile('pico_status');
    var out = Object.assign({}, p);
    // a live heartbeat: last beat ~1s before each poll
    out.last_heartbeat_utc = new Date(Date.now() - 900).toISOString();
    // reflect the newest loaded batch so every LED state is reachable
    if (state.batches && state.batches.length) {
      var d = state.batches[0].decision;
      out.led = d === 'accept' ? 'G' : (d === 'reject' ? 'R' : 'A');
      out.led_meaning = d === 'accept' ? 'batch accepted' : (d === 'reject' ? 'batch rejected' : 'review pending');
    }
    return out;
  },
};

var api = MOCK ? null : { get: liveGet, post: livePost };

/* ============================ data loading ============================ */

function indexKnown(batches) {
  state.known = {};
  batches.forEach(function (b) {
    b.rows.forEach(function (r) {
      state.known[r.receipt_id] = { batch_id: b.batch_id, segment_id: r.segment_id };
      state.known[r.segment_id] = { batch_id: b.batch_id, segment_id: r.segment_id };
    });
  });
}

async function loadBatches(soft) {
  if (!soft) {
    state.batches = null;
    state.batchesError = null;
    renderBatches();
  }
  try {
    var raw = MOCK ? await mockApi.batches() : await api.get('/batches');
    state.batches = normBatchList(raw);
    state.batches.sort(function (a, b) {
      return String(b.created_utc || '') < String(a.created_utc || '') ? -1 : 1;
    });
    state.batchesError = null;
    indexKnown(state.batches);
  } catch (e) {
    if (!soft) { state.batches = null; }
    state.batchesError = e;
  }
  renderBatches();
}

async function loadReceipt(id) {
  state.receipt = null;
  state.receiptError = null;
  state.receiptId = id;
  state.rv = { status: 'idle', result: null, error: null };
  renderSegment();
  try {
    if (MOCK) {
      var m = await mockApi.receipt(id);
      state.receipt = m.receipt;
      state.receiptId = m.id;
      state.receiptFallback = m.fallback || null;
    } else {
      state.receipt = await api.get('/receipts/' + encodeURIComponent(id));
      state.receiptFallback = null;
    }
  } catch (e) {
    state.receipt = null;
    state.receiptError = e;
  }
  renderSegment();
}

async function runReverify() {
  if (!state.receipt) return;
  state.rv = { status: 'busy', result: null, error: null };
  renderSegment(); // shows the "re-running" line; keeps receipt visible
  var id = state.receiptId;
  try {
    var expected = state.receipt.receipt_sha256 || state.receipt.receiptSha256;
    var res = MOCK
      ? await mockApi.reverify(id, expected)
      : await api.post('/receipts/' + encodeURIComponent(id) + '/reverify', {});
    state.rv = { status: 'done', result: res, error: null };
  } catch (e) {
    state.rv = { status: 'done', result: null, error: e };
  }
  renderSegment();
}

async function loadInsights() {
  if (state.risk === null && state.riskError === null) renderInsights();
  try {
    var raw = MOCK ? await mockApi.risk() : await api.get('/stats/risk_coverage');
    state.risk = normRisk(raw);
    state.riskError = null;
  } catch (e) {
    state.risk = null;
    state.riskError = e;
  }
  try {
    var g = MOCK ? await mockApi.gaps() : await api.get('/stats/sop_gaps');
    state.gaps = normGaps(g);
    state.gapsError = null;
  } catch (e) {
    state.gaps = null;
    state.gapsError = e;
  }
  renderInsights();
}

async function loadPico() {
  try {
    var raw = MOCK ? await mockApi.pico() : await api.get('/pico/status');
    state.pico = normPico(raw);
  } catch (e) {
    state.pico = { connected: false, led: 'OFF', label: null, ageMs: null, error: e };
  }
  renderPico();
}

/* ============================ batches screen ============================ */

function skeletonCards(n) {
  var html = '';
  for (var i = 0; i < n; i++) {
    html += '<div class="skel skel-card" aria-hidden="true"></div>';
  }
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

  var list = state.batches.filter(function (b) {
    return state.filter === 'all' || b.decision === state.filter;
  });
  var segTotal = list.reduce(function (n, b) { return n + b.rows.length; }, 0);

  meta.hidden = false;
  meta.textContent = list.length + (list.length === 1 ? ' batch' : ' batches') +
    ' · ' + segTotal + ' segments' +
    (state.batchesError ? ' · refresh failed, showing cached list' : '') +
    (MOCK ? ' · mock' : ' · live');

  $('#batches-sub').textContent = MOCK
    ? 'Fixture list from web/mock/batches.json. Switch off ?mock=1 for the live service.'
    : 'Every batch decision with its per-segment verdicts, newest first.';

  if (!list.length) {
    body.appendChild(el(
      '<div class="state" role="status">' +
        '<p class="state-title">No batches yet</p>' +
        '<p class="state-note">Nothing has been ingested' +
        (state.filter !== 'all' ? ' matching this filter' : '') +
        '. Batches appear here as soon as the service segments clips.</p>' +
      '</div>'));
    body.setAttribute('aria-busy', 'false');
    return;
  }

  list.forEach(function (b) {
    body.appendChild(batchCard(b));
  });
  body.setAttribute('aria-busy', 'false');
}

function batchCard(b) {
  var expanded = !!state.expanded[b.batch_id];
  var shown = expanded ? b.rows : b.rows.slice(0, 5);
  var card = el('<article class="card" aria-label="Batch ' + esc(b.batch_id) + '"></article>');

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

  var list = el('<div class="seglist"></div>');
  if (!shown.length) {
    list.appendChild(el('<p class="state-note" style="padding: var(--s3) var(--s4);">This batch has no segments.</p>'));
  }
  shown.forEach(function (r) {
    var reasons = (r.reasons && r.reasons.length)
      ? r.reasons.map(function (c) { return '<span class="reason">' + esc(c) + '</span>'; }).join('')
      : '<span class="reason-none">no flags</span>';
    list.appendChild(el(
      '<div class="seg-row">' +
        '<div class="seg-main">' +
          '<a class="seg-link" href="#/segment/' + encodeURIComponent(r.receipt_id) + '"' +
            ' title="Open receipt ' + esc(r.receipt_id) + '">' + esc(r.segment_id) + '</a>' +
          verdictChip(r.verdict) +
        '</div>' +
        '<div class="seg-reasons">' + reasons + '</div>' +
        '<div class="seg-meta">' +
          '<span>' + esc(r.worker_id) + '</span>' +
          '<span>' + esc(r.site_id) + '</span>' +
          '<span title="per-segment pipeline latency">' + esc(fmtLatency(r.latency_ms)) + '</span>' +
        '</div>' +
      '</div>'));
  });
  card.appendChild(list);

  if (b.rows.length > 5) {
    var more = el(
      '<div class="seg-more">' +
        '<button class="btn" type="button" data-expand="' + esc(b.batch_id) + '">' +
          (expanded ? 'Show less' : 'Show all ' + b.rows.length + ' segments') +
        '</button>' +
      '</div>');
    card.appendChild(more);
  }
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

  // opener for a raw receipt id (useful when a batch has unlinked segments)
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
    if (v) location.hash = '#/segment/' + encodeURIComponent(v);
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
  var v = r.verdict || {};
  var sig = r.signals || {};
  var dec = r.decisions || {};
  var reasons = v.reasons || [];

  var known = state.known[state.receiptId] || {};
  var segLabel = v.segment_id || known.segment_id || state.receiptId;

  $('#segment-title').textContent = 'Segment';
  $('#segment-sub').textContent = 'Receipt for ' + segLabel +
    (known.batch_id ? ' · from batch ' + known.batch_id : '');

  // ---- head ----
  var head = el(
    '<div class="seg-head">' +
      verdictChip(v.verdict) +
      '<span class="seg-head-id">' + esc(segLabel) + '</span>' +
    '</div>');
  body.appendChild(head);

  if (reasons.length) {
    var rc = el('<div class="seg-reasons" style="margin-bottom:var(--s4);"></div>');
    reasons.forEach(function (c) {
      rc.appendChild(el('<span class="reason">' + esc(c) + '</span>'));
    });
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

  // ---- meta + reverify/voice grid ----
  var grid = el('<div class="panel-grid panel-grid-2"></div>');

  var conf = Number(v.answer_conf);
  var metaCard = el(
    '<section class="card" aria-label="Segment facts">' +
      '<div class="card-head"><h2 class="card-title">Segment</h2>' +
      '<span class="card-note">' + esc(fullTime(r.created_utc)) + '</span></div>' +
      '<div class="card-body"><div class="kv">' +
        segHeadRow('snapshot', 'answer_conf ' + (isFinite(conf) ? num(conf) : '—') + ' · threshold ' + esc(first(v, ['threshold']) !== undefined ? first(v, ['threshold']) : '—')) +
        segHeadRow('source clip', first(sig, ['source_clip']) || first(r, ['source_clip']) || '—') +
        segHeadRow('site · worker', (function () {
          var w = known.segment_id && state.batches ? workerFor(known.segment_id) : null;
          return w ? w : 'see batch list';
        })()) +
      '</div></div>' +
    '</section>');

  var rvCard = el(
    '<section class="card" aria-label="Re-verify receipt">' +
      '<div class="card-head"><h2 class="card-title">Re-verify</h2>' +
      '<span class="card-note">re-runs the pipeline, compares hashes</span></div>' +
      '<div class="card-body">' +
        '<div class="rv-head">' +
          '<button class="btn btn-primary" type="button" id="rv-run"' +
            (state.rv.status === 'busy' ? ' disabled' : '') + '>Re-verify this receipt</button>' +
          (MOCK
            ? '<label class="card-note" style="display:flex;gap:var(--s1);align-items:center;">' +
              '<input type="checkbox" id="mock-mismatch"' + (state.mockMismatch ? ' checked' : '') + '> mock: force mismatch</label>'
            : '') +
        '</div>' +
        '<div id="rv-out"></div>' +
      '</div>' +
    '</section>');

  grid.appendChild(metaCard);
  grid.appendChild(rvCard);
  body.appendChild(grid);

  renderReverify($('#rv-out'));
  $('#rv-run').addEventListener('click', runReverify);
  var mm = $('#mock-mismatch');
  if (mm) mm.addEventListener('change', function () { state.mockMismatch = mm.checked; });

  // ---- voice ----
  var voiceCard = el(
    '<section class="card" aria-label="Voice message">' +
      '<div class="card-head"><h2 class="card-title">Voice message</h2>' +
      '<span class="card-note">' + (MOCK ? 'mock: placeholder tone, real audio comes from the service' : 'GET /voice/{code}/{lang}') + '</span></div>' +
      '<div class="card-body">' +
        '<div class="voice-row">' +
          '<div class="field"><span class="field-label" id="voice-code-label">Reason code</span>' +
            '<span class="mono" id="voice-code" style="font-size:.8125rem;">' + esc(reasons[0] || '—') + '</span></div>' +
          '<div class="field"><span class="field-label" id="voice-lang-label">Language</span>' +
            '<div class="langrow" role="group" aria-labelledby="voice-lang-label" id="lang-row"></div></div>' +
          '<button class="btn" type="button" id="voice-play"' + (reasons.length ? '' : ' disabled') + '>▶ Play</button>' +
        '</div>' +
        '<p class="audio-note" id="voice-note" role="status">' +
          (reasons.length ? 'Reads the supervisor message for ' + esc(reasons[0]) + '.' : 'No reason codes on this segment, so there is no voice message.') +
        '</p>' +
      '</div>' +
    '</section>');
  body.appendChild(voiceCard);

  var lr = $('#lang-row');
  LANGS.forEach(function (l) {
    var b = el('<button class="langbtn' + (l === state.lang ? ' is-on' : '') + '" type="button" data-lang="' + l + '"' +
      (l === state.lang ? ' aria-pressed="true"' : ' aria-pressed="false"') + '>' + l + '</button>');
    lr.appendChild(b);
  });
  $('#voice-play').addEventListener('click', function () { playVoice(reasons[0]); });

  // ---- hashes ----
  var hashCard = el(
    '<section class="card" aria-label="Hashes">' +
      '<div class="card-head"><h2 class="card-title">Hashes</h2>' +
      '<span class="card-note">SHA-256, per docs/SPEC.md</span></div>' +
      '<div class="card-body">' +
        '<div class="hash-row"><span class="hash-key">receipt id</span><span class="hash-val">' + esc(r.receipt_id || '—') + '</span></div>' +
        '<div class="hash-row"><span class="hash-key">receipt sha256</span><span class="hash-val">' + esc(r.receipt_sha256 || '—') + '</span></div>' +
        '<div class="hash-row"><span class="hash-key">segment sha256</span><span class="hash-val">' + esc(r.segment_sha256 || '—') + '</span></div>' +
        (sig.blurred_sha256
          ? '<div class="hash-row"><span class="hash-key">blurred file sha256</span><span class="hash-val">' + esc(sig.blurred_sha256) + '</span></div>'
          : '') +
      '</div>' +
    '</section>');
  body.appendChild(hashCard);

  // ---- signals ----
  body.appendChild(signalsCard(sig));

  // ---- decisions ----
  body.appendChild(decisionsCard(dec));

  // ---- raw receipt ----
  var rawCard = el(
    '<section class="card" aria-label="Receipt JSON">' +
      '<div class="card-head"><h2 class="card-title">Receipt JSON</h2>' +
      '<button class="btn" type="button" id="copy-receipt">Copy</button></div>' +
      '<div class="card-body"><pre class="pre" id="receipt-pre" tabindex="0"></pre></div>' +
    '</section>');
  body.appendChild(rawCard);
  $('#receipt-pre').textContent = JSON.stringify(r, null, 2);
  $('#copy-receipt').addEventListener('click', function () {
    copyText(JSON.stringify(r, null, 2), 'Receipt JSON copied.');
  });
}

function workerFor(segmentId) {
  if (!state.batches) return null;
  for (var i = 0; i < state.batches.length; i++) {
    var rows = state.batches[i].rows;
    for (var j = 0; j < rows.length; j++) {
      if (rows[j].segment_id === segmentId) return rows[j].worker_id + ' · ' + rows[j].site_id;
    }
  }
  return null;
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
      '<p class="rv-verdict ' + (match ? 'is-match' : 'is-mismatch') + '">' +
        (match ? 'MATCH — receipt reproduces' : 'MISMATCH — receipt does not reproduce') + '</p>' +
      '<p class="rv-detail">expected <span class="mono">' + esc(res.expected || '—') + '</span><br>' +
        'actual&nbsp;&nbsp;&nbsp; <span class="mono">' + esc(res.actual || '—') + '</span></p>' +
      (((res.diff || []).length)
        ? '<ul class="rv-diff">' + res.diff.map(function (d) { return '<li>' + esc(d) + '</li>'; }).join('') + '</ul>'
        : '<p class="rv-detail">no differing fields</p>') +
    '</div>');
  out.appendChild(banner);
}

function signalsCard(sig) {
  function stat(label, value, unit) {
    return '<div class="stat"><div class="stat-label">' + esc(label) + '</div>' +
      '<div class="stat-value">' + esc(value) + (unit ? '<span class="stat-unit">' + esc(unit) + '</span>' : '') + '</div></div>';
  }
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
  var extra = el('<div class="card-body"><div class="kv">' +
    '<div class="kv-row"><span class="kv-key">phash</span><span class="kv-val">' + esc(sig.phash || '—') + '</span></div>' +
    '<div class="kv-row"><span class="kv-key">duplicate of</span><span class="kv-val">' + esc(sig.dup_of || '—') + '</span></div>' +
    '<div class="kv-row"><span class="kv-key">over-exposed frac</span><span class="kv-val">' + esc(sig.over_exposed_frac === undefined ? '—' : num(sig.over_exposed_frac)) + '</span></div>' +
  '</div></div>');
  card.appendChild(extra);
  return card;
}

function probBars(title, probs, topChoice) {
  var keys = Object.keys(probs || {});
  if (!keys.length) return '<p class="state-note">No ' + esc(title) + ' scores.</p>';
  var max = Math.max.apply(null, keys.map(function (k) { return Number(probs[k]) || 0; }));
  var html = '<div class="bars" role="list" aria-label="' + esc(title) + '">';
  keys.sort(function (a, b) { return (Number(probs[b]) || 0) - (Number(probs[a]) || 0); }).forEach(function (k) {
    var p = Number(probs[k]) || 0;
    var pct = max > 0 ? (p / max) * 100 : 0;
    html += '<div class="bar-row" role="listitem">' +
      '<span class="bar-label" title="' + esc(k) + '">' + esc(k) + '</span>' +
      '<span class="bar-track"><span class="bar-fill' + (k === topChoice ? ' is-top' : '') +
        '" style="width:' + pct.toFixed(1) + '%"></span></span>' +
      '<span class="bar-value">' + num(p) + '</span></div>';
  });
  return html + '</div>';
}

function decisionsCard(dec) {
  var step = dec.step || {};
  var card = el(
    '<section class="card" aria-label="Decisions (L1)">' +
      '<div class="card-head"><h2 class="card-title">Decisions</h2><span class="card-note">typed decision models (L1)</span></div>' +
      '<div class="card-body">' +
        '<div class="kv" style="margin-bottom:var(--s4);">' +
          '<div class="kv-row"><span class="kv-key">hands ok — p(yes)</span><span class="kv-val">' + esc(dec.hands_ok && dec.hands_ok.p_yes !== undefined ? num(dec.hands_ok.p_yes) : '—') + '</span></div>' +
          '<div class="kv-row"><span class="kv-key">active work — p(yes)</span><span class="kv-val">' + esc(dec.active_work && dec.active_work.p_yes !== undefined ? num(dec.active_work.p_yes) : '—') + '</span></div>' +
          '<div class="kv-row"><span class="kv-key">step choice</span><span class="kv-val">' + esc(step.choice || '—') + '</span></div>' +
          '<div class="kv-row"><span class="kv-key">answer confidence</span><span class="kv-val">' + esc(step.answer_conf === undefined ? '—' : num(step.answer_conf)) + '</span></div>' +
        '</div>' +
        probBars('step probabilities', step.probs, step.choice) +
        '<div class="kv" style="margin-top:var(--s4);">' +
          Object.keys(dec.model_versions || {}).map(function (m) {
            return '<div class="kv-row"><span class="kv-key">model · ' + esc(m) + '</span><span class="kv-val">' + esc(dec.model_versions[m]) + '</span></div>';
          }).join('') +
        '</div>' +
      '</div>' +
    '</section>');
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
  var url = API + '/voice/' + encodeURIComponent(code) + '/' + encodeURIComponent(state.lang);
  fetch(url, { headers: { Accept: 'audio/*' } }).then(function (res) {
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
    var codes = ['UNDECODABLE', 'HANDS_NOT_VISIBLE', 'BLUR', 'TOO_DARK', 'OVEREXPOSED', 'SHAKY', 'DUPLICATE', 'WRONG_STEP', 'NO_ACTIVE_WORK', 'LOW_CONFIDENCE'];
    var idx = Math.max(0, codes.indexOf(code));
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

function renderInsights() {
  var body = $('#insights-body');
  body.innerHTML = '';
  body.setAttribute('aria-busy', 'true');

  $('#insights-mock-pill').hidden = !MOCK;

  // ---- risk / coverage ----
  var riskSec = el(
    '<section class="card" aria-label="Risk coverage curve">' +
      '<div class="card-head"><h2 class="card-title">Risk / coverage</h2>' +
      '<span class="card-note">GET /stats/risk_coverage</span></div>' +
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
    rOut.appendChild(riskLegend());
  }

  // ---- SOP gaps ----
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
  row.innerHTML =
    stat('error bound', d.error_bound === undefined ? '—' : num(d.error_bound), 'exact binomial, δ = ' + (d.delta !== undefined ? d.delta : 0.1)) +
    stat('target error', d.target_error === undefined ? '—' : num(d.target_error)) +
    stat('errors / n', (d.errors !== undefined && d.n !== undefined) ? d.errors + ' / ' + d.n : '—') +
    stat('chosen threshold', d.chosen_threshold === undefined ? '—' : num(d.chosen_threshold),
      d.min_accepted !== undefined ? '≥ ' + d.min_accepted + ' accepted' : '');
  return row;
}

function riskLegend() {
  return el(
    '<div class="legend">' +
      '<span class="legend-item"><span class="legend-swatch"></span> error vs coverage</span>' +
      '<span class="legend-item"><span class="legend-swatch reject dashed"></span> error bound (exact binomial)</span>' +
      '<span class="legend-item"><span class="legend-swatch dashed"></span> target error</span>' +
    '</div>');
}

function riskChart(d) {
  var NS = 'http://www.w3.org/2000/svg';
  var pts = (d.points || []).slice().sort(function (a, b) { return a.coverage - b.coverage; });
  if (!pts.length) return el('<p class="state-note">No risk/coverage points were returned.</p>');

  var W = 720, H = 300, L = 46, R = 16, T = 20, B = 34;
  var iw = W - L - R, ih = H - T - B;
  var maxErr = 0;
  pts.forEach(function (p) { maxErr = Math.max(maxErr, p.error); });
  if (d.error_bound !== undefined) maxErr = Math.max(maxErr, d.error_bound);
  if (d.target_error !== undefined) maxErr = Math.max(maxErr, d.target_error);
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

  // grid + axes
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
  node('text', { x: L + iw / 2, y: T + ih + 18, 'text-anchor': 'middle', 'class': 'label-t' }, 'coverage');
  node('text', { x: L, y: T - 7, 'class': 'label-t' }, 'error');

  // area + curve
  var dAttr = pts.map(function (p, k) { return (k ? 'L' : 'M') + x(p.coverage).toFixed(1) + ' ' + y(p.error).toFixed(1); }).join(' ');
  node('path', { d: dAttr + ' L' + x(pts[pts.length - 1].coverage).toFixed(1) + ' ' + (T + ih) + ' L' + x(pts[0].coverage).toFixed(1) + ' ' + (T + ih) + ' Z', 'class': 'curve-area' });
  node('path', { d: dAttr, 'class': 'curve' });

  // bound + target lines
  if (d.error_bound !== undefined && isFinite(d.error_bound)) {
    var yb = y(d.error_bound);
    node('line', { x1: L, y1: yb, x2: L + iw, y2: yb, 'class': 'bound' });
    node('text', { x: L + iw - 4, y: yb - 6, 'text-anchor': 'end', 'class': 'label-t' },
      'bound ' + num(d.error_bound));
  }
  if (d.target_error !== undefined && isFinite(d.target_error)) {
    var yt = y(d.target_error);
    node('line', { x1: L, y1: yt, x2: L + iw, y2: yt, 'class': 'target' });
    node('text', { x: L + 4, y: yt - 6, 'class': 'label-t' }, 'target ' + num(d.target_error));
  }

  // operating point: highest coverage whose error stays within both lines
  var budget = Math.min(
    d.target_error === undefined ? Infinity : d.target_error,
    d.error_bound === undefined ? Infinity : d.error_bound);
  var pick = null;
  pts.forEach(function (p) {
    if (p.error <= budget && (!pick || p.coverage > pick.coverage)) pick = p;
  });
  pts.forEach(function (p) {
    node('circle', { cx: x(p.coverage), cy: y(p.error), r: p === pick ? 5 : 2.5, 'class': 'dot' + (p === pick ? ' is-pick' : '') });
  });
  if (pick) {
    node('text', { x: clamp(x(pick.coverage) + 10, L + 4, L + iw - 120), y: clamp(y(pick.error) - 8, T + 8, T + ih), 'class': 'label-t' },
      'operating θ=' + num(pick.threshold, 2));
  }

  // accessible description
  var desc = 'Risk/coverage curve with ' + pts.length + ' points. ' +
    (d.error_bound !== undefined ? 'Error bound ' + num(d.error_bound) + '. ' : '') +
    (d.target_error !== undefined ? 'Target error ' + num(d.target_error) + '.' : '');
  svg.setAttribute('aria-label', desc);
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

/* ============================ pico strip ============================ */

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
    line.textContent = 'offline' + (p.error ? ' — ' + (p.error.message || 'unreachable') : ' — no serial device');
    return;
  }
  var word = { G: 'green', A: 'amber', R: 'red', OFF: 'off' }[p.led] || 'off';
  var bits = ['pico ' + word];
  if (p.label) bits.push(p.label);
  bits.push(p.ageMs === null || p.ageMs === undefined ? 'heartbeat age unknown' : 'beat ' + fmtBeatAge(p.ageMs));
  line.textContent = bits.join(' · ') + (MOCK ? ' · mock' : '');
}

function fmtBeatAge(ms) {
  if (ms < 1500) return 'just now';
  if (ms < 60000) return Math.round(ms / 1000) + ' s ago';
  return Math.round(ms / 60000) + ' min ago';
}

/* ============================ live channel ============================ */

function onPushEvent(data) {
  state.lastEvent = Date.now();
  var live = $('#live-pill');
  live.hidden = false;
  live.textContent = 'live · event ' + ageText(state.lastEvent);
  // tolerate any shape: full batch → merge, anything else → soft refresh
  try {
    if (data && (data.batch_id || data.batchId || data.id) && (data.verdicts || data.rows || data.segments)) {
      var nb = normBatch(data);
      var found = false;
      state.batches = (state.batches || []).map(function (b) {
        if (b.batch_id === nb.batch_id) { found = true; return nb; }
        return b;
      });
      if (!found) state.batches.unshift(nb);
      state.batches.sort(function (a, b) {
        return String(b.created_utc || '') < String(a.created_utc || '') ? -1 : 1;
      });
      indexKnown(state.batches);
      if (state.route === 'batches') renderBatches();
    } else if (state.route === 'batches') {
      loadBatches(true);
    }
  } catch (e) { /* a bad push must never break the list */ }
}

function connectWS() {
  if (MOCK) { return; } // mock mode: no socket; source is labelled mock
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
  if (state.pico && state.pico.connected) renderPico(); // refresh the heartbeat age
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

  if (name === 'batches' && (state.batches === null || state.batchesError)) loadBatches(false);
  if (name === 'segment') {
    if (!id) renderSegment(); // "no segment selected" state
    else if (id !== state.receiptId || (!state.receipt && !state.receiptError)) loadReceipt(id);
    else renderSegment();
  }
  if (name === 'insights' && (state.risk === null || state.gaps === null)) loadInsights();
}

function routeFromHash() {
  var h = location.hash || '#/batches';
  var m = h.match(/^#\/segment(?:\/(.+))?$/);
  if (m) {
    state.segmentId = m[1] ? decodeURIComponent(m[1]) : null;
    setRoute('segment', state.segmentId);
    return;
  }
  if (h === '#/insights') { state.segmentId = null; setRoute('insights'); return; }
  state.segmentId = null;
  setRoute('batches');
}

/* ============================ boot ============================ */

function bindStatic() {
  $$('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      location.hash = '#/' + tab.getAttribute('data-route') +
        (tab.getAttribute('data-route') === 'segment' && state.segmentId
          ? '/' + encodeURIComponent(state.segmentId) : '');
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

  $$('#batches-filters .chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      $$('#batches-filters .chip').forEach(function (c) { c.classList.remove('is-on'); });
      chip.classList.add('is-on');
      state.filter = chip.getAttribute('data-filter');
      if (state.batches) renderBatches();
    });
  });

  $('#batches-refresh').addEventListener('click', function () { loadBatches(false); });
  $('#insights-refresh').addEventListener('click', function () {
    state.risk = null; state.riskError = null;
    state.gaps = null; state.gapsError = null;
    loadInsights();
  });

  // delegated clicks: retry / expand / language
  document.addEventListener('click', function (ev) {
    var t = ev.target;
    if (t && t.getAttribute) {
      var retry = t.getAttribute('data-retry');
      if (retry === 'batches') { loadBatches(false); return; }
      if (retry === 'segment') { loadReceipt(state.segmentId); return; }
      if (retry === 'insights') { loadInsights(); return; }
      var exp = t.getAttribute('data-expand');
      if (exp) {
        state.expanded[exp] = !state.expanded[exp];
        renderBatches();
        return;
      }
      var lang = t.getAttribute('data-lang');
      if (lang) {
        state.lang = lang;
        $$('#lang-row .langbtn').forEach(function (b) {
          var on = b.getAttribute('data-lang') === lang;
          b.classList.toggle('is-on', on);
          b.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
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
  if (state.route === 'segment') loadReceipt(state.segmentId);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

})();
