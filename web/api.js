/* Jadal QA dashboard — API adapter (web/api.js).
 *
 * The ONLY module that touches service/mock field names. app.js must read
 * every payload through this module so switching from web/mock/*.json to the
 * live service shapes (docs/SPEC.md, docs/API.md when the integration branch
 * publishes it) is a one-file change.
 *
 * Works in the browser (global `JadalApi`) and under node (module.exports)
 * so web/test_dashboard.py can exercise the pure functions.
 *
 * NOT DONE:
 * - Per-head risk/coverage curves: the service returns one aggregate curve;
 *   head selection is a label-only switch until the API exposes per-head data.
 * - Per-segment worker/clip/latency: not exposed by the API; the worker
 *   filter populates opportunistically and renders "—" when absent.
 */
(function (root, factory) {
'use strict';
if (typeof module !== 'undefined' && module.exports) {
  module.exports = factory();
} else {
  root.JadalApi = factory();
}
}(typeof self !== 'undefined' ? self : this, function () {
'use strict';

var LANGS = ['hi-IN', 'kn-IN', 'ta-IN', 'te-IN', 'en-IN'];
var REASON_CODES = ['UNDECODABLE', 'HANDS_NOT_VISIBLE', 'BLUR', 'TOO_DARK',
  'OVEREXPOSED', 'SHAKY', 'DUPLICATE', 'WRONG_STEP', 'NO_ACTIVE_WORK', 'LOW_CONFIDENCE'];
var VERDICTS = ['accept', 'reject', 'review'];
var BATCH_DECISIONS = ['accept', 'reject'];
var MOCK_DELAY = 260;

function first(obj, keys) {
  for (var i = 0; i < keys.length; i++) {
    if (obj && obj[keys[i]] !== undefined && obj[keys[i]] !== null) return obj[keys[i]];
  }
  return undefined;
}

function asArray(v) { return Array.isArray(v) ? v : []; }

/* ============================ normalizers ============================ */

function normSegments(raw) {
  var verdicts = asArray(first(raw, ['verdicts']));
  var vById = {};
  verdicts.forEach(function (v) {
    if (v && (v.segment_id || v.segmentId)) vById[v.segment_id || v.segmentId] = v;
  });
  var ids = asArray(first(raw, ['segment_ids', 'segmentIds'])).slice();
  Object.keys(vById).forEach(function (id) {
    if (ids.indexOf(id) === -1) ids.push(id);
  });
  var site = first(raw, ['site_id', 'siteId']) || '—';
  return ids.map(function (id) {
    var v = vById[id] || {};
    return {
      segment_id: id,
      site_id: site,
      worker_id: first(v, ['worker_id', 'workerId']) || null,
      verdict: first(v, ['verdict']) || '—',
      reasons: asArray(first(v, ['reasons'])),
      conf: first(v, ['answer_conf', 'conf']),
      threshold: first(v, ['threshold']),
    };
  });
}

function normBatch(raw) {
  var rows = normSegments(raw);
  var rejected = rows.filter(function (r) { return r.verdict === 'reject'; }).length;
  return {
    batch_id: first(raw, ['batch_id', 'batchId', 'id']) || '—',
    site_id: first(raw, ['site_id', 'siteId']) || '—',
    decision: first(raw, ['batch_decision', 'batchDecision', 'decision']) || '—',
    rule: first(raw, ['rule']) || null,
    k: first(raw, ['k']),
    n: first(raw, ['n']) !== undefined ? first(raw, ['n']) : rows.length,
    rejected_count: rejected,
    created_utc: first(raw, ['created_utc', 'createdUtc', 'created']) || null,
    rows: rows,
  };
}

function normBatchList(raw) {
  var arr = Array.isArray(raw) ? raw : asArray(first(raw, ['batches', 'items', 'data', 'results']));
  return arr.map(normBatch);
}

function normReceipt(raw) {
  raw = raw || {};
  var v = raw.verdict || {};
  var dec = raw.decisions || {};
  var step = dec.step || {};
  var cfg = raw.config || {};
  var flat = [];
  Object.keys(step.probs || {}).forEach(function (k) { flat.push(k); });
  flat.sort();
  return {
    receipt_id: first(raw, ['receipt_id', 'receiptId']) || null,
    receipt_sha256: first(raw, ['receipt_sha256', 'receiptSha256']) || null,
    segment_sha256: first(raw, ['segment_sha256', 'segmentSha256']) || null,
    blurred_sha256: first(raw, ['blurred_sha256', 'blurredSha256']) || null,
    created_utc: first(raw, ['created_utc', 'createdUtc']) || null,
    segment_id: first(v, ['segment_id', 'segmentId']) || null,
    verdict: first(v, ['verdict']) || '—',
    reasons: asArray(first(v, ['reasons'])),
    answer_conf: first(v, ['answer_conf', 'answerConf', 'conf']),
    threshold: first(v, ['threshold']),
    signals: raw.signals || {},
    decisions: dec,
    step_choice: first(step, ['choice']) || null,
    step_probs: step.probs || {},
    step_names: flat,
    step_conf: first(step, ['answer_conf', 'answerConf']),
    model_versions: dec.model_versions || dec.modelVersions || {},
    pipeline_version: first(cfg, ['pipeline_version', 'pipelineVersion']) ||
      first(cfg.versions || {}, ['pipeline']) || null,
    config_versions: cfg.versions || {},
    raw: raw,
  };
}

function normRisk(raw) {
  var pts = Array.isArray(raw) ? raw : asArray(first(raw, ['points', 'curve', 'data']));
  pts = pts.map(function (p) {
    return {
      threshold: Number(first(p, ['threshold', 't'])),
      coverage: Number(first(p, ['coverage', 'c', 'cov'])),
      error: Number(first(p, ['error', 'err', 'e'])),
    };
  }).filter(function (p) { return isFinite(p.coverage) && isFinite(p.error); });
  return {
    points: pts,
    error_bound: Array.isArray(raw) ? undefined : first(raw, ['upper_bound', 'upperBound', 'error_bound', 'errorBound', 'bound']),
    chosen_threshold: Array.isArray(raw) ? undefined : first(raw, ['chosen_threshold', 'chosenThreshold', 'operating_threshold']),
    n: Array.isArray(raw) ? undefined : first(raw, ['n', 'n_val', 'samples']),
    errors: Array.isArray(raw) ? undefined : first(raw, ['errors']),
    target_error: Array.isArray(raw) ? undefined : first(raw, ['target_error', 'targetError', 'target']),
    delta: Array.isArray(raw) ? undefined : first(raw, ['delta']),
    min_accepted: Array.isArray(raw) ? undefined : first(raw, ['min_accepted', 'minAccepted']),
  };
}

function normGaps(raw) {
  var arr = Array.isArray(raw) ? raw : asArray(first(raw, ['gaps', 'bars', 'items', 'data']));
  return {
    gaps: arr.map(function (g) {
      if (typeof g !== 'object' || g === null) return { code: String(g), count: 0, section: '', share: null };
      return {
        code: first(g, ['reason_code', 'reasonCode', 'code', 'reason']) || '—',
        count: Number(first(g, ['count', 'n', 'value'])) || 0,
        section: first(g, ['sop_section', 'section', 'sop', 'label']) || '',
        share: first(g, ['share', 'fraction', 'frac']),
      };
    }),
    total: Array.isArray(raw) ? undefined : first(raw, ['total_segments', 'total', 'n']),
    window: Array.isArray(raw) ? undefined : first(raw, ['window_utc', 'window']),
  };
}

function normPico(raw) {
  raw = raw || {};
  var led = String(first(raw, ['last_hb_led', 'lastHbLed', 'led']) || 'OFF').toUpperCase();
  if (['G', 'A', 'R', 'OFF'].indexOf(led) === -1) {
    led = ({ GREEN: 'G', AMBER: 'A', RED: 'R', GRAY: 'OFF', OFF: 'OFF', ON: 'G' })[led] || 'OFF';
  }
  var ageS = first(raw, ['last_hb_age_s', 'lastHbAgeS', 'heartbeat_age_s', 'heartbeatAgeS']);
  return {
    connected: first(raw, ['connected']) !== undefined ? !!first(raw, ['connected']) : (led !== 'OFF'),
    led: led,
    port: first(raw, ['port']) || null,
    ageMs: (ageS === undefined || ageS === null) ? null : Number(ageS) * 1000,
    ticks: first(raw, ['last_hb_ticks_ms', 'lastHbTicksMs', 'ticks_ms', 'ticks']),
    offlineReason: first(raw, ['reason']) || null,
  };
}

function normReverify(raw) {
  raw = raw || {};
  return {
    match: !!first(raw, ['match']),
    expected: first(raw, ['expected']),
    actual: first(raw, ['actual']),
    diff: asArray(first(raw, ['diff', 'diffs', 'fields'])),
  };
}

/* ============================ filters ============================ */

var FILTER_DEFAULTS = { site: '', worker: '', batch: '', verdict: '', reason: '', q: '' };

function filterOptions(batches) {
  var sites = {}, workers = {}, batchIds = {}, verdicts = {}, reasons = {};
  (batches || []).forEach(function (b) {
    if (b.site_id && b.site_id !== '—') sites[b.site_id] = 1;
    if (b.batch_id) batchIds[b.batch_id] = 1;
    (b.rows || []).forEach(function (r) {
      if (r.worker_id) workers[r.worker_id] = 1;
      if (VERDICTS.indexOf(r.verdict) !== -1) verdicts[r.verdict] = 1;
      (r.reasons || []).forEach(function (c) { reasons[c] = 1; });
    });
    if (BATCH_DECISIONS.indexOf(b.decision) !== -1) verdicts[b.decision] = 1;
  });
  function keys(o) { return Object.keys(o).sort(); }
  return {
    sites: keys(sites), workers: keys(workers), batches: keys(batchIds),
    verdicts: VERDICTS.filter(function (v) { return verdicts[v]; }),
    reasons: REASON_CODES.filter(function (c) { return reasons[c]; }),
    langs: LANGS.slice(),
  };
}

function matchQuery(b, r, q) {
  if (!q) return true;
  q = String(q).toLowerCase();
  var hay = [b.batch_id, b.site_id, b.decision,
    r ? r.segment_id : '', r ? r.verdict : '',
    r ? (r.reasons || []).join(' ') : ''].join(' ').toLowerCase();
  return q.split(/\s+/).every(function (tok) { return tok === '' || hay.indexOf(tok) !== -1; });
}

// Returns a NEW list of batches; each batch keeps only matching rows.
// A batch survives if it matches the batch-level filters (site/batch id)
// AND keeps >= 1 row, or if every row filter is empty and the batch itself matches.
function applyFilters(batches, f) {
  f = f || {};
  var site = f.site || '', worker = f.worker || '', batch = f.batch || '';
  var verdict = f.verdict || '', reason = f.reason || '', q = (f.q || '').trim();
  return (batches || []).map(function (b) {
    if (site && b.site_id !== site) return null;
    if (batch && b.batch_id !== batch) return null;
    var rows = (b.rows || []).filter(function (r) {
      if (verdict && r.verdict !== verdict) return false;
      if (worker && r.worker_id !== worker) return false;
      if (reason && (r.reasons || []).indexOf(reason) === -1) return false;
      if (q && !matchQuery(b, r, q)) return false;
      return true;
    });
    // batch-level query with no row match: keep batch if the batch itself matches
    if (!rows.length) {
      if (q && matchQuery(b, null, q) && !verdict && !worker && !reason) {
        return { batch: b, rows: [] };
      }
      return null;
    }
    return { batch: b, rows: rows };
  }).filter(Boolean);
}

/* ============================ hash router ============================ */

function parseHash(h) {
  h = String(h || '#/batches');
  var m = h.match(/^#\/(batches|segment|insights)(?:\/([^?]*))?(?:\?(.*))?$/);
  if (!m) return { route: 'batches', id: null, filters: Object.assign({}, FILTER_DEFAULTS), lang: LANGS[0], head: '' };
  var params = {};
  (m[3] || '').split('&').forEach(function (pair) {
    if (!pair) return;
    var kv = pair.split('=');
    try { params[decodeURIComponent(kv[0])] = decodeURIComponent((kv[1] || '').replace(/\+/g, ' ')); }
    catch (e) { params[kv[0]] = kv[1] || ''; }
  });
  function clean(v, allowed) {
    if (!v) return '';
    if (allowed && allowed.indexOf(v) === -1) return '';
    return v;
  }
  var filters = {
    site: params.site || '', worker: params.worker || '', batch: params.batch || '',
    verdict: clean(params.verdict, VERDICTS.concat(BATCH_DECISIONS)),
    reason: clean(params.reason, REASON_CODES), q: params.q || '',
  };
  var lang = LANGS.indexOf(params.lang) !== -1 ? params.lang : LANGS[0];
  return {
    route: m[1],
    id: m[2] ? (() => { try { return decodeURIComponent(m[2]); } catch (e) { return m[2]; } })() : null,
    filters: filters, lang: lang, head: params.head || '',
  };
}

function buildHash(route, id, filters, extra) {
  var h = '#/' + route + (id ? '/' + encodeURIComponent(id) : '');
  var q = [];
  ['site', 'worker', 'batch', 'verdict', 'reason', 'q'].forEach(function (k) {
    if (filters && filters[k]) q.push(k + '=' + encodeURIComponent(filters[k]));
  });
  if (extra) Object.keys(extra).forEach(function (k) {
    if (extra[k]) q.push(k + '=' + encodeURIComponent(extra[k]));
  });
  return q.length ? h + '?' + q.join('&') : h;
}

/* ============================ transport ============================ */

// WebSocket URL for WS /ws (live push channel).
function wsUrl(apiBase, protocol, host) {
  if (apiBase) return apiBase.replace(/^http/, 'ws') + '/ws';
  return (protocol === 'https:' ? 'wss://' : 'ws://') + host + '/ws';
}

function httpGet(fetchFn, base, path, timeoutMs) {  var ctl = new AbortController();
  var timer = setTimeout(function () { ctl.abort(); }, timeoutMs || 12000);
  return fetchFn(base + path, { headers: { Accept: 'application/json' }, signal: ctl.signal })
    .then(function (res) {
      if (!res.ok) throw new Error('GET ' + path + ' → ' + res.status + ' ' + res.statusText);
      return res.json();
    })
    .then(function (data) { clearTimeout(timer); return data; },
      function (e) { clearTimeout(timer); throw e; });
}

function createClient(opts) {
  opts = opts || {};
  var mock = !!opts.mock;
  var base = (opts.apiBase || '').replace(/\/+$/, '');
  var fetchFn = opts.fetchFn || fetch;
  var cache = { files: {}, batchDetail: {}, receipts: {} };

  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function mockFile(name) {
    if (cache.files[name]) return Promise.resolve(cache.files[name]);
    return delay(MOCK_DELAY).then(function () {
      return fetchFn('mock/' + name + '.json', { headers: { Accept: 'application/json' } });
    }).then(function (res) {
      if (!res.ok) throw new Error('mock/' + name + '.json → ' + res.status);
      return res.json();
    }).then(function (data) { cache.files[name] = data; return data; });
  }

  // Every method returns the LIVE endpoint shape; mocks reshape fixtures
  // into those shapes here so app.js never sees fixture internals.
  return {
    mode: mock ? 'mock' : 'live',
    base: base,
    voiceUrl: function (code, lang) {
      return base + '/voice/' + encodeURIComponent(code) + '/' + encodeURIComponent(lang);
    },
    batches: function () {
      return mock ? mockFile('batches') : httpGet(fetchFn, base, '/batches');
    },
    batchDetail: function (id) {
      if (cache.batchDetail[id]) return Promise.resolve(cache.batchDetail[id]);
      var p = mock
        ? mockFile('batches').then(function (all) {
            var list = all.batches || all;
            var b = list.filter(function (x) {
              return (x.batch_id || x.batchId || x.id) === id;
            })[0];
            if (!b) throw new Error('mock: batch ' + id + ' not in fixture');
            return mockFile('receipts').then(function (receipts) {
              var ids = Object.keys(receipts).filter(function (k) { return k.charAt(0) !== '_'; });
              var mine = ids.filter(function (rid) {
                var seg = receipts[rid].verdict && receipts[rid].verdict.segment_id;
                return (b.segment_ids || []).indexOf(seg) !== -1;
              });
              mine.sort();
              var out = {};
              Object.keys(b).forEach(function (k) { out[k] = b[k]; });
              out.receipt_ids = mine;
              return out;
            });
          })
        : httpGet(fetchFn, base, '/batches/' + encodeURIComponent(id));
      return p.then(function (d) { cache.batchDetail[id] = d; return d; });
    },
    receipt: function (id) {
      // {body, id, fallback}: unknown ids resolve to the first receipt,
      // labelled — never silent. Live mode returns the body as-is.
      if (cache.receipts[id]) return Promise.resolve(cache.receipts[id]);
      var p = mock
        ? delay(MOCK_DELAY).then(function () { return mockFile('receipts'); }).then(function (all) {
            var keys = Object.keys(all).filter(function (k) { return k.charAt(0) !== '_'; });
            if (all[id]) return { body: all[id], id: id, fallback: null };
            return { body: all[keys[0]], id: keys[0], fallback: id };
          })
        : httpGet(fetchFn, base, '/receipts/' + encodeURIComponent(id))
          .then(function (body) { return { body: body, id: id, fallback: null }; });
      return p.then(function (r) { cache.receipts[r.id] = r; return r; });
    },
    reverify: function (id, expected) {
      var p = mock
        ? delay(900).then(function () {
            if (opts.forceMismatch) {
              return { match: false, expected: expected, actual: 'm' + String(expected || '').slice(1), diff: ['decisions.step.answer_conf', '_mock.forced_mismatch'] };
            }
            return { match: true, expected: expected, actual: expected, diff: [] };
          })
        : httpGet(fetchFn, base, '/__unused__').then(function () { return null; });
      if (!mock) {
        var ctl = new AbortController();
        var timer = setTimeout(function () { ctl.abort(); }, 60000);
        p = fetchFn(base + '/receipts/' + encodeURIComponent(id) + '/reverify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: '{}', signal: ctl.signal,
        }).then(function (res) {
          clearTimeout(timer);
          if (!res.ok) throw new Error('POST /receipts/' + id + '/reverify → ' + res.status);
          return res.json();
        }, function (e) { clearTimeout(timer); throw e; });
      }
      return p.then(normReverify);
    },
    risk: function () {
      return mock ? mockFile('risk_coverage') : httpGet(fetchFn, base, '/stats/risk_coverage');
    },
    gaps: function () {
      return mock ? mockFile('sop_gaps') : httpGet(fetchFn, base, '/stats/sop_gaps');
    },
    pico: function () {
      var p = mock ? mockFile('pico_status') : httpGet(fetchFn, base, '/pico/status');
      return p.then(function (raw) {
        var out = {};
        Object.keys(raw).forEach(function (k) { out[k] = raw[k]; });
        if (mock) { // a live heartbeat: last beat ~1s before each poll
          out.last_hb_age_s = 0.9;
          out.last_hb_ticks_ms = (raw.last_hb_ticks_ms || 0) + 900;
        }
        return out;
      });
    },
  };
}

// Pure segment→receipt resolution over already-fetched bodies:
// detail carries receipt_ids; each body carries verdict.segment_id.
function findReceiptForSegment(detail, bodiesById, segmentId) {
  var ids = detail.receipt_ids || detail.receiptIds || [];
  for (var i = 0; i < ids.length; i++) {
    var body = bodiesById[ids[i]];
    if (!body) continue;
    var v = body.verdict || {};
    if ((v.segment_id || v.segmentId) === segmentId) return ids[i];
  }
  return null;
}

function shortHash(h, n) {
  h = String(h || '');
  if (!h || h === '—') return '—';
  n = n || 12;
  return h.length > n + 1 ? h.slice(0, n) + '…' : h;
}

return {
  LANGS: LANGS, REASON_CODES: REASON_CODES, VERDICTS: VERDICTS,
  BATCH_DECISIONS: BATCH_DECISIONS, FILTER_DEFAULTS: FILTER_DEFAULTS,
  first: first,
  normBatch: normBatch, normBatchList: normBatchList, normReceipt: normReceipt,
  normRisk: normRisk, normGaps: normGaps, normPico: normPico, normReverify: normReverify,
  filterOptions: filterOptions, applyFilters: applyFilters, matchQuery: matchQuery,
  parseHash: parseHash, buildHash: buildHash, wsUrl: wsUrl,
  createClient: createClient, findReceiptForSegment: findReceiptForSegment,
  shortHash: shortHash,
};
}));
