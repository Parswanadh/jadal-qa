/* Screenshots for the Jadal QA dashboard (docs/screens/).
 *
 * Run from the repo root of this worktree:
 *   node docs/screens/take_shots.cjs
 *
 * Serves web/ on 127.0.0.1:8123 itself (no extra tooling), drives it with
 * Playwright, and asserts: zero console errors, zero failed requests, and no
 * horizontal overflow at phone width. Exits non-zero if any check fails.
 *
 * Needs the `playwright-core` package resolvable from NODE_PATH
 * (on this machine: /home/parshu/.tooling/node_modules).
 */
'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const PW = process.env.PLAYWRIGHT_CORE || '/home/parshu/.tooling/node_modules/playwright-core';
const { chromium } = require(PW);

const WEB = path.resolve(__dirname, '..', '..', 'web');
const OUT = __dirname;
const PORT = 8123;
const BASE = `http://127.0.0.1:${PORT}/index.html?mock=1`;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wav': 'audio/wav',
};

function serve() {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      const urlPath = decodeURIComponent(req.url.split('?')[0]);
      let file = path.normalize(path.join(WEB, urlPath === '/' ? 'index.html' : urlPath));
      if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404, { 'content-type': 'text/plain' });
        res.end('not found');
        return;
      }
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
      fs.createReadStream(file).pipe(res);
    });
    srv.listen(PORT, '127.0.0.1', () => resolve(srv));
  });
}

async function shot(page, name, url, viewport, colorScheme) {
  const errors = [];
  const failed = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('requestfailed', (r) => failed.push(r.url()));
  await page.setViewportSize(viewport);
  if (colorScheme) await page.emulateMedia({ colorScheme });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200); // mock latency + render
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  await page.screenshot({ path: path.join(OUT, name), fullPage: true });
  const size = fs.statSync(path.join(OUT, name)).size;
  console.log(`${name}: ${viewport.width}x${viewport.height}${colorScheme ? ' ' + colorScheme : ''} ` +
    `overflow=${overflow}px consoleErrors=${errors.length} failedReqs=${failed.length} bytes=${size}`);
  errors.forEach((e) => console.log('  console-error: ' + e.slice(0, 200)));
  failed.forEach((u) => console.log('  failed-req: ' + u));
  if (errors.length || failed.length) throw new Error(`${name}: page had errors`);
  if (overflow > 0) throw new Error(`${name}: horizontal overflow of ${overflow}px`);
}

(async () => {
  const srv = await serve();
  const browser = await chromium.launch();
  try {
    // newest batch + a reject segment, resolved to its receipt exactly like
    // the UI does (batch detail receipt_ids are ordered by receipt_id)
    const doc = JSON.parse(fs.readFileSync(path.join(WEB, 'mock', 'batches.json'), 'utf8'));
    const batches = doc.batches || doc;
    const newest = batches[batches.length - 1];
    const receipts = JSON.parse(fs.readFileSync(path.join(WEB, 'mock', 'receipts.json'), 'utf8'));
    const rids = Object.keys(receipts).filter((k) => k[0] !== '_').sort();
    const rejectSeg = (newest.verdicts.find((v) => v.verdict === 'reject') || newest.verdicts[0]).segment_id;
    const segId = rids.find((rid) => receipts[rid].verdict.segment_id === rejectSeg);
    if (!segId) throw new Error('fixture has no receipt for segment ' + rejectSeg);

    // Fresh page per shot: same-document hash navigations would otherwise
    // leak JS state between shots (e.g. the forced-mismatch checkbox).
    async function freshShot(name, url, viewport, colorScheme) {
      const pg = await browser.newPage();
      try {
        await shot(pg, name, url, viewport, colorScheme);
      } finally {
        await pg.close();
      }
    }

    await freshShot('batches-1280.png', BASE + '#/batches', { width: 1280, height: 900 });
    await freshShot('queue-1280.png', BASE + '#/queue', { width: 1280, height: 900 });
    await freshShot('videos-1280.png', BASE + '#/videos', { width: 1280, height: 900 });
    const page = await browser.newPage();
    await shot(page, 'segment-1280.png', BASE + '#/segment/' + segId, { width: 1280, height: 900 });
    // click Re-verify to capture the MATCH banner
    await page.click('#rv-run');
    await page.waitForTimeout(1600);
    await page.screenshot({ path: path.join(OUT, 'reverify-1280.png'), fullPage: true });
    console.log('reverify-1280.png: after clicking Re-verify');
    const banner = await page.textContent('#rv-out');
    console.log('  banner text: ' + banner.replace(/\s+/g, ' ').slice(0, 120));
    if (!/MATCH/.test(banner)) throw new Error('re-verify banner missing MATCH');
    // force-mismatch path: the ✕ state must be just as prominent
    await page.click('#mock-mismatch');
    await page.click('#rv-run');
    await page.waitForTimeout(1600);
    await page.screenshot({ path: path.join(OUT, 'reverify-mismatch-1280.png'), fullPage: true });
    console.log('reverify-mismatch-1280.png: after forcing a mismatch');
    const banner2 = await page.textContent('#rv-out');
    console.log('  banner text: ' + banner2.replace(/\s+/g, ' ').slice(0, 140));
    if (!/MISMATCH/.test(banner2)) throw new Error('re-verify banner missing MISMATCH');
    await page.close(); // drop the forced-mismatch state before the rest
    await freshShot('insights-1280.png', BASE + '#/insights', { width: 1280, height: 900 });
    await freshShot('batches-390.png', BASE + '#/batches', { width: 390, height: 844 });
    await freshShot('queue-390.png', BASE + '#/queue', { width: 390, height: 844 });
    await freshShot('segment-390.png', BASE + '#/segment/' + segId, { width: 390, height: 844 });
    await freshShot('videos-390.png', BASE + '#/videos', { width: 390, height: 844 });
    await freshShot('insights-390.png', BASE + '#/insights', { width: 390, height: 844 });
    await freshShot('batches-dark-1280.png', BASE + '#/batches', { width: 1280, height: 900 }, 'dark');
    await freshShot('queue-dark-1280.png', BASE + '#/queue', { width: 1280, height: 900 }, 'dark');
    await freshShot('videos-dark-1280.png', BASE + '#/videos', { width: 1280, height: 900 }, 'dark');
    await freshShot('segment-dark-1280.png', BASE + '#/segment/' + segId, { width: 1280, height: 900 }, 'dark');
    await freshShot('insights-dark-1280.png', BASE + '#/insights', { width: 1280, height: 900 }, 'dark');
    console.log('ALL SHOTS OK');
  } finally {
    await browser.close();
    srv.close();
  }
})().catch((e) => { console.error('FAILED: ' + e.message); process.exit(1); });
