import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, lstatSync } from 'node:fs';
import { lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// MS4 Oracle real-browser harness (R3).
//
// One invocation runs ONE named variant against a fresh, no-store loopback
// fixture and proves, against the REAL runtime index.html, the runtime and
// ownership contracts the fresh reviewers required:
//   * analyser-driven hive-cell mouth movement (visual);
//   * stale-audio ownership race (decode completes after barge-in);
//   * full replacement-turn cancellation;
//   * retained-page audible-onset + blocked zero-audio + recovery + drain;
//   * both VAD lifecycle branches + adaptive hangover;
//   * bounded, ownership-aware stream deadlines (stalled fetch AND stalled read)
//     that abort once, clean up, and permit a subsequent valid turn (finding 1);
//   * delayed legacy barge-ack turn ownership (finding 2);
//   * decoded silence and good+audio_error resolve to degraded, not done
//     (finding 3);
//   * full-duplex enable/disable generation ownership + concurrent enables
//     (finding 4);
//   * monotonic latest-turn acceptance-snapshot ownership (finding 5).
//
// Evidence integrity:
//   * the EXACT served HTML buffer is hashed (SHA-256 + byte length) and bound
//     in identity so an ABA path/file swap cannot execute different bytes while
//     recorded source hashes still match (finding 6);
//   * artifacts publish into a unique <variant>__<runId> directory with
//     exclusive-create (non-replacing) writes, and the persisted result JSON is
//     byte-identical to stdout (finding 9);
//   * every acquired resource (fixture server, profile, Chrome, CDP socket) is
//     released inside ONE cleanup boundary, and Chrome exit is awaited by the
//     observed 'exit' event (finding 10).
// ---------------------------------------------------------------------------

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = process.env.MS4_BROWSER_REPO_ROOT
  ? resolve(process.env.MS4_BROWSER_REPO_ROOT)
  : resolve(scriptDir, '..', '..');
const indexPath = join(repoRoot, 'machine_spirit_4', 'web', 'index.html');
const evidenceDir = process.env.MS4_BROWSER_EVIDENCE_DIR
  ? resolve(process.env.MS4_BROWSER_EVIDENCE_DIR)
  : null;
const variant = process.env.MS4_BROWSER_VARIANT === 'undecodable' ? 'undecodable' : 'canonical';
const undecodableFlag = variant === 'undecodable' ? 'true' : 'false';
const variantId = `oracle-browser-${variant}`;
// R6: the wrapper generates the runId so it can pre-create the identity-bound
// staging/publish paths and own the final commit. Raw-node runs generate one.
const runId = process.env.MS4_BROWSER_RUN_ID || randomUUID();
const PROFILE_PREFIX = 'ms4-oracle-browser-';
const STAGING_PREFIX = '.staging-';
const CDP_DEFAULT_TIMEOUT_MS = 15_000;
const CDP_BEHAVIOR_TIMEOUT_MS = 30_000;

function chromePath() {
  const candidates = [
    process.env.MS4_CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].filter(Boolean);
  const found = candidates.find(candidate => existsSync(candidate));
  if (!found) throw new Error('Chrome/Edge not found; set MS4_CHROME_PATH');
  return found;
}

async function sha256File(path) {
  // B-01: bind the exact bytes of an on-disk dependency (harness, proof audio,
  // Node/Chrome executable). Returns null if the path cannot be read so a
  // missing digest is visible rather than silently absent.
  try {
    return createHash('sha256').update(await readFile(path)).digest('hex').toUpperCase();
  } catch (_error) {
    return null;
  }
}

async function fetchJson(url, timeoutMs = 3_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function listen(server) {
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  return server.address().port;
}

async function waitForTarget(debugPort, pageUrl, stderrLines, chrome) {
  const timeoutMs = 15_000;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (chrome && (chrome.__spawnFailed || chrome.exitCode !== null || chrome.signalCode !== null)) {
      throw new Error(`Chrome failed to start [spawnFailed=${chrome.__spawnFailed || ''} exitCode=${chrome.exitCode} signal=${chrome.signalCode}] stderr=${stderrLines.join('')}`);
    }
    try {
      const targets = await fetchJson(`http://127.0.0.1:${debugPort}/json/list`);
      const target = targets.find(item => item.type === 'page' && item.url.startsWith(pageUrl));
      if (target) return target;
    } catch (_error) {
      // Chrome is still starting.
    }
    await new Promise(resolveWait => setTimeout(resolveWait, 100));
  }
  throw new Error(
    `waitForTarget timeout [phase=devtools-target elapsedMs=${Date.now() - start} thresholdMs=${timeoutMs} pageUrl=${pageUrl}] stderr=${stderrLines.join('')}`,
  );
}

// Finding 10 / B-03: resolve on the OBSERVED child 'exit' event and return a
// CHECKED boolean — true only if the child was actually observed to exit
// (either already dead, or via the 'exit' event). A hard timer escalates to
// SIGKILL if the tree is stubborn; the absolute cap resolves FALSE so the
// caller can fail closed instead of trusting an unobserved exit.
function awaitChildExit(child, timeoutMs) {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise(resolveExit => {
    const onExit = () => { clearTimeout(hard); clearTimeout(cap); resolveExit(true); };
    child.once('exit', onExit);
    const hard = setTimeout(() => { try { child.kill('SIGKILL'); } catch (_e) { /* gone */ } }, timeoutMs);
    const cap = setTimeout(() => { try { child.removeListener('exit', onExit); } catch (_e) {} resolveExit(false); }, timeoutMs * 2);
  });
}

// R5/R6: kill the ENTIRE owned browser tree (main + renderer/GPU/utility
// children), not just the main process. On Windows the children keep the
// profile's files LOCKED after the parent is killed. Returns a CHECKED status
// (never swallowed). NOTE: for wrapper-driven acceptance runs the authoritative
// tree-death mechanism is the Python wrapper's handle-bound Job Object (kill +
// active-process-count-zero wait), which is immune to PID reuse; this PID kill
// is a same-run accelerator only.
function killProcessTree(pid) {
  if (pid == null) return { killed: false, method: 'none', ok: true, note: 'no pid' };
  try {
    if (process.platform === 'win32') {
      const r = spawnSync('taskkill', ['/F', '/T', '/PID', String(pid)], { stdio: 'ignore', windowsHide: true });
      // 0 = killed; 128 (not found) = already dead. Anything else is a failure.
      const ok = r.status === 0 || r.status === 128;
      return { killed: r.status === 0, method: 'taskkill', status: r.status, ok, error: ok ? null : `taskkill status ${r.status}` };
    }
    try { process.kill(-pid, 'SIGKILL'); return { killed: true, method: 'killpg', ok: true }; }
    catch (_e) {
      try { process.kill(pid, 'SIGKILL'); return { killed: true, method: 'kill', ok: true }; }
      catch (_e2) { return { killed: false, method: 'kill', ok: true, note: 'already gone' }; }
    }
  } catch (error) {
    return { killed: false, method: 'error', ok: false, error: String((error && error.message) || error) };
  }
}

// R6: no-follow identity capture (device + file index via lstat; Node populates
// Stats.dev/ino from the Windows volume serial + file id and the POSIX
// device/inode) plus reparse/symlink detection. Used to prove the SAME object
// before every delete, so a rename-aside + replacement swap cannot be deleted.
async function captureDirIdentity(dir) {
  // A2: no-follow, and { bigint: true } so a 64-bit Windows file ID (ino) is
  // preserved losslessly rather than truncated to a JS Number.
  const st = await lstat(dir, { bigint: true });
  return { dev: String(st.dev), ino: String(st.ino), isSymlink: st.isSymbolicLink(), isDir: st.isDirectory() };
}

// A2: no-follow existence — a dangling symlink/reparse must NOT read as absent.
function lexistsSync(p) {
  try { lstatSync(p); return true; } catch (_e) { return false; }
}

// R6: component-aware containment (NOT string prefixing). `C:\Temp-sibling\x`
// must NOT be considered contained under `C:\Temp` even though the string
// prefix matches. Case-insensitive on Windows.
function componentsContained(childPath, parentPath) {
  const norm = p => resolve(p).split(/[\\/]+/).filter(Boolean).map(s => (process.platform === 'win32' ? s.toLowerCase() : s));
  const child = norm(childPath);
  const parent = norm(parentPath);
  if (child.length <= parent.length) return false;  // must be a strict subdir
  for (let i = 0; i < parent.length; i += 1) {
    if (child[i] !== parent[i]) return false;
  }
  return true;
}

// R6: remove ONLY an exact run-owned directory, identity-bound, with retries.
// Refuses any path not component-contained under `expectedParent`, whose
// basename lacks `prefix`, that is a symlink/reparse point, or whose device+ino
// identity does not match the identity captured at creation (a replacement swap).
// Returns a CHECKED result and NEVER swallows a genuine failure; the caller
// fails closed on residue. Never deletes a foreign/replacement object.
async function removeOwnedDir(dir, expectedParent, prefix, capturedIdentity) {
  if (!dir) return { removed: true, dir: null, attempts: 0, remaining: [], verified: true };
  const resolved = resolve(dir);
  if (!componentsContained(resolved, expectedParent) || !basename(resolved).startsWith(prefix)) {
    return { removed: false, dir: resolved, attempts: 0, remaining: [resolved], error: 'refusing: not component-contained under owned parent / wrong prefix' };
  }
  let lastError = null;
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    if (!lexistsSync(resolved)) return { removed: true, dir: resolved, attempts: attempt, remaining: [], verified: true };
    let id = null;
    try { id = await captureDirIdentity(resolved); }
    catch (_e) { if (!lexistsSync(resolved)) return { removed: true, dir: resolved, attempts: attempt, remaining: [], verified: true }; }
    if (!id) {
      return { removed: false, dir: resolved, attempts: attempt, remaining: [resolved], error: 'refusing: could not capture no-follow identity' };
    }
    if (id.isSymlink) {
      return { removed: false, dir: resolved, attempts: attempt, remaining: [resolved], error: 'refusing: path is a symlink/reparse point' };
    }
    if (capturedIdentity && (id.dev !== capturedIdentity.dev || id.ino !== capturedIdentity.ino)) {
      return { removed: false, dir: resolved, attempts: attempt, remaining: [resolved], error: 'refusing: directory identity changed since creation (replacement/foreign object)' };
    }
    try { await rm(resolved, { recursive: true, force: true }); }
    catch (error) { lastError = String((error && error.message) || error); }
    if (!lexistsSync(resolved)) return { removed: true, dir: resolved, attempts: attempt, remaining: [], verified: true };
    await new Promise(r => setTimeout(r, 300));
  }
  return { removed: !lexistsSync(resolved), dir: resolved, attempts: 10, remaining: lexistsSync(resolved) ? [resolved] : [], error: lastError };
}

class CdpClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.closed = false;
    this.socket.addEventListener('message', event => {
      const message = JSON.parse(String(event.data));
      if (!message.id) return;
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      clearTimeout(waiter.timer);
      this.pending.delete(message.id);
      if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
      else waiter.resolve(message.result);
    });
    this.socket.addEventListener('close', () => this.rejectAllPending('CDP socket closed'));
    this.socket.addEventListener('error', () => this.rejectAllPending('CDP socket error'));
  }

  async open() {
    if (this.socket.readyState === WebSocket.OPEN) return;
    await new Promise((resolveOpen, reject) => {
      const timer = setTimeout(() => reject(new Error('CDP open timeout [thresholdMs=10000]')), 10_000);
      this.socket.addEventListener('open', () => { clearTimeout(timer); resolveOpen(); }, { once: true });
      this.socket.addEventListener('error', () => { clearTimeout(timer); reject(new Error('CDP socket error before open')); }, { once: true });
    });
  }

  rejectAllPending(reason) {
    this.closed = true;
    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error(reason));
    }
    this.pending.clear();
  }

  call(method, params = {}, options = {}) {
    const timeoutMs = options.timeoutMs || CDP_DEFAULT_TIMEOUT_MS;
    if (this.closed) return Promise.reject(new Error(`CDP client closed before call ${method}`));
    return new Promise((resolveCall, reject) => {
      const id = this.nextId++;
      const start = Date.now();
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout [method=${method} elapsedMs=${Date.now() - start} thresholdMs=${timeoutMs}]`));
        }
      }, timeoutMs);
      this.pending.set(id, { resolve: resolveCall, reject, timer });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  async evaluate(expression, options = {}) {
    const result = await this.call('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    }, options);
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description || JSON.stringify(result.exceptionDetails));
    }
    return result.result.value;
  }

  dispose() {
    this.rejectAllPending('CDP client disposed');
    try { this.socket.close(); } catch (_error) { /* already closed */ }
  }
}

const PAGE_HELPERS = `
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const waitFor = async (predicate, phase, timeoutMs) => {
    timeoutMs = timeoutMs || 3000;
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      let value = null;
      try { value = predicate(); } catch (e) { value = null; }
      if (value) return value;
      await delay(10);
    }
    throw new Error('waitFor timeout [phase=' + phase + ' elapsedMs=' + (Date.now() - start) + ' thresholdMs=' + timeoutMs + ']');
  };
`;

// The EXACT bytes we serve, bound by hash (finding 6).
const htmlBuffer = await readFile(indexPath);
const servedHtmlSha256 = createHash('sha256').update(htmlBuffer).digest('hex').toUpperCase();
const servedHtmlBytes = htmlBuffer.length;
const html = htmlBuffer.toString('utf8');
const requestedDspPath = join(repoRoot, 'machine_spirit_4', 'web', 'ms4_voice_dsp.js');
const dspPath = existsSync(requestedDspPath)
  ? requestedDspPath
  : join(resolve(scriptDir, '..', '..'), 'machine_spirit_4', 'web', 'ms4_voice_dsp.js');
const dspBuffer = await readFile(dspPath);
const dspSha256 = createHash('sha256').update(dspBuffer).digest('hex').toUpperCase();
const voiceInputSessionPath = join(repoRoot, 'machine_spirit_4', 'web', 'voice_input_session.js');
const voiceInputSessionBuffer = await readFile(voiceInputSessionPath);
const voiceInputSessionSha256 = createHash('sha256')
  .update(voiceInputSessionBuffer).digest('hex').toUpperCase();
const pwaAssetSpecs = [
  ['/static/oracle_remote_pwa.js', 'oracle_remote_pwa.js', 'application/javascript; charset=utf-8'],
  ['/service-worker.js', 'service-worker.js', 'application/javascript; charset=utf-8'],
  ['/manifest.webmanifest', 'manifest.webmanifest', 'application/manifest+json; charset=utf-8'],
  ['/static/oracle-icon.svg', 'oracle-icon.svg', 'image/svg+xml; charset=utf-8'],
];
const pwaAssets = new Map();
for (const [route, name, contentType] of pwaAssetSpecs) {
  const buffer = await readFile(join(repoRoot, 'machine_spirit_4', 'web', name));
  pwaAssets.set(route, {
    buffer,
    contentType,
    sha256: createHash('sha256').update(buffer).digest('hex').toUpperCase(),
  });
}
// B-01: self-hash the EXACT executed harness bytes plus the proof-audio bytes
// and the Node executable, so retained evidence proves WHICH harness + inputs +
// runtime generated it (served-HTML binding alone did not close this gap).
const harnessPath = fileURLToPath(import.meta.url);
const harnessSha256 = await sha256File(harnessPath);
const proofAudioPath = join(repoRoot, 'machine_spirit_4', 'canned_audio', 'alloy', 'ack_listening.wav');
const proofAudioBuffer = await readFile(proofAudioPath);
const proofAudioBase64 = proofAudioBuffer.toString('base64');
const proofAudioSha256 = createHash('sha256').update(proofAudioBuffer).digest('hex').toUpperCase();
const nodeExecPath = process.execPath;
const nodeExecSha256 = await sha256File(nodeExecPath);

// A syntactically valid all-zero PCM16 WAV: decodes cleanly, plays as silence,
// and must NEVER cross the non-silent onset threshold (finding 3).
function silentWavBase64(sampleRate, ms) {
  const numSamples = Math.round((sampleRate * ms) / 1000);
  const dataLen = numSamples * 2;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + dataLen, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24); buf.writeUInt32LE(sampleRate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(dataLen, 40);
  return buf.toString('base64');
}
const silentAudioBase64 = silentWavBase64(24000, 140);
const silentAudioSha256 = createHash('sha256').update(Buffer.from(silentAudioBase64, 'base64')).digest('hex').toUpperCase();

const fixtureNonce = randomUUID();

let fixtureServer = null;
let fixturePort = null;
let loopbackUrl = null;
let pageUrl = null;
let acceptanceTarget = true;
let inheritedTarget = process.env.MS4_BROWSER_TARGET_URL || null;
let debugServer = null;
let debugPort = null;
let resolvedChromePath = null;
let chromeArgs = null;
let profileDir = null;
let chrome = null;
let cdp = null;
let chromeIdentity = null;
let fixtureIdentity = null;
let publishDir = null;
// B-03/B-04: run-owned staging + fail-closed seal state.
let stagingDir = null;
let bodySucceeded = false;
let chromeExitObserved = true;
let resultJsonForSeal = null;
// R5/R6 cleanup-ownership state.
let ownsProfileCleanup = true;
let ownsStagingCleanup = false;
let profileResidue = null;
let stagingResidue = null;
let profileIdentity = null;   // R6: no-follow dev/ino captured at creation
let stagingIdentity = null;
let treeKillStatus = null;
// R6: the harness NEVER publishes. Wrapper-driven ⟺ MS4_BROWSER_STAGING_DIR set.
const wrapperDriven = !!process.env.MS4_BROWSER_STAGING_DIR;

let visual = null;
let ownershipRace = null;
let abort = null;
let retained = null;
let voiceReadinessState = null;
let vad = null;
let streamDeadline = null;
let bargeAck = null;
let audibleVerdict = null;
let fullDuplexRace = null;
let experiencePolish = null;
let deferredAutoArm = null;
let productRepair = null;
let restBatch = null;
let restObservability = null;
let snapshotOwnership = null;
let schedulerTiming = null;
let dspDelegation = null;
let staleCues = null;
let longInputSession = null;
let facePrewarm = null;

try {
  // ---- All acquired resources live inside this ONE cleanup boundary. --------
  fixtureServer = createServer((request, response) => {
    const path = new URL(request.url || '/', 'http://127.0.0.1').pathname;
    if (path === '/' || path === '/index.html') {
      response.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-MS4-Fixture-Nonce': fixtureNonce,
        'X-MS4-Served-Sha256': servedHtmlSha256,
      });
      response.end(htmlBuffer);   // serve the EXACT hashed bytes
      return;
    }
    if (path === '/static/ms4_voice_dsp.js') {
      response.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-MS4-Served-Sha256': dspSha256,
      });
      response.end(dspBuffer);
      return;
    }
    if (path === '/static/voice_input_session.js') {
      response.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-MS4-Served-Sha256': voiceInputSessionSha256,
      });
      response.end(voiceInputSessionBuffer);
      return;
    }
    if (pwaAssets.has(path)) {
      const asset = pwaAssets.get(path);
      const headers = {
        'Content-Type': asset.contentType,
        'Cache-Control': 'no-store',
        'X-MS4-Served-Sha256': asset.sha256,
      };
      if (path === '/service-worker.js') headers['Service-Worker-Allowed'] = '/';
      response.writeHead(200, headers);
      response.end(asset.buffer);
      return;
    }
    if (path === '/__identity') {
      response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      response.end(JSON.stringify({
        nonce: fixtureNonce, variant, runId,
        servedHtmlSha256, servedHtmlBytes, pid: process.pid,
      }));
      return;
    }
    if (path === '/favicon.ico') { response.writeHead(204); response.end(); return; }
    if (path === '/api/v1/ms4_gateway/status') {
      response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      response.end(JSON.stringify({
        ok: true,
        service: 'ms4-gateway',
        status: 'ready',
        runtime: 'ms4-fusion',
      }));
      return;
    }
    response.writeHead(404, { 'Content-Type': 'application/json' });
    response.end('{"error":"browser-runtime-fixture"}');
  });
  fixturePort = await listen(fixtureServer);
  loopbackUrl = `http://127.0.0.1:${fixturePort}/`;

  // Refuse an inherited target URL for acceptance unless pinned to THIS run's
  // loopback fixture (a stale env var cannot masquerade as the target).
  pageUrl = loopbackUrl;
  if (inheritedTarget) {
    if (inheritedTarget === loopbackUrl || process.env.MS4_BROWSER_TARGET_PINNED === loopbackUrl) {
      pageUrl = inheritedTarget;
    } else if (process.env.MS4_BROWSER_ALLOW_EXTERNAL_TARGET === '1') {
      pageUrl = inheritedTarget;
      acceptanceTarget = false;
    } else {
      throw new Error(
        `Refusing inherited MS4_BROWSER_TARGET_URL=${inheritedTarget} for acceptance: it is not this run's loopback fixture ${loopbackUrl}. ` +
        `Set MS4_BROWSER_TARGET_PINNED to the loopback URL, or MS4_BROWSER_ALLOW_EXTERNAL_TARGET=1 for a non-acceptance run.`,
      );
    }
  }

  debugServer = createServer();
  debugPort = await listen(debugServer);
  await new Promise(resolveClose => debugServer.close(resolveClose));
  debugServer = null;

  resolvedChromePath = chromePath();
  const chromeExecSha256 = await sha256File(resolvedChromePath);  // B-01: bind the browser binary bytes
  // R6 profile ownership: when the Python wrapper spawns us it CREATES the
  // profile dir and passes it in MS4_BROWSER_PROFILE_DIR — the wrapper OWNS
  // teardown (handle-bound Job Object kill + active-count-zero wait, then
  // identity-bound profile deletion, then atomic commit), covering even a hung
  // or killed harness. When run directly with no wrapper, WE create and own it
  // and clean it up in our own guaranteed finally (identity-bound, fail-closed).
  if (process.env.MS4_BROWSER_PROFILE_DIR) {
    profileDir = resolve(process.env.MS4_BROWSER_PROFILE_DIR);
    ownsProfileCleanup = false;
    await mkdir(profileDir, { recursive: true });
  } else {
    profileDir = await mkdtemp(join(tmpdir(), PROFILE_PREFIX));
    ownsProfileCleanup = true;
    profileIdentity = await captureDirIdentity(profileDir);  // R6: bind creation identity
  }
  // R6 staging ownership: the harness writes evidence ONLY into a private staging
  // dir and NEVER publishes. Wrapper-driven runs use the wrapper's staging dir
  // (the wrapper commits it after tree death + profile deletion). Raw-node runs
  // self-own a staging dir under the evidence root and self-clean it (no accepted
  // evidence is produced by a raw-node run).
  if (process.env.MS4_BROWSER_STAGING_DIR) {
    stagingDir = resolve(process.env.MS4_BROWSER_STAGING_DIR);
    ownsStagingCleanup = false;
    await mkdir(stagingDir, { recursive: true });
  } else if (evidenceDir) {
    stagingDir = join(evidenceDir, `${STAGING_PREFIX}${variant}__${runId}`);
    await rm(stagingDir, { recursive: true, force: true }).catch(() => {});
    await mkdir(stagingDir, { recursive: true });
    ownsStagingCleanup = true;
    stagingIdentity = await captureDirIdentity(stagingDir);
  }
  chromeArgs = [
    '--headless=new',
    '--disable-gpu',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--autoplay-policy=no-user-gesture-required',
    '--no-first-run',
    '--no-default-browser-check',
    // R8 (independent replay P1): NETWORK ISOLATION. The replay recorded Chrome's
    // Media Router / cast discovery reaching a non-loopback LAN device
    // (192.168.0.207:8009). Disable Media Router + all background networking, and
    // force every non-loopback hostname to fail to resolve. The only permitted
    // traffic is the loopback CDP + the loopback fixture page.
    '--disable-features=MediaRouter,DialMediaRouteProvider,CastMediaRouteProvider,CastMediaRouteProviderStreaming,AutofillServerCommunication,OptimizationHints,Translate,NetworkTimeServiceQuerying',
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-domain-reliability',
    '--disable-sync',
    '--disable-client-side-phishing-detection',
    '--disable-breakpad',
    '--metrics-recording-only',
    '--no-pings',
    '--no-proxy-server',
    '--disable-default-apps',
    '--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1 , EXCLUDE ::1 , EXCLUDE localhost',
    `--remote-debugging-port=${debugPort}`,
    '--remote-debugging-address=127.0.0.1',
    `--user-data-dir=${profileDir}`,
    '--window-size=1440,1000',
    pageUrl,
  ];
  const stderrLines = [];
  chrome = spawn(resolvedChromePath, chromeArgs, { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true });
  chrome.stderr.on('data', chunk => {
    if (stderrLines.join('').length < 12_000) stderrLines.push(chunk.toString());
  });
  // Fail fast on a spawn/startup failure (e.g. a non-executable Chrome path)
  // rather than waiting out the DevTools deadline.
  chrome.once('error', (err) => { chrome.__spawnFailed = String((err && err.message) || err); });

  const target = await waitForTarget(debugPort, pageUrl, stderrLines, chrome);
  try { chromeIdentity = await fetchJson(`http://127.0.0.1:${debugPort}/json/version`); } catch (_error) { chromeIdentity = null; }
  cdp = new CdpClient(target.webSocketDebuggerUrl);
  await cdp.open();
  await cdp.call('Page.enable');
  await cdp.call('Runtime.enable');
  await cdp.call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  const readyDeadline = Date.now() + 10_000;
  let pageReady = false;
  while (Date.now() < readyDeadline && !pageReady) {
    try {
      pageReady = await cdp.evaluate(
        `document.readyState === 'complete' && typeof submitWavBlobAsVoiceTurn === 'function' && typeof vadTick === 'function' && typeof enableFullDuplex === 'function'`,
      );
    } catch (error) {
      if (!/Execution context was destroyed/i.test(String(error))) throw error;
    }
    if (!pageReady) await new Promise(resolveWait => setTimeout(resolveWait, 50));
  }
  if (!pageReady) throw new Error(`page-ready timeout [phase=page-ready thresholdMs=10000 pageUrl=${pageUrl}]`);

  if (acceptanceTarget) {
    fixtureIdentity = await cdp.evaluate(`fetch('/__identity').then(function(r){ return r.json(); })`);
    assert.equal(fixtureIdentity.nonce, fixtureNonce, 'served page must be this run loopback fixture');
    assert.equal(fixtureIdentity.variant, variant, 'fixture identity variant must match this run');
    assert.equal(fixtureIdentity.servedHtmlSha256, servedHtmlSha256, 'fixture must report the served-bytes digest');
  }

  // ---- DSP asset must be loaded AND delegated through production VAD. -------
  dspDelegation = await cdp.evaluate(`(async () => {
    const methodNames = [
      'isFiniteSampleArray',
      'effectiveOnsetThreshold',
      'updateNoiseFloor',
      'updateEchoReference',
      'echoGuardOnset',
      'echoGuardTransition',
      'shouldSubmitUtterance',
    ];
    const counts = Object.fromEntries(methodNames.map(name => [name, 0]));
    const dsp = window.MS4DSP;
    const result = {
      loaded: !!dsp,
      version: dsp && dsp.VERSION,
      available: false,
      counts,
    };
    if (!dsp) return result;
    const pageSeamsReady = (
      typeof vadAdaptiveThresholds === 'function'
      && typeof vadReadPlaybackForEcho === 'function'
      && typeof vadTick === 'function'
      && typeof onVadSpeechOffset === 'function'
    );
    if (!pageSeamsReady || methodNames.some(name => typeof dsp[name] !== 'function')) return result;

    const originals = {};
    const savedVad = {...vadState};
    const savedInputSession = activeOracleVoiceInputSession;
    const savedPreRoll = vadPreRoll;
    const savedReadPlayback = readOraclePlaybackSignal;
    const savedVoiceStatus = voiceStatusLine.textContent;
    const savedVoiceTitle = voiceStatusLine.title;
    try {
      for (const name of methodNames) {
        originals[name] = dsp[name];
        dsp[name] = function (...args) {
          counts[name] += 1;
          return originals[name].apply(this, args);
        };
      }
      readOraclePlaybackSignal = () => ({active: true, level: 0.5, voice: 0.5, air: 0});
      Object.assign(vadState, {
        enabled: true,
        analyser: {
          fftSize: 8,
          getFloatTimeDomainData(buffer) { buffer.fill(0.03); },
        },
        phase: 'silence',
        smoothedRms: 0.03,
        frameClipped: false,
        lastTransitionAt: Date.now(),
        speechStartedAt: 0,
        lastSpeechSampleAt: 0,
        noiseFloor: 0.0002,
        uttPeakRms: 0,
        uttFrames: 0,
        uttPlaybackFrames: 0,
        uttPlaybackLevelSum: 0,
        echoReference: null,
        lastPlaybackAt: 0,
        sourceSampleRate: 48000,
      });
      vadTick();
      vadState.phase = 'maybe_speech';
      vadState.lastTransitionAt = Date.now() - vadMinSpeechMs() - 20;
      vadState.smoothedRms = 0.03;
      vadTick();
      vadState.phase = 'speech';
      vadState.speechStartedAt = Date.now() - 100;
      vadState.lastSpeechSampleAt = Date.now();
      const inputSession = beginActiveOracleVoiceInputSession({
        mode: 'vad',
        sourceLabel: 'dsp-delegation',
      });
      inputSession.appendPcm(new Float32Array([0.01, 0.01]));
      vadState.uttPeakRms = 0.03;
      vadState.uttFrames = 1;
      vadState.uttPlaybackFrames = 1;
      vadState.uttPlaybackLevelSum = 0.5;
      await onVadSpeechOffset();
      encodeWavBlob(new Float32Array([0, 0.01]), 16000);
      result.available = true;
      return result;
    } finally {
      for (const name of methodNames) {
        if (originals[name]) dsp[name] = originals[name];
      }
      await cancelActiveOracleVoiceInputSession('dsp-delegation-cleanup');
      activeOracleVoiceInputSession = savedInputSession;
      Object.assign(vadState, savedVad);
      vadPreRoll = savedPreRoll;
      readOraclePlaybackSignal = savedReadPlayback;
      voiceStatusLine.textContent = savedVoiceStatus;
      voiceStatusLine.title = savedVoiceTitle;
    }
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(dspDelegation.loaded, true, 'the real /static/ms4_voice_dsp.js asset must load in the fixture');
  assert.equal(dspDelegation.available, true, 'the inline VAD must expose the approved DSP delegation seams');
  for (const [name, count] of Object.entries(dspDelegation.counts)) {
    assert.ok(count >= 1, `production VAD/WAV paths must invoke MS4DSP.${name} (count=${count})`);
  }

  // ---- Sink admission may resolve after cue ownership became stale. --------
  const reloadPageAndWait = async predicate => {
    await cdp.call('Page.reload', {ignoreCache: true});
    const started = Date.now();
    for (;;) {
      try {
        if (await cdp.evaluate(predicate)) return;
      } catch (_error) {
        // The old execution context is expected to disappear during reload.
      }
      if (Date.now() - started > 10000) {
        throw new Error(`reload readiness timeout: ${predicate}`);
      }
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  };
  await cdp.evaluate(`(() => {
    localStorage.setItem('ms4_voice_full_duplex', 'false');
    localStorage.setItem('ms4_voice_vad_ack', 'true');
    sessionStorage.setItem('ms4_oracle_exact_binding_manifest_v1', JSON.stringify({
      required: true,
      runToken: 'stale-caller-run-token',
      sealAuthorization: 'stale-caller-seal-authorization',
      manifestHash: '${'a'.repeat(64)}',
    }));
    return true;
  })()`);
  await reloadPageAndWait(`typeof oracleExactBinding !== 'undefined'
    && oracleExactBinding.isRequired() === true
    && typeof onVadSpeechOnset === 'function'`);

  staleCues = await cdp.evaluate(`(async () => {
    const saved = {
      playbackCtx,
      voiceModeActive,
      currentVoiceTurnId,
      activeVoiceStreamController,
      activeAudioSources,
      gate: oracleAwaitSinkReadyBeforeStart,
      disable: disableFullDuplex,
      vad: {...vadState},
      vadPreRoll,
      inputSession: activeOracleVoiceInputSession,
    };
    let created = 0;
    let started = 0;
    const fakeCtx = {
      state: 'running',
      currentTime: 1,
      createOscillator() {
        created += 1;
        return {
          type: 'sine',
          frequency: {
            value: 0,
            setValueAtTime() {},
            exponentialRampToValueAtTime() {},
          },
          connect() {},
          start() { started += 1; },
          stop() {},
        };
      },
      createGain() {
        created += 1;
        return {
          gain: {
            value: 0,
            setValueAtTime() {},
            exponentialRampToValueAtTime() {},
          },
          connect() {},
        };
      },
    };
    try {
      playbackCtx = fakeCtx;
      activeVoiceStreamController = null;
      activeAudioSources = [];
      voiceModeActive = true;
      vadState.enabled = false;

      currentVoiceTurnId = 4100;
      let releaseNotify;
      oracleAwaitSinkReadyBeforeStart = () => new Promise(resolve => { releaseNotify = resolve; });
      const notifyPromise = playNotifyChime();
      currentVoiceTurnId = 4101;
      releaseNotify();
      const notifyResult = await notifyPromise;
      const afterNotify = {created, started};

      currentVoiceTurnId = 4200;
      Object.assign(vadState, {
        enabled: true,
        enableGeneration: 77,
        phase: 'speech',
        speechStartedAt: 123456,
        smoothedRms: 0.42,
        uttPeakRms: 0.11,
        uttFrames: 11,
        uttPlaybackFrames: 3,
        uttPlaybackLevelSum: 1.5,
        captureSentinel: 0.1,
        sourceSampleRate: 48000,
      });
      vadPreRoll = [new Float32Array([0.2])];
      let disableCalls = 0;
      disableFullDuplex = async () => {
        disableCalls += 1;
        vadState.enableGeneration += 1;
        vadState.enabled = false;
      };
      let releaseOnset;
      oracleAwaitSinkReadyBeforeStart = () => new Promise(resolve => { releaseOnset = resolve; });
      const onsetPromise = onVadSpeechOnset();
      Object.assign(vadState, {
        enabled: true,
        enableGeneration: 78,
        phase: 'speech',
        speechStartedAt: 654321,
        smoothedRms: 0.91,
        uttPeakRms: 0.91,
        uttFrames: 91,
        uttPlaybackFrames: 17,
        uttPlaybackLevelSum: 8.5,
        captureSentinel: 0.9,
        sourceSampleRate: 48000,
      });
      vadPreRoll = [new Float32Array([0.8])];
      currentVoiceTurnId = 4202;
      releaseOnset();
      await onsetPromise;
      const afterOnset = {created, started};
      const replacement = {
        disableCalls,
        enabled: vadState.enabled,
        generation: vadState.enableGeneration,
        speechStartedAt: vadState.speechStartedAt,
        uttPeakRms: vadState.uttPeakRms,
        uttFrames: vadState.uttFrames,
        uttPlaybackFrames: vadState.uttPlaybackFrames,
        uttPlaybackLevelSum: vadState.uttPlaybackLevelSum,
        captureSentinel: vadState.captureSentinel,
        preRollFirst: vadPreRoll[0] && vadPreRoll[0][0],
      };

      return {notifyResult, afterNotify, afterOnset, replacement};
    } finally {
      playbackCtx = saved.playbackCtx;
      voiceModeActive = saved.voiceModeActive;
      currentVoiceTurnId = saved.currentVoiceTurnId;
      activeVoiceStreamController = saved.activeVoiceStreamController;
      activeAudioSources = saved.activeAudioSources;
      oracleAwaitSinkReadyBeforeStart = saved.gate;
      disableFullDuplex = saved.disable;
      await cancelActiveOracleVoiceInputSession('stale-cue-cleanup');
      activeOracleVoiceInputSession = saved.inputSession;
      Object.assign(vadState, saved.vad);
      if (!Object.prototype.hasOwnProperty.call(saved.vad, 'captureSentinel')) {
        delete vadState.captureSentinel;
      }
      vadPreRoll = saved.vadPreRoll;
    }
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  await reloadPageAndWait(`typeof oracleExactBinding !== 'undefined'
    && oracleExactBinding.isRequired() === false
    && typeof onVadSpeechOnset === 'function'`);
  staleCues.normalReplacement = await cdp.evaluate(`(async () => {
    const saved = {
      playbackCtx,
      currentVoiceTurnId,
      gate: oracleAwaitSinkReadyBeforeStart,
      vad: {...vadState},
      vadPreRoll,
      inputSession: activeOracleVoiceInputSession,
    };
    const fakeCtx = {state: 'running', currentTime: 1};
    try {
      playbackCtx = fakeCtx;
      currentVoiceTurnId = 4300;
      Object.assign(vadState, {
        enabled: true,
        enableGeneration: 87,
        phase: 'speech',
        speechStartedAt: 111111,
        smoothedRms: 0.31,
        uttPeakRms: 0.31,
        uttFrames: 31,
        uttPlaybackFrames: 7,
        uttPlaybackLevelSum: 3.5,
        captureSentinel: 0.3,
        sourceSampleRate: 48000,
      });
      vadPreRoll = [new Float32Array([0.4])];
      let releaseOnset;
      oracleAwaitSinkReadyBeforeStart = () => new Promise(resolve => { releaseOnset = resolve; });
      const onsetPromise = onVadSpeechOnset();
      Object.assign(vadState, {
        enabled: true,
        enableGeneration: 88,
        phase: 'speech',
        speechStartedAt: 222222,
        smoothedRms: 0.81,
        uttPeakRms: 0.81,
        uttFrames: 81,
        uttPlaybackFrames: 18,
        uttPlaybackLevelSum: 9,
        captureSentinel: 0.8,
        sourceSampleRate: 48000,
      });
      vadPreRoll = [new Float32Array([0.7])];
      currentVoiceTurnId = 4302;
      releaseOnset();
      await onsetPromise;
      return {
        enabled: vadState.enabled,
        generation: vadState.enableGeneration,
        speechStartedAt: vadState.speechStartedAt,
        uttPeakRms: vadState.uttPeakRms,
        uttFrames: vadState.uttFrames,
        uttPlaybackFrames: vadState.uttPlaybackFrames,
        uttPlaybackLevelSum: vadState.uttPlaybackLevelSum,
        captureSentinel: vadState.captureSentinel,
        preRollFirst: vadPreRoll[0] && vadPreRoll[0][0],
      };
    } finally {
      playbackCtx = saved.playbackCtx;
      currentVoiceTurnId = saved.currentVoiceTurnId;
      oracleAwaitSinkReadyBeforeStart = saved.gate;
      await cancelActiveOracleVoiceInputSession('normal-stale-cue-cleanup');
      activeOracleVoiceInputSession = saved.inputSession;
      Object.assign(vadState, saved.vad);
      if (!Object.prototype.hasOwnProperty.call(saved.vad, 'captureSentinel')) {
        delete vadState.captureSentinel;
      }
      vadPreRoll = saved.vadPreRoll;
    }
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(staleCues.notifyResult, false, 'a completion chime stale after sink admission must be suppressed');
  assert.deepEqual(staleCues.afterNotify, {created: 0, started: 0}, 'stale chime constructs/starts no output nodes');
  assert.deepEqual(staleCues.afterOnset, {created: 0, started: 0}, 'stale onset constructs/starts no output nodes');
  assert.deepEqual(staleCues.replacement, {
    disableCalls: 0,
    enabled: true,
    generation: 78,
    speechStartedAt: 654321,
    uttPeakRms: 0.91,
    uttFrames: 91,
    uttPlaybackFrames: 17,
    uttPlaybackLevelSum: 8.5,
    captureSentinel: 0.9,
    preRollFirst: 0.800000011920929,
  }, 'stale exact caller must not disable or mutate replacement VAD state');
  assert.deepEqual(staleCues.normalReplacement, {
    enabled: true,
    generation: 88,
    speechStartedAt: 222222,
    uttPeakRms: 0.81,
    uttFrames: 81,
    uttPlaybackFrames: 18,
    uttPlaybackLevelSum: 9,
    captureSentinel: 0.8,
    preRollFirst: 0.699999988079071,
  }, 'stale normal caller must not reset replacement utterance counters/state');

  const completionAnnouncement = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const originalWarn = console.warn;
    const announcementWarnings = [];
    const savedVoiceModeActive = voiceModeActive;
    const savedSpeakCompletions = localStorage.getItem(SPEAK_COMPLETIONS_KEY);
    voiceModeActive = true;
    localStorage.setItem(SPEAK_COMPLETIONS_KEY, 'true');
    console.warn = (...args) => announcementWarnings.push(args.map(value => String(value)).join(' '));
    window.fetch = (url, options = {}) => {
      if (String(url) === '/voice/synthesize/stream') {
        const frames =
          'event: audio_chunk\\ndata: {"index":0,"text":"completion","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
          + 'event: done\\ndata: {"completed":true,"audio_chunks":1}\\n\\n';
        return Promise.resolve(new Response(frames, {status: 200, headers: {'Content-Type': 'text/event-stream'}}));
      }
      return originalFetch(url, options);
    };
    try {
      const ctx = getPlaybackCtx();
      await ctx.resume();
      setOracleStageState('idle', 'Prior completion subtitle.', 'Prior completion detail.');
      const completionSpeakResult = await speakCompletionWhenQuiet('The background job completed.', 1);
      if (!completionSpeakResult) {
        throw new Error(
          'completion announcement failed before playback; activeSources='
          + activeAudioSources.length + '; warnings=' + announcementWarnings.join(' | ')
        );
      }
      const queuedPresence = oracleStage.dataset.presence;
      await waitFor(
        () => oracleStage.dataset.presence === 'speaking',
        'completion-announcement:measured-onset',
        2000,
      );
      const during = {
        presence: oracleStage.dataset.presence,
        activeSources: activeAudioSources.length,
        onsetAnalyserRetained: Boolean(activeAudioSources[0]?.onsetAnalyser),
      };
      const presenceAtDrain = await waitFor(
        () => activeAudioSources.length === 0 ? oracleStage.dataset.presence : null,
        'completion-announcement:drain',
        5000,
      );
      const restored = await waitFor(
        () => oracleStage.dataset.presence === 'idle' ? {
          presence: oracleStage.dataset.presence,
          subtitle: oracleSubtitle.textContent,
          detail: oracleTimingLine.textContent,
          activeSources: activeAudioSources.length,
        } : null,
        'completion-announcement:idle',
        2000,
      );

      setOracleStageState('idle', 'Stale completion subtitle.', 'Stale completion detail.');
      await speakCompletionWhenQuiet('A stale background job completed.', 1);
      const staleOwner = currentVoiceTurnId;
      bargeIn(null, {halfContext: false, playAck: false});
      setOracleStageState('speaking', 'NEWER_COMPLETION_OWNER', 'newer turn active');
      await delay(800);
      const afterStaleDrain = {
        ownerAdvanced: currentVoiceTurnId > staleOwner,
        presence: oracleStage.dataset.presence,
        subtitle: oracleSubtitle.textContent,
        detail: oracleTimingLine.textContent,
        activeSources: activeAudioSources.length,
      };
      return {queuedPresence, during, presenceAtDrain, restored, afterStaleDrain};
    } finally {
      window.fetch = originalFetch;
      console.warn = originalWarn;
      voiceModeActive = savedVoiceModeActive;
      if (savedSpeakCompletions === null) localStorage.removeItem(SPEAK_COMPLETIONS_KEY);
      else localStorage.setItem(SPEAK_COMPLETIONS_KEY, savedSpeakCompletions);
      if (activeAudioSources.length) bargeIn(null, {halfContext: false, playAck: false});
      if (oraclePlaybackAnalyser) {
        try { oraclePlaybackAnalyser.disconnect(); } catch (_error) {}
      }
      oraclePlaybackAnalyser = null;
      oraclePlaybackAnalyserData = null;
      oraclePlaybackTimeData = null;
      oraclePlaybackNode = null;
      resetOraclePlaybackSignal();
      setOracleStageState('idle', 'Ready.', '');
    }
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.notEqual(completionAnnouncement.queuedPresence, 'speaking',
    'a queued completion announcement must not claim speaking before measured onset');
  assert.deepEqual(completionAnnouncement.during, {
    presence: 'speaking',
    activeSources: 1,
    onsetAnalyserRetained: true,
  }, 'a measured audible completion announcement must enter speaking with one owned source');
  assert.equal(completionAnnouncement.presenceAtDrain, 'speaking', 'the Oracle must not become idle before announcement audio drains');
  assert.deepEqual(completionAnnouncement.restored, {
    presence: 'idle',
    subtitle: 'Prior completion subtitle.',
    detail: 'Prior completion detail.',
    activeSources: 0,
  }, 'a drained completion announcement must restore its prior idle copy');
  assert.deepEqual(completionAnnouncement.afterStaleDrain, {
    ownerAdvanced: true,
    presence: 'speaking',
    subtitle: 'NEWER_COMPLETION_OWNER',
    detail: 'newer turn active',
    activeSources: 0,
  }, 'a stale completion owner must not reset a newer turn');

  // ---- Bounded Oracle experience: truthful states, geometry, cues. ---------
  experiencePolish = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const states = ['idle', 'listening', 'thinking', 'reflex', 'speaking', 'blocked', 'error'];
    const stateGeometry = {};
    for (const state of states) {
      setOracleStageState(state, state + ' proof', state + ' detail');
      await delay(30);
      const specs = oracleMirrorCellSpecs();
      const pupils = specs.filter(([, , , , shape]) => shape === 'pupil');
      const eyes = specs.filter(([, , , , shape]) => shape === 'eye');
      const baseVoiceCells = specs.filter(([, , , , shape]) => shape === 'voice');
      const leakingStatusCells = [...oracleMirror.querySelectorAll('.mirror-cell.is-status')]
        .filter(cell => !['eye', 'pupil'].includes(cell.dataset.cellShape || ''));
      stateGeometry[state] = {
        presence: oracleStage.dataset.presence,
        pupils: pupils.length,
        leftEyeCells: eyes.filter(([x]) => x < 180).length,
        rightEyeCells: eyes.filter(([x]) => x > 180).length,
        baseVoiceCells: baseVoiceCells.length,
        speakingBarsOpacity: Number.parseFloat(
          getComputedStyle(oracleMirror.querySelector('.mirror-speaking-bars')).opacity),
        neuralWaveOpacity: Number.parseFloat(
          getComputedStyle(oracleMirror.querySelector('.mirror-wave')).opacity),
        leakingStatusCells: leakingStatusCells.length,
        signature: [
          oracleMirror.querySelectorAll('.mirror-cell.is-active').length,
          oracleMirror.querySelectorAll('.mirror-cell.is-status').length,
          getComputedStyle(oracleStage).getPropertyValue('--mirror-a-rgb').trim(),
          getComputedStyle(oracleStage).getPropertyValue('--mirror-c-rgb').trim(),
        ].join('|'),
      };
    }

    localStorage.removeItem(THINKING_AMBIENCE_KEY);
    const defaultAmbience = isThinkingAmbienceEnabled();
    const defaultAmbienceToggle = settingsThinkingAmbienceToggle.checked;
    const reducedMotionQuery = window.matchMedia;
    window.matchMedia = query => ({
      matches: query === '(prefers-reduced-motion: reduce)',
      media: query,
      addEventListener() {},
      removeEventListener() {},
    });
    const reducedMotionOptionalSound = oracleOptionalSoundEnabled();
    window.matchMedia = reducedMotionQuery;

    const ctx = getPlaybackCtx();
    await ctx.resume();
    const binary = atob('${proofAudioBase64}');
    const reflexBytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      reflexBytes[index] = binary.charCodeAt(index);
    }
    const reflexBuffer = await ctx.decodeAudioData(reflexBytes.buffer);
    reflexBuffers.set('__experience_reflex__', reflexBuffer);
    reflexMetaById.set('__experience_reflex__', {
      id: '__experience_reflex__', text: 'Fallback acknowledged.',
      category: '__experience_category__', available: true,
    });
    const missingReflexFallback = resolvePlayableReflexId(
      '__missing_experience_reflex__', '__experience_category__');
    const failedReflexTelemetry = {};
    const failedReflexSelection = beginVoiceReflexTelemetry(
      failedReflexTelemetry, '__missing_experience_reflex__', '__experience_category__');
    let failedReflexOnsetCallbacks = 0;
    const failedReflexScheduled = await playReflex(
      failedReflexSelection.selectedId, currentVoiceTurnId + 1, {
        onAudible: () => {
          failedReflexOnsetCallbacks += 1;
          recordVoiceReflexAudible(failedReflexTelemetry, failedReflexSelection);
        },
      });
    if (failedReflexScheduled) {
      recordVoiceReflexScheduled(failedReflexTelemetry, failedReflexSelection);
    }
    const reflexFailureTelemetry = {
      scheduled: failedReflexScheduled,
      onsetCallbacks: failedReflexOnsetCallbacks,
      telemetry: {...failedReflexTelemetry},
    };
    const successfulReflexTelemetry = {};
    const successfulReflexSelection = beginVoiceReflexTelemetry(
      successfulReflexTelemetry, '__missing_experience_reflex__', '__experience_category__');
    setOracleStageState('thinking', 'Preparing acknowledgment.', '');
    let reflexOnsetCallbacks = 0;
    const reflexScheduled = await playReflex(successfulReflexSelection.selectedId, currentVoiceTurnId, {
      onAudible: () => {
        reflexOnsetCallbacks += 1;
        recordVoiceReflexAudible(successfulReflexTelemetry, successfulReflexSelection);
      },
    });
    if (reflexScheduled) {
      recordVoiceReflexScheduled(successfulReflexTelemetry, successfulReflexSelection);
    }
    const reflexQueuedPresence = oracleStage.dataset.presence;
    await waitFor(
      () => oracleStage.dataset.presence === 'reflex'
        && oracleMirror.dataset.audioReactive === 'true',
      'experience-polish:reflex-onset',
      5000,
    );
    const reflexEntry = activeAudioSources.find(entry => entry.kind === 'reflex');
    const reflexAudibleState = {
      presence: oracleStage.dataset.presence,
      audioReactive: oracleMirror.dataset.audioReactive,
      audible: reflexEntry?.audible === true,
      onsetAnalyserRetained: Boolean(reflexEntry?.onsetAnalyser),
      onsetCallbacks: reflexOnsetCallbacks,
    };
    bargeIn(null, {halfContext: false, playAck: false});
    reflexBuffers.delete('__experience_reflex__');
    reflexMetaById.delete('__experience_reflex__');
    setOracleStageState('idle', 'Ready.', '');
    resetOraclePlaybackSignal();
    await delay(80);

    const ownerTurnId = currentVoiceTurnId;
    const mouthBeforeCue = Number.parseFloat(
      oracleMirror.style.getPropertyValue('--audio-voice') || '0');
    const cue = await playOracleProceduralCue('pending', ownerTurnId, {random: () => 0.5});
    const cueSnapshot = oracleOptionalAudioState();
    // Measure the scheduling edge itself. Waiting another animation frame here
    // folds unrelated idle/playback-signal decay into the cue delta and can
    // falsely attribute ordinary mirror motion to the non-speech cue.
    const mouthAfterCueSchedule = Number.parseFloat(
      oracleMirror.style.getPropertyValue('--audio-voice') || '0');
    await delay(80);
    const cancelStartedAt = performance.now();
    bargeIn('experience-cue-cancel', {halfContext: false, playAck: false});
    const cueStopped = await waitFor(
      () => oracleOptionalAudioState().activeSources === 0
        ? Math.round(performance.now() - cancelStartedAt)
        : null,
      'experience-polish:cue-cancel',
      500,
    );

    document.documentElement.dataset.oracleMuted = 'true';
    const mutedCue = await playOracleProceduralCue('pending', currentVoiceTurnId);
    delete document.documentElement.dataset.oracleMuted;

    const originalGetPlaybackCtx = getPlaybackCtx;
    getPlaybackCtx = () => ({
      state: 'suspended',
      currentTime: 0,
      resume: async () => { throw new Error('autoplay blocked'); },
    });
    const suspendedCue = await playOracleProceduralCue('pending', currentVoiceTurnId);
    getPlaybackCtx = originalGetPlaybackCtx;

    const originalSinkGate = oracleAwaitSinkReadyBeforeStart;
    oracleAwaitSinkReadyBeforeStart = async () => {
      const error = new Error('test sink rejection');
      error.oracleExactBindingFailure = true;
      throw error;
    };
    const sinkFailureCue = await playOracleProceduralCue('pending', currentVoiceTurnId);
    oracleAwaitSinkReadyBeforeStart = originalSinkGate;

    // Exercise the production wait controller across arm -> active -> spoken
    // reflex pause -> immediate resume -> measured-onset/terminal stop. The
    // short delay is a test-only override of the same production timer.
    stopOracleThinkingBridge();
    const priorBridgeDelay = window.__ms4ThinkingBridgeDelayMs;
    const priorPulseInterval = window.__ms4ThinkingPulseIntervalMs;
    window.__ms4ThinkingBridgeDelayMs = 20;
    window.__ms4ThinkingPulseIntervalMs = 100;
    const bridgeOwner = currentVoiceTurnId;
    beginOracleThinkingBridge(bridgeOwner, {playTransition: true});
    const bridgeArmed = oracleThinkingBridgeState();
    await delay(900);
    const bridgeActive = {
      state: oracleThinkingBridgeState(),
      optionalAudio: oracleOptionalAudioState(),
    };
    const bridgePausedOk = pauseOracleThinkingBridge(bridgeOwner);
    const bridgePaused = {
      state: oracleThinkingBridgeState(),
      optionalAudio: oracleOptionalAudioState(),
    };
    const bridgeResumedOk = resumeOracleThinkingBridge(bridgeOwner, {immediate: true});
    await delay(500);
    const bridgeResumed = {
      state: oracleThinkingBridgeState(),
      optionalAudio: oracleOptionalAudioState(),
    };
    const bridgeStoppedOk = stopOracleThinkingBridge(bridgeOwner);
    const bridgeStopped = {
      state: oracleThinkingBridgeState(),
      optionalAudio: oracleOptionalAudioState(),
    };

    // A selected sink may admit asynchronously. Pause, stop, and mute must
    // invalidate work that was launched before that await; resolving the gate
    // afterward may not construct a cue or ambience graph.
    const deferredSinkRace = {};
    const priorMutedSetting = localStorage.getItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY);
    window.__ms4ThinkingPulseIntervalMs = 10_000;
    try {
      let releasePauseGate;
      let pauseGateEntries = 0;
      const pauseGate = new Promise(resolve => { releasePauseGate = resolve; });
      oracleAwaitSinkReadyBeforeStart = async () => {
        pauseGateEntries += 1;
        await pauseGate;
      };
      beginOracleThinkingBridge(bridgeOwner, {immediate: true});
      await waitFor(() => pauseGateEntries > 0, 'thinking-bridge:pause-sink-entered', 2_000);
      const pausedWhileSinkPending = pauseOracleThinkingBridge(bridgeOwner);
      releasePauseGate();
      await delay(180);
      deferredSinkRace.pause = {
        pausedWhileSinkPending,
        state: oracleThinkingBridgeState(),
        optionalAudio: oracleOptionalAudioState(),
      };
      stopOracleThinkingBridge(bridgeOwner);
      await delay(180);

      let releaseAmbientGate;
      let ambientGateEntries = 0;
      const ambientGate = new Promise(resolve => { releaseAmbientGate = resolve; });
      oracleAwaitSinkReadyBeforeStart = async () => {
        ambientGateEntries += 1;
        if (ambientGateEntries > 1) await ambientGate;
      };
      beginOracleThinkingBridge(bridgeOwner, {immediate: true});
      await waitFor(() => ambientGateEntries > 1, 'thinking-bridge:ambient-sink-entered', 2_000);
      const stoppedWhileSinkPending = stopOracleThinkingBridge(bridgeOwner);
      releaseAmbientGate();
      await delay(180);
      deferredSinkRace.stop = {
        stoppedWhileSinkPending,
        state: oracleThinkingBridgeState(),
        optionalAudio: oracleOptionalAudioState(),
      };

      let releaseMuteGate;
      let muteGateEntries = 0;
      const muteGate = new Promise(resolve => { releaseMuteGate = resolve; });
      oracleAwaitSinkReadyBeforeStart = async () => {
        muteGateEntries += 1;
        await muteGate;
      };
      beginOracleThinkingBridge(bridgeOwner, {immediate: true});
      await waitFor(() => muteGateEntries > 0, 'thinking-bridge:mute-sink-entered', 2_000);
      localStorage.setItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY, 'true');
      stopOracleOptionalAudio();
      releaseMuteGate();
      await delay(180);
      deferredSinkRace.mute = {
        state: oracleThinkingBridgeState(),
        optionalAudio: oracleOptionalAudioState(),
      };

      stopOracleThinkingBridge(bridgeOwner);
      localStorage.setItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY, 'false');
      const originalGetPlaybackCtxForResume = getPlaybackCtx;
      const priorPlaybackAudioUnlocked = playbackAudioUnlocked;
      let releaseResumeGate;
      let resumeGateEntries = 0;
      let resumeCreatedOscillators = 0;
      const resumeGate = new Promise(resolve => { releaseResumeGate = resolve; });
      const resumeCtx = {
        state: 'suspended',
        currentTime: 0,
        async resume() {
          resumeGateEntries += 1;
          await resumeGate;
          this.state = 'running';
        },
        createOscillator() {
          resumeCreatedOscillators += 1;
          throw new Error('resume race created an oscillator after mute');
        },
      };
      try {
        playbackAudioUnlocked = true;
        getPlaybackCtx = () => resumeCtx;
        oracleAwaitSinkReadyBeforeStart = async () => {};
        const resumeCuePromise = playOracleProceduralCue('pending', currentVoiceTurnId);
        await waitFor(() => resumeGateEntries > 0, 'thinking-bridge:resume-entered', 2_000);
        localStorage.setItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY, 'true');
        releaseResumeGate();
        const resumeCue = await resumeCuePromise;
        deferredSinkRace.resumeMute = {
          cueReason: resumeCue.reason,
          createdOscillators: resumeCreatedOscillators,
          optionalAudio: oracleOptionalAudioState(),
        };
      } finally {
        getPlaybackCtx = originalGetPlaybackCtxForResume;
        playbackAudioUnlocked = priorPlaybackAudioUnlocked;
      }

      localStorage.setItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY, 'false');
      const replacementCtx = getPlaybackCtx();
      const replacementCreateOscillator = replacementCtx.createOscillator;
      let replacementCreatedOscillators = 0;
      let replacementStartedOscillators = 0;
      let releaseReplacementGate;
      let replacementGateEntries = 0;
      const replacementGate = new Promise(resolve => { releaseReplacementGate = resolve; });
      replacementCtx.createOscillator = (...args) => {
        replacementCreatedOscillators += 1;
        const oscillator = replacementCreateOscillator.apply(replacementCtx, args);
        const originalStart = oscillator.start.bind(oscillator);
        oscillator.start = (...startArgs) => {
          replacementStartedOscillators += 1;
          return originalStart(...startArgs);
        };
        return oscillator;
      };
      try {
        oracleAwaitSinkReadyBeforeStart = async () => {
          replacementGateEntries += 1;
          await replacementGate;
        };
        const replacementOwner = currentVoiceTurnId;
        beginOracleThinkingBridge(replacementOwner, {immediate: true});
        await waitFor(
          () => replacementGateEntries > 0,
          'thinking-bridge:replacement-sink-entered',
          2_000,
        );
        bargeIn(null, {halfContext: false, playAck: false});
        releaseReplacementGate();
        await delay(180);
        deferredSinkRace.replacement = {
          ownerAdvanced: currentVoiceTurnId > replacementOwner,
          createdOscillators: replacementCreatedOscillators,
          startedOscillators: replacementStartedOscillators,
          state: oracleThinkingBridgeState(),
          optionalAudio: oracleOptionalAudioState(),
        };
      } finally {
        replacementCtx.createOscillator = replacementCreateOscillator;
      }

      // ABA control: an old admission must not become current again merely
      // because pause -> immediate resume returns the same bridge object to the
      // active phase. Only the resumed admission generation may construct or
      // start Web Audio nodes.
      const abaCtx = getPlaybackCtx();
      const abaCreateOscillator = abaCtx.createOscillator;
      let abaCreatedOscillators = 0;
      let abaStartedOscillators = 0;
      let abaGateEntries = 0;
      let releaseAbaOldGate;
      let releaseAbaNewGate;
      const abaOldGate = new Promise(resolve => { releaseAbaOldGate = resolve; });
      const abaNewGate = new Promise(resolve => { releaseAbaNewGate = resolve; });
      abaCtx.createOscillator = (...args) => {
        abaCreatedOscillators += 1;
        const oscillator = abaCreateOscillator.apply(abaCtx, args);
        const originalStart = oscillator.start.bind(oscillator);
        oscillator.start = (...startArgs) => {
          abaStartedOscillators += 1;
          return originalStart(...startArgs);
        };
        return oscillator;
      };
      try {
        oracleAwaitSinkReadyBeforeStart = async () => {
          abaGateEntries += 1;
          if (abaGateEntries === 1) await abaOldGate;
          else if (abaGateEntries === 2) await abaNewGate;
        };
        const abaOwner = currentVoiceTurnId;
        beginOracleThinkingBridge(abaOwner, {immediate: true});
        await waitFor(() => abaGateEntries >= 1, 'thinking-bridge:aba-old-entered', 2_000);
        const abaPaused = pauseOracleThinkingBridge(abaOwner);
        const abaResumed = resumeOracleThinkingBridge(abaOwner, {immediate: true});
        await waitFor(() => abaGateEntries >= 2, 'thinking-bridge:aba-new-entered', 2_000);

        releaseAbaOldGate();
        await delay(120);
        const afterOldRelease = {
          createdOscillators: abaCreatedOscillators,
          startedOscillators: abaStartedOscillators,
          phase: oracleThinkingBridgeState()?.phase || null,
          optionalAudio: oracleOptionalAudioState(),
        };

        releaseAbaNewGate();
        await waitFor(
          () => abaCreatedOscillators > 0 && abaStartedOscillators > 0,
          'thinking-bridge:aba-new-admitted',
          2_000,
        );
        await delay(120);
        const afterNewRelease = {
          createdOscillators: abaCreatedOscillators,
          startedOscillators: abaStartedOscillators,
          phase: oracleThinkingBridgeState()?.phase || null,
          optionalAudio: oracleOptionalAudioState(),
        };

        const stoppedAfterResume = stopOracleThinkingBridge(abaOwner);
        await delay(180);
        deferredSinkRace.pauseResume = {
          abaPaused,
          abaResumed,
          afterOldRelease,
          afterNewRelease,
          stoppedAfterResume,
          finalState: oracleThinkingBridgeState(),
          finalOptionalAudio: oracleOptionalAudioState(),
        };
      } finally {
        abaCtx.createOscillator = abaCreateOscillator;
      }
    } finally {
      oracleAwaitSinkReadyBeforeStart = originalSinkGate;
      stopOracleThinkingBridge(bridgeOwner);
      if (priorMutedSetting === null) localStorage.removeItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY);
      else localStorage.setItem(ORACLE_OPTIONAL_AUDIO_MUTED_KEY, priorMutedSetting);
      await delay(180);
    }
    if (priorBridgeDelay === undefined) delete window.__ms4ThinkingBridgeDelayMs;
    else window.__ms4ThinkingBridgeDelayMs = priorBridgeDelay;
    if (priorPulseInterval === undefined) delete window.__ms4ThinkingPulseIntervalMs;
    else window.__ms4ThinkingPulseIntervalMs = priorPulseInterval;

    setOracleStageState('idle', 'Ready.', '');
    return {
      stateGeometry,
      distinctSignatures: new Set(
        Object.values(stateGeometry).map(item => item.signature)).size,
      defaultAmbience,
      defaultAmbienceToggle,
      reducedMotionOptionalSound,
      reflexScheduled,
      missingReflexFallback,
      reflexFailureTelemetry,
      reflexSuccessTelemetry: {...successfulReflexTelemetry},
      reflexQueuedPresence,
      reflexAudibleState,
      cue,
      cueSnapshot,
      cueStoppedMs: cueStopped,
      mouthDeltaDuringCue: mouthAfterCueSchedule - mouthBeforeCue,
      mutedCue,
      suspendedCue,
      sinkFailureCue,
      thinkingBridgeLifecycle: {
        bridgeArmed,
        bridgeActive,
        bridgePausedOk,
        bridgePaused,
        bridgeResumedOk,
        bridgeResumed,
        bridgeStoppedOk,
        bridgeStopped,
      },
      deferredSinkRace,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  for (const [state, geometry] of Object.entries(experiencePolish.stateGeometry)) {
    assert.equal(geometry.presence, state, state + ' must remain a truthful visual state');
    assert.equal(geometry.pupils, 2, state + ' must retain exactly two pupils');
    assert.ok(geometry.leftEyeCells > 0 && geometry.leftEyeCells === geometry.rightEyeCells,
      state + ' must retain balanced eye cells');
    assert.equal(geometry.baseVoiceCells, 0,
      state + ' must keep the resting face continuous, with no structural mouth cells');
    const audibleOverlay = state === 'speaking' || state === 'reflex';
    assert.equal(geometry.speakingBarsOpacity > 0, audibleOverlay,
      state + ' speech-cell overlay visibility must match audible presence');
    assert.equal(geometry.neuralWaveOpacity > 0, audibleOverlay,
      state + ' neural-wave overlay visibility must match audible presence');
    assert.equal(geometry.leakingStatusCells, 0,
      state + ' must not leak procedural status-green into forehead/edge cells');
  }
  assert.equal(experiencePolish.distinctSignatures, 7,
    'idle/listening/thinking/reflex/speaking/blocked/error must remain visually distinct');
  assert.equal(experiencePolish.defaultAmbience, true, 'restrained thinking ambience defaults on');
  assert.equal(experiencePolish.defaultAmbienceToggle, true, 'visible ambience toggle matches default');
  assert.equal(experiencePolish.reducedMotionOptionalSound, false,
    'reduced-motion preference suppresses optional ambience/cues');
  assert.equal(experiencePolish.reflexScheduled, true, 'canned reflex must schedule');
  assert.deepEqual(experiencePolish.reflexFailureTelemetry, {
    scheduled: false,
    onsetCallbacks: 0,
    telemetry: {
      reflexRequestedId: '__missing_experience_reflex__',
      reflexSelectedId: '__experience_reflex__',
      reflexScheduledId: null,
      reflexPlayedId: null,
      reflexFallbackSelected: true,
      reflexFallbackUsed: false,
    },
  }, 'a stale or failed fallback must remain selected-only and never claim playback');
  assert.deepEqual(experiencePolish.reflexSuccessTelemetry, {
    reflexRequestedId: '__missing_experience_reflex__',
    reflexSelectedId: '__experience_reflex__',
    reflexScheduledId: '__experience_reflex__',
    reflexPlayedId: '__experience_reflex__',
    reflexFallbackSelected: true,
    reflexFallbackUsed: true,
  }, 'fallback-used evidence must commit only after measured audible onset');
  assert.notEqual(experiencePolish.reflexQueuedPresence, 'reflex',
    'canned reflex must not claim audible state while merely queued');
  assert.deepEqual(experiencePolish.reflexAudibleState, {
    presence: 'reflex',
    audioReactive: 'true',
    audible: true,
    onsetAnalyserRetained: true,
    onsetCallbacks: 1,
  }, 'canned reflex must enter and retain its audio-reactive state only after measured onset');
  assert.equal(experiencePolish.cue.played, true, 'dedicated pending cue must schedule');
  assert.equal(experiencePolish.cue.semantic, 'cue', 'pending cue uses non-speech analyser semantics');
  assert.ok(experiencePolish.cue.peakGain <= 0.03, 'pending cue stays sparse and low-volume');
  assert.equal(experiencePolish.cueSnapshot.speechLikeSources, 0,
    'non-speech pending cues must not enter the mouth analyser');
  assert.ok(experiencePolish.cueStoppedMs <= 250, 'barge-in must stop cues inside the 250ms budget');
  assert.ok(Math.abs(experiencePolish.mouthDeltaDuringCue) < 0.02,
    'non-speech cues must not drive mouth movement');
  assert.equal(experiencePolish.mutedCue.reason, 'optional_sound_disabled',
    'muted optional sound must degrade to silence');
  assert.equal(experiencePolish.suspendedCue.reason, 'audio_context_suspended',
    'an unresumable AudioContext must degrade to silence');
  assert.equal(experiencePolish.sinkFailureCue.reason, 'sink_failure',
    'a selected-sink rejection must fail closed without default output');
  assert.deepEqual(experiencePolish.thinkingBridgeLifecycle.bridgeArmed, {
    turnId: experiencePolish.thinkingBridgeLifecycle.bridgeArmed.turnId,
    phase: 'armed',
    timerArmed: true,
    pulseTimerArmed: false,
    pulseCount: 0,
    transitionPlayed: true,
  }, 'capture handoff must arm one turn-owned delayed bridge and transition cue');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeActive.state.phase, 'active',
    'the delayed bridge must become active during the wait');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeActive.optionalAudio.ambienceActive, true,
    'the active wait bridge must own restrained non-speech ambience');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeActive.state.pulseTimerArmed, true,
    'the active wait bridge must retain one turn-owned periodic pulse timer');
  assert.ok(experiencePolish.thinkingBridgeLifecycle.bridgeActive.state.pulseCount >= 2,
    'the active wait bridge must emit repeated perceptible progress pulses');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgePausedOk, true,
    'an audible spoken reflex must be able to pause its wait bridge');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgePaused.state.phase, 'paused',
    'spoken reflex pause must retain turn ownership without a live timer');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgePaused.state.pulseTimerArmed, false,
    'spoken reflex pause must cancel the periodic progress pulse timer');
  assert.deepEqual(experiencePolish.thinkingBridgeLifecycle.bridgePaused.optionalAudio,
    {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
    'spoken reflex pause must duck all non-speech waiting audio');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeResumedOk, true,
    'reflex drain must be able to resume the same bridge immediately');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeResumed.state.phase, 'active',
    'the resumed bridge must remain active until generated audio is measured');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeResumed.optionalAudio.ambienceActive, true,
    'reflex drain must not create a second silent model-wait gap');
  assert.ok(
    experiencePolish.thinkingBridgeLifecycle.bridgeResumed.state.pulseCount
      > experiencePolish.thinkingBridgeLifecycle.bridgePaused.state.pulseCount,
    'reflex drain must resume periodic progress pulses on the same turn owner');
  assert.equal(experiencePolish.thinkingBridgeLifecycle.bridgeStoppedOk, true,
    'measured onset or terminal cleanup must stop the owned bridge');
  assert.deepEqual(experiencePolish.thinkingBridgeLifecycle.bridgeStopped, {
    state: null,
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'stopping the bridge must leave no waiting-audio residue');
  assert.deepEqual(experiencePolish.deferredSinkRace.pause, {
    pausedWhileSinkPending: true,
    state: {
      turnId: experiencePolish.deferredSinkRace.pause.state.turnId,
      phase: 'paused',
      timerArmed: false,
      pulseTimerArmed: false,
      pulseCount: 0,
      transitionPlayed: false,
    },
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'pause during sink admission must not resurrect a pending cue');
  assert.deepEqual(experiencePolish.deferredSinkRace.stop, {
    stoppedWhileSinkPending: true,
    state: null,
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'stop during ambience sink admission must not resurrect the pad');
  assert.deepEqual(experiencePolish.deferredSinkRace.mute.optionalAudio,
    {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
    'mute during sink admission must remain silent after admission resolves');
  assert.deepEqual(experiencePolish.deferredSinkRace.resumeMute, {
    cueReason: 'optional_sound_disabled',
    createdOscillators: 0,
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'mute during AudioContext resume must win before node construction');
  assert.deepEqual(experiencePolish.deferredSinkRace.replacement, {
    ownerAdvanced: true,
    createdOscillators: 0,
    startedOscillators: 0,
    state: null,
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'replacement while sink-blocked must leave no stale bridge, pulse, or audio node');
  assert.equal(experiencePolish.deferredSinkRace.pauseResume.abaPaused, true);
  assert.equal(experiencePolish.deferredSinkRace.pauseResume.abaResumed, true);
  assert.deepEqual(experiencePolish.deferredSinkRace.pauseResume.afterOldRelease, {
    createdOscillators: 0,
    startedOscillators: 0,
    phase: 'active',
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'the pre-pause sink admission must remain stale after the same bridge resumes');
  assert.ok(
    experiencePolish.deferredSinkRace.pauseResume.afterNewRelease.createdOscillators > 0,
    'the resumed admission generation must be allowed to construct its own audio graph',
  );
  assert.ok(
    experiencePolish.deferredSinkRace.pauseResume.afterNewRelease.startedOscillators > 0,
    'the resumed admission generation must be allowed to start its own audio graph',
  );
  assert.equal(experiencePolish.deferredSinkRace.pauseResume.afterNewRelease.phase, 'active');
  assert.equal(
    experiencePolish.deferredSinkRace.pauseResume.afterNewRelease.optionalAudio.ambienceActive,
    true,
  );
  assert.equal(experiencePolish.deferredSinkRace.pauseResume.stoppedAfterResume, true);
  assert.equal(experiencePolish.deferredSinkRace.pauseResume.finalState, null);
  assert.deepEqual(
    experiencePolish.deferredSinkRace.pauseResume.finalOptionalAudio,
    {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  );

  // Principal v5 fail-first probe: instrument REAL Web Audio nodes, then drive
  // production stop/cancel/replacement/failure/repeat paths. Logical source
  // counts are reported separately and cannot satisfy the disconnect verdict.
  experiencePolish.audioGraphLifecycle = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    stopOracleOptionalAudio();
    if (activeAudioSources.length) bargeIn(null, {halfContext: false, playAck: false});
    releaseOraclePlaybackAnalyserIfIdle();
    if (oracleCueAnalyser) {
      try { oracleCueAnalyser.disconnect(); } catch (_error) {}
    }
    oracleCueAnalyser = null;
    oracleCueNode = null;

    const ctx = getPlaybackCtx();
    await ctx.resume();
    const created = [];
    const methods = {
      createAnalyser: 'analyser',
      createGain: 'gain',
      createOscillator: 'oscillator',
      createBiquadFilter: 'filter',
      createBufferSource: 'source',
    };
    const originals = {};
    for (const [method, type] of Object.entries(methods)) {
      originals[method] = ctx[method].bind(ctx);
      ctx[method] = (...args) => {
        const node = originals[method](...args);
        const disconnect = node.disconnect.bind(node);
        const record = {type, node, disconnectCalls: 0};
        node.disconnect = (...disconnectArgs) => {
          record.disconnectCalls += 1;
          return disconnect(...disconnectArgs);
        };
        created.push(record);
        return node;
      };
    }
    const mark = () => created.length;
    const verdict = start => {
      const records = created.slice(start);
      return {
        created: records.length,
        types: records.map(record => record.type),
        disconnected: records.filter(record => record.disconnectCalls > 0).length,
        allDisconnected: records.length > 0
          && records.every(record => record.disconnectCalls > 0),
      };
    };

    let speechStop;
    let auxiliarySpeechFailure;
    let gainBackedAuxiliarySpeechFailure;
    let generatedSpeechFailure;
    let reflexStartFailure;
    let announcementStartFailure;
    let eggStartFailure;
    let eggConstructionFailure;
    let cueCancellation;
    let ambienceStop;
    let ambienceStopFailure;
    let ambienceReplacement;
    let ambienceFailure;
    let ambienceOscillatorFailure;
    let repeatedCycles;
    try {
      let start = mark();
      ensureOraclePlaybackAnalyser(ctx);
      releaseOraclePlaybackAnalyserIfIdle();
      speechStop = verdict(start);

      start = mark();
      const auxSource = ctx.createBufferSource();
      auxSource.buffer = ctx.createBuffer(1, 32, ctx.sampleRate);
      const auxEntry = {
        source: auxSource,
        turnId: currentVoiceTurnId,
        kind: 'announcement',
      };
      activeAudioSources.push(auxEntry);
      const realCreateAnalyser = ctx.createAnalyser;
      ctx.createAnalyser = (...args) => {
        const analyser = realCreateAnalyser(...args);
        analyser.connect = () => { throw new Error('principal-aux-connect-failure'); };
        return analyser;
      };
      try {
        watchOracleAuxSpeechOnset(ctx, auxSource, auxEntry);
      } catch (_error) { /* expected fail-first path */ }
      ctx.createAnalyser = realCreateAnalyser;
      auxiliarySpeechFailure = verdict(start);
      activeAudioSources = activeAudioSources.filter(entry => entry !== auxEntry);
      cleanupOracleAudioEntry(auxEntry);
      releaseOraclePlaybackAnalyserIfIdle();

      start = mark();
      const gainBackedSource = ctx.createBufferSource();
      gainBackedSource.buffer = ctx.createBuffer(1, 32, ctx.sampleRate);
      const gainBackedOutput = ctx.createGain();
      gainBackedSource.connect(gainBackedOutput);
      const gainBackedEntry = {
        source: gainBackedSource,
        gain: gainBackedOutput,
        turnId: currentVoiceTurnId,
        kind: 'egg',
      };
      activeAudioSources.push(gainBackedEntry);
      const realGainBackedCreateAnalyser = ctx.createAnalyser;
      ctx.createAnalyser = (...args) => {
        const analyser = realGainBackedCreateAnalyser(...args);
        analyser.connect = () => {
          throw new Error('principal-gain-backed-aux-connect-failure');
        };
        return analyser;
      };
      try {
        watchOracleAuxSpeechOnset(
          ctx,
          gainBackedOutput,
          gainBackedEntry,
        );
      } catch (_error) { /* expected fail-first path */ }
      ctx.createAnalyser = realGainBackedCreateAnalyser;
      gainBackedAuxiliarySpeechFailure = verdict(start);
      activeAudioSources = activeAudioSources.filter(
        entry => entry !== gainBackedEntry,
      );
      cleanupOracleAudioEntry(gainBackedEntry);
      releaseOraclePlaybackAnalyserIfIdle();

      start = mark();
      const generatedTurn = {
        turnId: currentVoiceTurnId,
        aborted: false,
        audibleFinalized: false,
        terminalKind: null,
        onsetAnalyser: null,
        onsetData: null,
        onsetMonitorFrame: 0,
        telemetry: {
          scheduledChunks: 0,
          firstScheduledMs: null,
          firstScheduledContextTime: null,
          firstNonSilentOnsetMs: null,
          onsetRms: null,
          underflowGapCount: 0,
          underflowGapTotalMs: 0,
          underflowGapMaxMs: 0,
        },
      };
      const realCreateBufferSource = ctx.createBufferSource;
      ctx.createBufferSource = (...args) => {
        const source = realCreateBufferSource(...args);
        source.start = () => { throw new Error('principal-generated-start-failure'); };
        return source;
      };
      try {
        await scheduleDecodedChunk(
          ctx.createBuffer(1, 32, ctx.sampleRate),
          currentVoiceTurnId,
          generatedTurn,
        );
      } catch (_error) { /* expected fail-first path */ }
      ctx.createBufferSource = realCreateBufferSource;
      generatedSpeechFailure = verdict(start);
      cleanupVoiceTurnOnset(generatedTurn);
      releaseOraclePlaybackAnalyserIfIdle();

      const cleanupFailedSpeech = () => {
        for (const entry of activeAudioSources.splice(0)) {
          cleanupOracleAudioEntry(entry);
        }
        releaseOraclePlaybackAnalyserIfIdle();
      };
      const failNextSourceStart = () => {
        const real = ctx.createBufferSource;
        ctx.createBufferSource = (...args) => {
          const source = real(...args);
          source.start = () => {
            throw new Error('principal-auxiliary-start-failure');
          };
          return source;
        };
        return real;
      };

      start = mark();
      const reflexFailureId = '__principal_reflex_start_failure__';
      reflexBuffers.set(reflexFailureId, ctx.createBuffer(1, 32, ctx.sampleRate));
      let restoreCreateBufferSource = failNextSourceStart();
      await playReflex(reflexFailureId, currentVoiceTurnId);
      ctx.createBufferSource = restoreCreateBufferSource;
      reflexBuffers.delete(reflexFailureId);
      reflexStartFailure = verdict(start);
      cleanupFailedSpeech();

      start = mark();
      const originalFetch = window.fetch;
      window.fetch = url => String(url) === '/voice/synthesize/stream'
        ? Promise.resolve(new Response(
          'event: audio_chunk\\ndata: {"index":0,"text":"start failure","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
            + 'event: done\\ndata: {"completed":true,"audio_chunks":1}\\n\\n', {
            status: 200,
            headers: {'Content-Type': 'text/event-stream'},
          }))
        : originalFetch(url);
      restoreCreateBufferSource = failNextSourceStart();
      await speakAnnouncement('Principal start failure.', currentVoiceTurnId);
      ctx.createBufferSource = restoreCreateBufferSource;
      window.fetch = originalFetch;
      announcementStartFailure = verdict(start);
      cleanupFailedSpeech();

      start = mark();
      const savedEggClipBuffer = eggClipBuffer;
      const savedEggFallback = reflexBuffers.get('egg_honorable');
      eggClipBuffer = ctx.createBuffer(1, 32, ctx.sampleRate);
      reflexBuffers.delete('egg_honorable');
      restoreCreateBufferSource = failNextSourceStart();
      await playEggClip('/unused-principal-egg', 1, currentVoiceTurnId);
      ctx.createBufferSource = restoreCreateBufferSource;
      eggClipBuffer = savedEggClipBuffer;
      if (savedEggFallback) reflexBuffers.set('egg_honorable', savedEggFallback);
      eggStartFailure = verdict(start);
      cleanupFailedSpeech();

      start = mark();
      const savedConstructionEggClipBuffer = eggClipBuffer;
      const savedConstructionEggFallback = reflexBuffers.get('egg_honorable');
      eggClipBuffer = ctx.createBuffer(1, 32, ctx.sampleRate);
      reflexBuffers.delete('egg_honorable');
      const realConstructionCreateBufferSource = ctx.createBufferSource;
      ctx.createBufferSource = (...args) => {
        const source = realConstructionCreateBufferSource(...args);
        source.connect = () => {
          throw new Error('principal-egg-construction-connect-failure');
        };
        return source;
      };
      await playEggClip('/unused-principal-egg-construction', 1, currentVoiceTurnId);
      ctx.createBufferSource = realConstructionCreateBufferSource;
      eggClipBuffer = savedConstructionEggClipBuffer;
      if (savedConstructionEggFallback) {
        reflexBuffers.set('egg_honorable', savedConstructionEggFallback);
      }
      eggConstructionFailure = verdict(start);
      cleanupFailedSpeech();

      start = mark();
      await playOracleProceduralCue('pending', currentVoiceTurnId, {random: () => 0.5});
      bargeIn('principal-cue-cancel', {halfContext: false, playAck: false});
      await delay(30);
      cueCancellation = verdict(start);

      start = mark();
      await startThinkingAmbient(currentVoiceTurnId);
      stopThinkingAmbient();
      await delay(180);
      ambienceStop = verdict(start);

      start = mark();
      await startThinkingAmbient(currentVoiceTurnId);
      const ambienceMaster = created.slice(start)
        .find(record => record.type === 'gain')?.node;
      if (ambienceMaster) {
        ambienceMaster.gain.cancelScheduledValues = () => {
          throw new Error('principal-ambience-stop-failure');
        };
      }
      stopThinkingAmbient();
      await delay(180);
      ambienceStopFailure = verdict(start);

      start = mark();
      await startThinkingAmbient(currentVoiceTurnId);
      bargeIn('principal-ambience-replacement', {halfContext: false, playAck: false});
      await delay(180);
      ambienceReplacement = verdict(start);

      start = mark();
      const realCreateBiquadFilter = ctx.createBiquadFilter;
      ctx.createBiquadFilter = () => { throw new Error('principal-partial-graph-failure'); };
      await startThinkingAmbient(currentVoiceTurnId);
      ctx.createBiquadFilter = realCreateBiquadFilter;
      await delay(30);
      ambienceFailure = verdict(start);

      start = mark();
      const realCreateOscillator = ctx.createOscillator;
      let padOscillatorsCreated = 0;
      ctx.createOscillator = (...args) => {
        const oscillator = realCreateOscillator(...args);
        if (padOscillatorsCreated++ === 1) {
          oscillator.start = () => {
            throw new Error('principal-mid-oscillator-start-failure');
          };
        }
        return oscillator;
      };
      await startThinkingAmbient(currentVoiceTurnId);
      ctx.createOscillator = realCreateOscillator;
      await delay(30);
      ambienceOscillatorFailure = verdict(start);

      start = mark();
      for (let cycle = 0; cycle < 3; cycle += 1) {
        await startThinkingAmbient(currentVoiceTurnId);
        stopThinkingAmbient();
        await delay(180);
      }
      repeatedCycles = verdict(start);
    } finally {
      stopOracleOptionalAudio();
      for (const [method, original] of Object.entries(originals)) ctx[method] = original;
    }
    return {
      speechStop,
      auxiliarySpeechFailure,
      gainBackedAuxiliarySpeechFailure,
      generatedSpeechFailure,
      reflexStartFailure,
      announcementStartFailure,
      eggStartFailure,
      eggConstructionFailure,
      cueCancellation,
      ambienceStop,
      ambienceStopFailure,
      ambienceReplacement,
      ambienceFailure,
      ambienceOscillatorFailure,
      repeatedCycles,
      logicalState: oracleOptionalAudioState(),
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });

  experiencePolish.muteControl = await cdp.evaluate(`(async () => {
    const control = document.getElementById('settingsOracleOptionalSound');
    if (!control) return {reachable: false, persisted: null, cueReason: null};
    control.checked = false;
    control.dispatchEvent(new Event('change', {bubbles: true}));
    const cue = await playOracleProceduralCue('pending', currentVoiceTurnId);
    const result = {
      reachable: true,
      persisted: localStorage.getItem(ORACLE_OPTIONAL_SOUND_KEY),
      cueReason: cue.reason,
    };
    control.checked = true;
    control.dispatchEvent(new Event('change', {bubbles: true}));
    return result;
  })()`);

  await cdp.call('Emulation.setEmulatedMedia', {
    features: [{name: 'prefers-reduced-motion', value: 'reduce'}],
  });
  experiencePolish.reducedMotionVisual = await cdp.evaluate(`(() => ({
    stageBefore: getComputedStyle(oracleStage, '::before').animationName,
    stageAfter: getComputedStyle(oracleStage, '::after').animationName,
    activeCell: getComputedStyle(oracleMirror.querySelector('.mirror-cell.is-active')).animationName,
    statusCell: getComputedStyle(oracleMirror.querySelector('.mirror-cell.is-status')).animationName,
    waveCell: getComputedStyle(oracleMirror.querySelector('.mirror-wave-cell')).animationName,
  }))()`);
  await cdp.call('Emulation.setEmulatedMedia', {features: []});

  const collectLayout = async () => cdp.evaluate(`(() => {
    const rect = element => {
      const box = element.getBoundingClientRect();
      return {
        left: box.left, top: box.top, right: box.right, bottom: box.bottom,
        width: box.width, height: box.height,
      };
    };
    const stage = rect(oracleStage);
    const mirror = rect(oracleMirror);
    const consoleBox = rect(document.querySelector('.oracle-console'));
    const action = rect(document.querySelector('.oracle-action-stack'));
    const controls = rect(document.querySelector('.front-door-control-row'));
    const actionVisible = action.width > 0 && action.height > 0;
    return {
      viewport: {width: innerWidth, height: innerHeight},
      horizontalOverflowPx: Math.max(0, document.documentElement.scrollWidth - innerWidth),
      stageInsideViewportWidth: stage.left >= -0.5 && stage.right <= innerWidth + 0.5,
      mirrorInsideStage: mirror.left >= stage.left - 0.5 && mirror.right <= stage.right + 0.5,
      mirrorConsoleOverlap: mirror.bottom - consoleBox.top,
      consoleActionOverlap: actionVisible ? consoleBox.bottom - action.top : 0,
      stageControlsOverlap: stage.bottom - controls.top,
    };
  })()`);
  const desktopLayout = await collectLayout();
  await cdp.call('Emulation.setDeviceMetricsOverride', {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await cdp.evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const mobileLayout = await collectLayout();
  let oracleMobilePng = null;
  if (evidenceDir) {
    await cdp.evaluate(`window.scrollTo(0, 0)`);
    const mobileShot = await cdp.call('Page.captureScreenshot', {
      format: 'png',
      fromSurface: true,
      captureBeyondViewport: false,
    });
    oracleMobilePng = Buffer.from(mobileShot.data, 'base64');
    experiencePolish.mobileArtifact = {
      width: oracleMobilePng.readUInt32BE(16),
      height: oracleMobilePng.readUInt32BE(20),
    };
  }
  await cdp.call('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await cdp.evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  experiencePolish.layouts = {desktop: desktopLayout, mobile390x844: mobileLayout};
  for (const [label, layout] of Object.entries(experiencePolish.layouts)) {
    assert.equal(layout.horizontalOverflowPx, 0, label + ' must not overflow horizontally');
    assert.equal(layout.stageInsideViewportWidth, true, label + ' stage must remain inside viewport width');
    assert.equal(layout.mirrorInsideStage, true, label + ' mirror must stay inside stage');
    assert.ok(layout.mirrorConsoleOverlap <= 0.5, label + ' mirror/console must not overlap');
    assert.ok(layout.consoleActionOverlap <= 0.5, label + ' console/actions must not overlap');
    assert.ok(layout.stageControlsOverlap <= 0.5, label + ' stage/controls must not overlap');
  }

  // Finding 10 test hook: hang with Chrome alive so the parent process-tree
  // cleanup boundary (Windows Job Object / POSIX process group) must reap Node
  // AND its Chrome descendant. Never used by acceptance runs.
  if (process.env.MS4_BROWSER_TEST_HANG === '1') {
    process.stdout.write(`{"hang":true,"debugPort":${debugPort},"profileDir":${JSON.stringify(profileDir)}}\n`);
    await new Promise(() => {});
  }

  let oracleIdlePng = null;
  if (evidenceDir) {
    await cdp.call('Page.bringToFront');
    await cdp.evaluate(`(() => {
      window.scrollTo(0, 0);
      setOracleStageState('idle', 'Ready.', 'Deterministic idle-state evidence.');
    })()`);
    const idleClip = await cdp.evaluate(`(() => {
      const r = oracleStage.getBoundingClientRect();
      return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height, scale: 1};
    })()`);
    await cdp.evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
    const idleShot = await cdp.call('Page.captureScreenshot', {
      format: 'png',
      fromSurface: true,
      captureBeyondViewport: true,
      clip: idleClip,
    });
    oracleIdlePng = Buffer.from(idleShot.data, 'base64');
  }

  visual = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const cssNumber = (element, name) => Number.parseFloat(element.style.getPropertyValue(name) || '0');
    resetOraclePlaybackSignal();
    setOracleStageState('speaking', 'Oracle browser proof.', 'Audio analyser active.');
    await delay(140);
    const before = {
      reactive: oracleMirror.dataset.audioReactive,
      voice: Number.parseFloat(oracleMirror.style.getPropertyValue('--audio-voice') || '0'),
      maxBarScale: Math.max(...[...oracleMirror.querySelectorAll('.mirror-speaking-bar')].map(bar => cssNumber(bar, '--audio-scale'))),
    };
    const ctx = getPlaybackCtx();
    await ctx.resume();
    const binary = atob('${proofAudioBase64}');
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(oraclePlaybackNodeForSource(ctx));
    source.start();
    window.__ms4OracleProofAudio = { source };
    await delay(320);
    const speakingCells = [...oracleMirror.querySelectorAll('.mirror-speaking-cell')];
    const signalNodes = speakingCells.filter(cell => cell.classList.contains('is-signal-node'));
    const breathNodes = speakingCells.filter(cell => cell.classList.contains('is-hivemind-breath'));
    const centerSignalNodes = signalNodes.filter(cell => Number.parseFloat(cell.style.getPropertyValue('--column-distance') || '0') < 6);
    const mouthBox = oracleMirror.querySelector('.mirror-speaking-bars').getBBox();
    const waveBox = oracleMirror.querySelector('.mirror-wave-cells').getBBox();
    const eyeCells = [...oracleMirror.querySelectorAll('.mirror-cell.shape-eye')];
    const eyeBoxes = eyeCells.map(cell => cell.getBBox());
    const eyeCenterY = eyeBoxes.reduce((sum, box) => sum + box.y + box.height / 2, 0) / eyeBoxes.length;
    const pupilCount = oracleMirror.querySelectorAll('.mirror-cell.shape-pupil').length;
    const barScales = [...oracleMirror.querySelectorAll('.mirror-speaking-bar')].map(bar => cssNumber(bar, '--audio-scale'));
    const signalStyle = getComputedStyle(signalNodes[0]);
    const breathStyle = getComputedStyle(breathNodes[0]);
    return {
      before, proofAudioKind: 'TMR canned speech WAV', proofAudioDuration: audioBuffer.duration,
      reactive: oracleMirror.dataset.audioReactive, playbackActive: oraclePlaybackSignal.active,
      level: Number.parseFloat(oracleMirror.style.getPropertyValue('--audio-level') || '0'),
      voice: Number.parseFloat(oracleMirror.style.getPropertyValue('--audio-voice') || '0'),
      maxBarScale: Math.max(...barScales), minBarScale: Math.min(...barScales),
      speakingCellCount: speakingCells.length, waveCellCount: oracleMirror.querySelectorAll('.mirror-wave-cell').length,
      signalNodeCount: signalNodes.length, centerSignalNodeCount: centerSignalNodes.length,
      eyeCellCount: eyeCells.length, pupilCount, eyeCenterY,
      mouthBox: { x: mouthBox.x, y: mouthBox.y, width: mouthBox.width, height: mouthBox.height },
      waveBox: { x: waveBox.x, y: waveBox.y, width: waveBox.width, height: waveBox.height },
      signalStroke: signalStyle.stroke, breathStroke: breathStyle.stroke,
      literalMouthElement: Boolean(oracleMirror.querySelector('[class*="mouth"], .mirror-speaking-bars path, .mirror-speaking-bars circle')),
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(visual.before.reactive, 'false');
  assert.equal(visual.reactive, 'true');
  assert.equal(visual.playbackActive, true);
  assert.ok(visual.voice > visual.before.voice, 'real analyser signal must raise the voice band');
  assert.ok(visual.maxBarScale > visual.before.maxBarScale, 'real analyser signal must move the mouth-cell bars');
  assert.ok(visual.speakingCellCount >= 100);
  assert.ok(visual.waveCellCount >= 35);
  assert.equal(visual.centerSignalNodeCount, 0);
  assert.ok(visual.eyeCellCount >= 32 && visual.pupilCount === 2);
  assert.equal(visual.literalMouthElement, false);

  let hiveCellsPng = null;
  if (evidenceDir) {
    await cdp.call('Page.bringToFront');
    const clip = await cdp.evaluate(`(() => { window.scrollTo(0,0); const r = oracleStage.getBoundingClientRect(); return { x: r.x+scrollX, y: r.y+scrollY, width: r.width, height: r.height, scale: 1 }; })()`);
    await cdp.evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
    const shot = await cdp.call('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip });
    hiveCellsPng = Buffer.from(shot.data, 'base64');
  }

  await cdp.evaluate(`(async () => {
    const proof = window.__ms4OracleProofAudio;
    if (proof) { try { proof.source.stop(); } catch (_e) {} try { proof.source.disconnect(); } catch (_e) {} }
    delete window.__ms4OracleProofAudio;
    await new Promise(resolve => setTimeout(resolve, 120));
    setOracleStageState('idle', 'Ready.', '');
  })()`);

  // ---- Selected Face-model admission: catalog, exact warm, frozen route. ---
  facePrewarm = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const originalPlayRandomReflex = window.playRandomReflex;
    const originalCatalogPromise = modelCatalogReadyPromise;
    const originalCatalogReady = modelCatalogReady;
    const originalOptions = modelSelect.innerHTML;
    const originalValue = modelSelect.value;
    const encoder = new TextEncoder();
    const check = (condition, message) => { if (!condition) throw new Error(message); };
    const equal = (actual, expected, message) => check(
      actual === expected,
      (message || 'values differ') + ': actual=' + JSON.stringify(actual) + ' expected=' + JSON.stringify(expected),
    );
    const bounded = (promise, label, timeoutMs = 5000) => Promise.race([
      promise,
      new Promise((_resolve, reject) => setTimeout(() => reject(new Error(label + ' timed out')), timeoutMs)),
    ]);
    const expectReject = async (promise, pattern) => {
      try {
        await promise;
      } catch (error) {
        check(pattern.test(String(error && (error.message || error))), 'unexpected rejection: ' + String(error));
        return;
      }
      throw new Error('expected promise rejection');
    };
    let voiceCalls = 0, synthCalls = 0, reflexCalls = 0;
    let prewarmCalls = [];
    let prewarmMode = 'catalog';
    let resolveCatalog = null, resolveCatalogWarm = null, resolveA = null, resolveB = null;
    let resolveAutoInvalidate = null, resolveStaleAutoInvalidate = null;
    let catalogWarmStarted = false;
    let catalogObservedTurn = null;
    const voiceUrls = [];
    const jsonResponse = (status, data) => Promise.resolve({
      ok: status >= 200 && status < 300,
      status,
      statusText: status >= 200 && status < 300 ? 'OK' : 'Unavailable',
      json: () => Promise.resolve(data),
    });
    const exactPayload = (model, generation) => ({
      warmed: true,
      requested_model: model,
      resolved_requested_model: model,
      effective_model: model,
      fallback_used: false,
      reply_len: 5,
      admission_profile: 'ms4-face-voice-production-uncached-700-v1',
      first_token_ms: 1200,
      latency_budget_ms: 15000,
      latency_admitted: true,
      generation,
    });
    const exactInvalidationPayload = generation => ({
      warmed: false,
      invalidated: true,
      automatic: true,
      operation: 'invalidate',
      generation,
    });
    const voiceResponse = model => {
      const text = 'event: transcript\\ndata: {"text":"prewarm browser proof","asr_ms":1}\\n\\n' +
        'event: text_delta\\ndata: {"text":"Exact warm route ' + model + '."}\\n\\n' +
        'event: done\\ndata: {"session_id":"prewarm-proof","reply_text":"Exact warm route.","metrics":{"audio_chunks":0,"audio_client_written":0,"audio_errors":0,"total_ms":5}}\\n\\n';
      const bytes = encoder.encode(text);
      let sent = false;
      return Promise.resolve({
        ok: true,
        statusText: 'OK',
        body: {getReader: () => ({
          read() {
            if (!sent) { sent = true; return Promise.resolve({value: bytes, done: false}); }
            return Promise.resolve({value: undefined, done: true});
          },
          cancel() { return Promise.resolve(); },
        })},
      });
    };
    window.playRandomReflex = (...args) => {
      reflexCalls += 1;
      return Promise.resolve(false);
    };
    window.fetch = (url, options = {}) => {
      const target = String(url);
      if (target === '/voice/prewarm/face') {
        const prewarmBody = JSON.parse(String(options.body || '{}'));
        const model = prewarmBody.model;
        const generation = prewarmBody.generation;
        const operation = prewarmBody.operation || 'prewarm';
        prewarmCalls.push({model, generation, operation, client_id: prewarmBody.client_id, mode: prewarmMode});
        if (operation === 'invalidate') {
          if (prewarmMode === 'auto-pending') {
            return new Promise(resolve => {
              resolveAutoInvalidate = () => resolve(jsonResponse(200, exactInvalidationPayload(generation)));
            }).then(value => value);
          }
          if (prewarmMode === 'auto-stale') {
            // Deliberately ignore AbortSignal: a late server receipt must still
            // be unable to authorize Auto after a newer B generation exists.
            return new Promise(resolve => {
              resolveStaleAutoInvalidate = () => resolve(jsonResponse(200, exactInvalidationPayload(generation)));
            }).then(value => value);
          }
          if (prewarmMode === 'auto-fail') {
            return jsonResponse(503, {
              warmed: false,
              invalidated: false,
              fail_closed: true,
              error: 'automatic invalidation unavailable',
              generation,
            });
          }
          if (prewarmMode === 'auto-mismatch') {
            return jsonResponse(200, {
              ...exactInvalidationPayload(generation),
              invalidated: false,
            });
          }
          if (prewarmMode === 'auto-stall') {
            return new Promise((_resolve, reject) => {
              if (options.signal) {
                options.signal.addEventListener(
                  'abort',
                  () => reject(new DOMException('Aborted', 'AbortError')),
                  {once: true},
                );
              }
            });
          }
          return jsonResponse(200, exactInvalidationPayload(generation));
        }
        if (prewarmMode === 'catalog') {
          catalogWarmStarted = true;
          return new Promise(resolve => { resolveCatalogWarm = () => resolve(jsonResponse(200, exactPayload(model, generation))); })
            .then(value => value);
        }
        if (prewarmMode === 'race-a') {
          return new Promise((resolve, reject) => {
            resolveA = () => resolve(jsonResponse(200, exactPayload(model, generation)));
            if (options.signal) options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {once: true});
          }).then(value => value);
        }
        if (prewarmMode === 'race-b') {
          return new Promise(resolve => { resolveB = () => resolve(jsonResponse(200, exactPayload(model, generation))); })
            .then(value => value);
        }
        if (prewarmMode === 'fail') {
          return jsonResponse(503, {warmed: false, fail_closed: true, error: 'selected model unavailable'});
        }
        if (prewarmMode === 'mismatch') {
          return jsonResponse(200, exactPayload('wrong:4b', generation));
        }
        if (prewarmMode === 'latency-fail') {
          return jsonResponse(503, {
            ...exactPayload(model, generation),
            warmed: false,
            first_token_ms: 15001,
            latency_admitted: false,
            fail_closed: true,
            error: 'Selected Face model missed the production latency admission budget.',
          });
        }
        return jsonResponse(200, exactPayload(model, generation));
      }
      if (target.startsWith('/voice/turn/stream')) {
        voiceCalls += 1;
        voiceUrls.push(target);
        return voiceResponse(new URL(target, location.href).searchParams.get('model') || 'auto');
      }
      if (target.startsWith('/voice/synthesize')) synthCalls += 1;
      return originalFetch(url, options);
    };

    try {
      modelSelect.innerHTML = '<option value="">Auto</option>' +
        '<option value="restored:4b">Restored</option>' +
        '<option value="race-a:4b">A</option>' +
        '<option value="race-b:4b">B</option>' +
        '<option value="failure:4b">Failure</option>';
      modelSelect.value = '';
      selectedFacePrewarmState = null;
      const initialAutoPrewarms = prewarmCalls.length;
      const initialAutoAdmission = await startSelectedFaceModelPrewarm('initial automatic selection');
      equal(initialAutoAdmission.skipped, true, 'initial untouched Auto admission');
      equal(prewarmCalls.length, initialAutoPrewarms, 'initial untouched Auto must not request invalidation');
      modelCatalogReady = false;
      modelCatalogReadyPromise = new Promise(resolve => {
        resolveCatalog = () => {
          modelSelect.value = 'restored:4b';
          void startSelectedFaceModelPrewarm('restored selection').catch(() => {});
          modelCatalogReady = true;
          resolve();
        };
      });

      const earlyVoice = submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {
          durationSec: 0.1,
          sourceLabel: 'catalog-race',
          onTurnStateCreated: turn => { catalogObservedTurn = turn; },
        },
      );
      await delay(30);
      const beforeCatalog = {voiceCalls, prewarmCalls: prewarmCalls.length};
      resolveCatalog();
      await waitFor(() => catalogWarmStarted, 'face-prewarm:catalog-warm', 1000);
      const beforeCatalogWarm = {voiceCalls, prewarmCalls: prewarmCalls.length};
      resolveCatalogWarm();
      await bounded(earlyVoice, 'catalog voice');
      const catalogVoiceUrl = voiceUrls[voiceUrls.length - 1];

      prewarmMode = 'race-a';
      modelSelect.value = 'race-a:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => prewarmCalls.some(call => call.model === 'race-a:4b'), 'face-prewarm:race-a', 1000);
      const voicesBeforeRace = voiceCalls;
      const raceSubmit = submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'selection-race'},
      );
      prewarmMode = 'race-b';
      modelSelect.value = 'race-b:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await bounded(expectReject(raceSubmit, /changed while preparing/i), 'A to B rejection');
      const voicesAfterSupersede = voiceCalls;
      await waitFor(() => typeof resolveB === 'function', 'face-prewarm:race-b', 1000);
      resolveB();
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.status === 'ready', 'face-prewarm:race-b-ready', 1000);
      prewarmMode = 'success';
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'selection-race-retry'},
      ), 'B retry voice');
      const raceRetryUrl = voiceUrls[voiceUrls.length - 1];

      prewarmMode = 'fail';
      modelSelect.value = 'failure:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.status === 'failed', 'face-prewarm:503-selection', 1000);
      const failBaseline = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'prewarm-503'},
        ),
        /selected model unavailable/i,
      ), '503 rejection');
      const after503 = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
        presence: oracleStage.dataset.presence,
        stageText: oracleStage.textContent,
      };

      prewarmMode = 'mismatch';
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'prewarm-mismatch'},
        ),
        /did not warm exactly/i,
      ), 'mismatch rejection');
      const afterMismatchVoiceCalls = voiceCalls;

      prewarmMode = 'latency-fail';
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'prewarm-latency-fail'},
        ),
        /production latency admission budget/i,
      ), 'latency admission rejection');
      const afterLatencyFailure = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };

      prewarmMode = 'success';
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'prewarm-recovery'},
      ), 'recovery voice');
      const recoveryUrl = voiceUrls[voiceUrls.length - 1];

      prewarmMode = 'success';
      localStorage.setItem('ms4_model_id', 'saved-offline:4b');
      const prewarmsBeforeCatalogFailure = prewarmCalls.length;
      await loadModels();
      await waitFor(
        () => selectedFacePrewarmState && selectedFacePrewarmState.status === 'ready',
        'face-prewarm:saved-catalog-failure-ready',
        1000,
      );
      const retainedAfterCatalogFailure = modelSelect.value;
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'saved-catalog-failure'},
      ), 'saved catalog failure voice');
      const savedCatalogFailureUrl = voiceUrls[voiceUrls.length - 1];
      localStorage.removeItem('ms4_model_id');
      for (const model of ['race-a:4b', 'race-b:4b', 'failure:4b']) {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
      }

      // An explicit A is still warming when the operator selects Auto.  The
      // immediate turn must remain inert until the exact model-free server
      // invalidation receipt arrives.
      prewarmMode = 'race-a';
      resolveA = null;
      modelSelect.value = 'race-a:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => typeof resolveA === 'function', 'face-prewarm:auto-race-a', 1000);
      const pendingAutoBaseline = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      prewarmMode = 'auto-pending';
      resolveAutoInvalidate = null;
      modelSelect.value = '';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => typeof resolveAutoInvalidate === 'function', 'face-prewarm:auto-invalidate-pending', 1000);
      const pendingAutoSubmit = submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'auto-invalidation-pending'},
      );
      await delay(30);
      const pendingAutoBeforeReceipt = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      resolveAutoInvalidate();
      await bounded(pendingAutoSubmit, 'pending Auto invalidation voice');
      const pendingAutoUrl = voiceUrls[voiceUrls.length - 1];

      // Programmatic selection changes do not necessarily emit a change event.
      // Turn admission must still notice that explicit A no longer matches the
      // empty select and create/await the Auto tombstone itself.
      prewarmMode = 'race-a';
      resolveA = null;
      modelSelect.value = 'race-a:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => typeof resolveA === 'function', 'face-prewarm:programmatic-auto-race-a', 1000);
      const programmaticAutoBaseline = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      prewarmMode = 'auto-pending';
      resolveAutoInvalidate = null;
      modelSelect.value = '';
      const programmaticAutoSubmit = submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'programmatic-auto-invalidation-pending'},
      );
      await waitFor(() => typeof resolveAutoInvalidate === 'function', 'face-prewarm:programmatic-auto-invalidate', 1000);
      await delay(30);
      const programmaticAutoBeforeReceipt = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      resolveAutoInvalidate();
      await bounded(programmaticAutoSubmit, 'programmatic Auto invalidation voice');
      const programmaticAutoUrl = voiceUrls[voiceUrls.length - 1];

      // Failed and structurally mismatched invalidation receipts fail closed:
      // retrying the turn cannot reach voice/TTS/history/reflex side effects.
      prewarmMode = 'success';
      modelSelect.value = 'failure:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.status === 'ready', 'face-prewarm:before-auto-failure', 1000);
      prewarmMode = 'auto-fail';
      modelSelect.value = '';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.status === 'failed', 'face-prewarm:auto-failure', 1000);
      const autoFailureBaseline = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'auto-invalidation-failure'},
        ),
        /automatic invalidation unavailable/i,
      ), 'Auto invalidation failure rejection');
      const afterAutoFailure = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };

      prewarmMode = 'auto-mismatch';
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'auto-invalidation-mismatch'},
        ),
        /not admitted exactly/i,
      ), 'Auto invalidation mismatch rejection');
      const afterAutoMismatch = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };

      prewarmMode = 'success';
      await bounded(startSelectedFaceModelPrewarm('automatic recovery before stall'), 'Auto recovery before stall');
      modelSelect.value = 'failure:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.model === 'failure:4b' && selectedFacePrewarmState.status === 'ready', 'face-prewarm:before-auto-stall', 1000);
      prewarmMode = 'auto-stall';
      modelSelect.value = '';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.model === '' && selectedFacePrewarmState.status === 'pending', 'face-prewarm:auto-stall', 1000);
      const autoStallBaseline = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      await bounded(expectReject(
        submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'auto-invalidation-stall'},
        ),
        /not admitted within 5 seconds/i,
      ), 'Auto invalidation deadline rejection', 7000);
      const afterAutoStall = {
        voiceCalls,
        synthCalls,
        reflexCalls,
        historyChildren: messages.children.length,
      };
      prewarmMode = 'success';
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'auto-invalidation-stall-retry'},
      ), 'Auto invalidation deadline recovery');
      const autoStallRecoveryUrl = voiceUrls[voiceUrls.length - 1];

      // A -> Auto -> B: even if the aborted Auto request later returns a
      // superficially valid receipt, it cannot replace or authorize B.
      prewarmMode = 'success';
      await bounded(startSelectedFaceModelPrewarm('automatic recovery'), 'Auto invalidation recovery');
      prewarmMode = 'race-a';
      resolveA = null;
      modelSelect.value = 'race-a:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => typeof resolveA === 'function', 'face-prewarm:stale-auto-race-a', 1000);
      prewarmMode = 'auto-stale';
      resolveStaleAutoInvalidate = null;
      modelSelect.value = '';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => typeof resolveStaleAutoInvalidate === 'function', 'face-prewarm:stale-auto-pending', 1000);
      prewarmMode = 'success';
      modelSelect.value = 'race-b:4b';
      modelSelect.dispatchEvent(new Event('change'));
      await waitFor(() => selectedFacePrewarmState && selectedFacePrewarmState.model === 'race-b:4b' && selectedFacePrewarmState.status === 'ready', 'face-prewarm:B-after-auto', 1000);
      const bGenerationBeforeStaleAuto = selectedFacePrewarmState.generation;
      resolveStaleAutoInvalidate();
      await delay(30);
      const bStateAfterStaleAuto = {
        model: selectedFacePrewarmState && selectedFacePrewarmState.model,
        status: selectedFacePrewarmState && selectedFacePrewarmState.status,
        generation: selectedFacePrewarmState && selectedFacePrewarmState.generation,
      };
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'B-after-stale-auto'},
      ), 'B after stale Auto voice');
      const bAfterStaleAutoUrl = voiceUrls[voiceUrls.length - 1];

      // Fresh-page Auto remains request-free and routes without a model.
      selectedFacePrewarmState = null;
      modelSelect.value = '';
      const prewarmsBeforeInitialAuto = prewarmCalls.length;
      const freshAutoAdmission = await startSelectedFaceModelPrewarm('fresh automatic selection');
      await bounded(submitWavBlobAsVoiceTurn(
        new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
        {durationSec: 0.1, sourceLabel: 'fresh-auto-selection'},
      ), 'fresh Auto voice');
      const freshAutoUrl = voiceUrls[voiceUrls.length - 1];

      equal(JSON.stringify(beforeCatalog), JSON.stringify({voiceCalls: 0, prewarmCalls: 0}), 'catalog gate');
      equal(beforeCatalogWarm.voiceCalls, 0, 'voice must wait for exact warm');
      check(catalogObservedTurn && catalogObservedTurn.kind === 'voice', 'capture observer must receive created voice turn');
      check(catalogObservedTurn.telemetry && catalogObservedTurn.terminalKind, 'capture observer must retain terminal telemetry');
      equal(new URL(catalogVoiceUrl, location.href).searchParams.get('model'), 'restored:4b', 'restored model route');
      equal(voicesAfterSupersede, voicesBeforeRace, 'superseded A turn must not infer');
      equal(new URL(raceRetryUrl, location.href).searchParams.get('model'), 'race-b:4b', 'B retry route');
      equal(after503.voiceCalls, failBaseline.voiceCalls, '503 voice calls');
      equal(after503.synthCalls, failBaseline.synthCalls, '503 synth calls');
      equal(after503.reflexCalls, failBaseline.reflexCalls, '503 reflex calls');
      equal(after503.historyChildren, failBaseline.historyChildren, '503 history');
      equal(after503.presence, 'error', '503 visible state');
      check(/not ready/i.test(after503.stageText), '503 stage must explain not ready');
      equal(afterMismatchVoiceCalls, failBaseline.voiceCalls, 'mismatch voice calls');
      equal(JSON.stringify(afterLatencyFailure), JSON.stringify(failBaseline), 'latency admission failure must have zero effects');
      equal(new URL(recoveryUrl, location.href).searchParams.get('model'), 'failure:4b', 'recovery route');
      equal(retainedAfterCatalogFailure, 'saved-offline:4b', 'saved selection retained on catalog failure');
      check(prewarmCalls.length > prewarmsBeforeCatalogFailure, 'saved selection must exact-prewarm after catalog failure');
      equal(new URL(savedCatalogFailureUrl, location.href).searchParams.get('model'), 'saved-offline:4b', 'saved catalog failure route');
      equal(JSON.stringify(pendingAutoBeforeReceipt), JSON.stringify(pendingAutoBaseline), 'pending Auto invalidation must have zero effects');
      equal(new URL(pendingAutoUrl, location.href).searchParams.has('model'), false, 'admitted Auto voice route');
      equal(JSON.stringify(programmaticAutoBeforeReceipt), JSON.stringify(programmaticAutoBaseline), 'programmatic Auto invalidation must have zero effects');
      equal(new URL(programmaticAutoUrl, location.href).searchParams.has('model'), false, 'programmatic Auto voice route');
      equal(JSON.stringify(afterAutoFailure), JSON.stringify(autoFailureBaseline), 'failed Auto invalidation must have zero effects');
      equal(JSON.stringify(afterAutoMismatch), JSON.stringify(autoFailureBaseline), 'mismatched Auto invalidation must have zero effects');
      equal(JSON.stringify(afterAutoStall), JSON.stringify(autoStallBaseline), 'stalled Auto invalidation must have zero effects');
      equal(new URL(autoStallRecoveryUrl, location.href).searchParams.has('model'), false, 'stalled Auto invalidation retry route');
      equal(JSON.stringify(bStateAfterStaleAuto), JSON.stringify({model: 'race-b:4b', status: 'ready', generation: bGenerationBeforeStaleAuto}), 'stale Auto receipt must not authorize over B');
      equal(new URL(bAfterStaleAutoUrl, location.href).searchParams.get('model'), 'race-b:4b', 'B route after stale Auto receipt');
      equal(freshAutoAdmission.skipped, true, 'fresh Auto remains admission-free');
      equal(prewarmCalls.length, prewarmsBeforeInitialAuto, 'fresh Auto must not request invalidation');
      equal(new URL(freshAutoUrl, location.href).searchParams.has('model'), false, 'fresh Auto voice route');
      return {
        beforeCatalog,
        beforeCatalogWarm,
        catalogVoiceUrl,
        catalogObservedTurnId: catalogObservedTurn && catalogObservedTurn.turnId,
        catalogObservedTerminal: catalogObservedTurn && catalogObservedTurn.terminalKind,
        voicesBeforeRace,
        voicesAfterSupersede,
        raceRetryUrl,
        failBaseline,
        after503,
        afterMismatchVoiceCalls,
        afterLatencyFailure,
        recoveryUrl,
        prewarmsBeforeCatalogFailure,
        retainedAfterCatalogFailure,
        savedCatalogFailureUrl,
        pendingAutoBaseline,
        pendingAutoBeforeReceipt,
        pendingAutoUrl,
        programmaticAutoBaseline,
        programmaticAutoBeforeReceipt,
        programmaticAutoUrl,
        autoFailureBaseline,
        afterAutoFailure,
        afterAutoMismatch,
        autoStallBaseline,
        afterAutoStall,
        autoStallRecoveryUrl,
        bGenerationBeforeStaleAuto,
        bStateAfterStaleAuto,
        bAfterStaleAutoUrl,
        prewarmsBeforeInitialAuto,
        freshAutoUrl,
        prewarmCalls,
      };
    } finally {
      window.fetch = originalFetch;
      window.playRandomReflex = originalPlayRandomReflex;
      if (resolveA) resolveA();
      modelSelect.innerHTML = originalOptions;
      modelSelect.value = originalValue;
      selectedFacePrewarmState = null;
      modelCatalogReadyPromise = originalCatalogPromise || Promise.resolve();
      modelCatalogReady = originalCatalogReady;
      localStorage.removeItem('ms4_model_id');
      resetPlaybackQueue();
      messages.innerHTML = '';
      setOracleStageState('idle', 'Ready.', '');
    }
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });

  // ---- Finding 1: bounded, ownership-aware liveness deadline. ---------------
  streamDeadline = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    window.__ms4VoiceStreamStallTimeoutMs = 250;   // shorten for the test
    const encoder = new TextEncoder();

    // (a) initial fetch() never resolves -> deadline aborts it.
    let fetchAborted = false;
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) {
        const signal = options.signal;
        signal.addEventListener('abort', () => { fetchAborted = true; }, { once: true });
        return new Promise((_res, rej) => { signal.addEventListener('abort', () => rej(new DOMException('Aborted','AbortError')), { once: true }); });
      }
      return originalFetch(url, options);
    };
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'stalled-fetch' });
    await delay(40);
    const afterFetch = window.__ms4OracleVoiceAcceptanceState();
    const fetchCase = {
      fetchAborted,
      terminalKind: afterFetch.lastVoiceTurn && afterFetch.lastVoiceTurn.terminalKind,
      presence: oracleStage.dataset.presence,
      status: voiceStatusLine.textContent,
      controllerCleared: afterFetch.activeVoiceControllerPresent === false,
      turnCleared: afterFetch.activeVoiceTurn === null,
    };

    // (b) response opens but reader.read() never resolves -> deadline aborts it.
    let readAborted = false;
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) {
        const signal = options.signal;
        const reader = {
          read() { return new Promise((_res, rej) => { signal.addEventListener('abort', () => { readAborted = true; rej(new DOMException('Aborted','AbortError')); }, { once: true }); }); },
          cancel() { return Promise.resolve(); },
        };
        return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => reader } });
      }
      return originalFetch(url, options);
    };
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'stalled-read' });
    await delay(40);
    const afterRead = window.__ms4OracleVoiceAcceptanceState();
    const readCase = {
      readAborted,
      terminalKind: afterRead.lastVoiceTurn && afterRead.lastVoiceTurn.terminalKind,
      presence: oracleStage.dataset.presence,
      controllerCleared: afterRead.activeVoiceControllerPresent === false,
    };

    // (b2) Finding 1 (R4): endless heartbeats with NO payload must NOT refresh
    // the payload-progress deadline. Establish two received heartbeats without
    // relying on host timer cadence, then pace later frames normally while the
    // turn reaches exactly one bounded terminal timeout.
    window.__ms4VoiceStreamStallTimeoutMs = 500;
    delete window.__ms4VoiceTurnAbsoluteDeadlineMs;
    let heartbeatsSent = 0;
    const heartbeatFrame =
      'event: heartbeat\\ndata: {"status":"running"}\\n\\n';
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) {
        const signal = options.signal;
        const reader = {
          read() {
            if (signal.aborted) return Promise.resolve({ value: undefined, done: true });
            if (heartbeatsSent < 2) {
              heartbeatsSent += 1;
              return Promise.resolve({
                value: encoder.encode(heartbeatFrame),
                done: false,
              });
            }
            return new Promise((res, rej) => {
              const timer = setTimeout(() => {
                heartbeatsSent += 1;
                res({ value: encoder.encode(heartbeatFrame), done: false });
              }, 40);
              signal.addEventListener('abort', () => { clearTimeout(timer); rej(new DOMException('Aborted','AbortError')); }, { once: true });
            });
          },
          cancel() { return Promise.resolve(); },
        };
        return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => reader } });
      }
      return originalFetch(url, options);
    };
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'endless-heartbeat' });
    const afterHeartbeat = window.__ms4OracleVoiceAcceptanceState();
    const heartbeatCase = {
      heartbeatsSent,
      terminalKind: afterHeartbeat.lastVoiceTurn && afterHeartbeat.lastVoiceTurn.terminalKind,
      controllerCleared: afterHeartbeat.activeVoiceControllerPresent === false,
    };

    // (b3) Finding 1 (R4): an ABSOLUTE turn deadline bounds the turn even when
    // REAL payload keeps arriving. Continuous text_delta frames re-arm the
    // payload deadline but must NOT defeat the absolute bound.
    window.__ms4VoiceStreamStallTimeoutMs = 8000;
    window.__ms4VoiceTurnAbsoluteDeadlineMs = 600;
    let deltasSent = 0;
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) {
        const signal = options.signal;
        let started = false;
        const reader = {
          read() {
            if (signal.aborted) return Promise.resolve({ value: undefined, done: true });
            return new Promise((res, rej) => {
              const frame = started
                ? 'event: text_delta\\ndata: {"text":"tick "}\\n\\n'
                : 'event: transcript\\ndata: {"text":"absolute","asr_ms":1}\\n\\n';
              started = true;
              // Establish payload progress before yielding to wall-clock pacing.
              // The absolute deadline starts before stream setup, so two full
              // browser harnesses running concurrently can otherwise consume
              // most of this fixture's 600 ms budget before frame two arrives.
              // Subsequent reads remain paced and must still be stopped by the
              // absolute deadline even though real payload keeps arriving.
              const delayMs = deltasSent < 2 ? 0 : 60;
              const timer = setTimeout(() => { deltasSent += 1; res({ value: encoder.encode(frame), done: false }); }, delayMs);
              signal.addEventListener('abort', () => { clearTimeout(timer); rej(new DOMException('Aborted','AbortError')); }, { once: true });
            });
          },
          cancel() { return Promise.resolve(); },
        };
        return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => reader } });
      }
      return originalFetch(url, options);
    };
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'absolute-deadline' });
    const afterAbsolute = window.__ms4OracleVoiceAcceptanceState();
    const absoluteCase = {
      deltasSent,
      terminalKind: afterAbsolute.lastVoiceTurn && afterAbsolute.lastVoiceTurn.terminalKind,
    };
    delete window.__ms4VoiceTurnAbsoluteDeadlineMs;

    // (c) a subsequent VALID turn works without any page reload.
    delete window.__ms4VoiceStreamStallTimeoutMs;
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) {
        const frames = encoder.encode(
          'event: transcript\\ndata: {"text":"post-timeout recovery","asr_ms":1}\\n\\n' +
          'event: text_delta\\ndata: {"text":"Recovered reply."}\\n\\n' +
          'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
          'event: done\\ndata: {"session_id":"recovery","reply_text":"Recovered reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"total_ms":5}}\\n\\n'
        );
        let sent = false;
        const reader = { read() { if (!sent) { sent = true; return Promise.resolve({ value: frames, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } };
        return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => reader } });
      }
      return originalFetch(url, options);
    };
    const preTurn = currentVoiceTurnId;
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'recovery' });
    const recovered = await waitFor(() => {
      const s = window.__ms4OracleVoiceAcceptanceState();
      const t = s.lastVoiceTurn;
      return t && t.turnId > preTurn && (t.terminalKind === 'done' || t.terminalKind === 'awaiting_audible') && t.telemetry.firstNonSilentOnsetMs != null ? s : null;
    }, 'stream-deadline:recovery', 5000);
    window.fetch = originalFetch;
    return {
      fetchCase, readCase, heartbeatCase, absoluteCase,
      recoveryTerminalKind: recovered.lastVoiceTurn.terminalKind,
      recoveryOnsetMs: recovered.lastVoiceTurn.telemetry.firstNonSilentOnsetMs,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(streamDeadline.fetchCase.fetchAborted, true, 'a stalled initial fetch must be aborted by the deadline');
  assert.equal(streamDeadline.fetchCase.terminalKind, 'error_timeout');
  assert.equal(streamDeadline.fetchCase.presence, 'error');
  assert.match(streamDeadline.fetchCase.status, /timed out/i);
  assert.equal(streamDeadline.fetchCase.controllerCleared, true);
  assert.equal(streamDeadline.fetchCase.turnCleared, true);
  assert.equal(streamDeadline.readCase.readAborted, true, 'a stalled body read must be aborted by the deadline');
  assert.equal(streamDeadline.readCase.terminalKind, 'error_timeout');
  assert.equal(streamDeadline.readCase.controllerCleared, true);
  // Finding 1 (R4): heartbeats flowed but did NOT keep the turn alive.
  assert.ok(streamDeadline.heartbeatCase.heartbeatsSent >= 2, 'heartbeats must actually have been received; got=' + JSON.stringify(streamDeadline.heartbeatCase));
  assert.equal(streamDeadline.heartbeatCase.terminalKind, 'error_timeout', 'endless heartbeats must not defeat the payload deadline');
  assert.equal(streamDeadline.heartbeatCase.controllerCleared, true);
  // Finding 1 (R4): continuous real payload did NOT defeat the absolute bound.
  assert.ok(streamDeadline.absoluteCase.deltasSent >= 2, 'payload must actually have kept arriving');
  assert.equal(streamDeadline.absoluteCase.terminalKind, 'error_timeout', 'the absolute turn deadline must bound a payload-progressing turn');
  assert.ok(streamDeadline.recoveryOnsetMs != null, 'a valid turn must work after a timeout without reload');

  // ---- Finding 2: delayed legacy barge acknowledgement ownership. -----------
  bargeAck = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    localStorage.setItem('ms4_reflex_barge_ack', 'true');
    const originalPlayRandomReflex = window.playRandomReflex;
    let ackCount = 0;
    window.playRandomReflex = (category) => { if (category === 'ack') ackCount += 1; return Promise.resolve(true); };
    // VAD onset explicitly declines a spoken ack; a second onset advances the turn.
    bargeIn('vad_speech_onset', { halfContext: false, playAck: false });
    bargeIn('vad_speech_onset', { halfContext: false, playAck: false });
    // Internal null-reason reset must never speak.
    resetPlaybackQueue();
    await delay(60);
    const ackAfterPlayAckFalseAndNull = ackCount;
    // A real, non-declined barge schedules one ack; a second barge within 20ms
    // cancels the first pending ack and re-binds ownership -> exactly one fires.
    bargeIn('user_tap_1');
    bargeIn('user_tap_2');
    await delay(60);
    const ackAfterCancelAndOwn = ackCount;
    window.playRandomReflex = originalPlayRandomReflex;
    localStorage.setItem('ms4_reflex_barge_ack', 'false');
    return { ackAfterPlayAckFalseAndNull, ackAfterCancelAndOwn };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(bargeAck.ackAfterPlayAckFalseAndNull, 0, 'playAck:false onsets and null-reason resets must not speak');
  assert.equal(bargeAck.ackAfterCancelAndOwn, 1, 'a canceled pending ack plus one owned ack equals exactly one');

  ownershipRace = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const encoder = new TextEncoder();
    const ctx = getPlaybackCtx();
    await ctx.resume();
    const origDecode = ctx.decodeAudioData.bind(ctx);
    let decodeStarted = false, gatedOnce = false, releaseGate = null;
    const gate = new Promise(res => { releaseGate = res; });
    ctx.decodeAudioData = function (buf) {
      if (!gatedOnce) { gatedOnce = true; decodeStarted = true; return gate.then(() => origDecode(buf)); }
      return origDecode(buf);
    };
    const raceFrames = encoder.encode(
      'event: transcript\\ndata: {"text":"ownership race","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"STALE_A_REPLY"}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
    );
    let raceRead = 0, racePending = null;
    const raceReader = {
      read() { if (raceRead++ === 0) return Promise.resolve({ value: raceFrames, done: false }); return new Promise(res => { racePending = res; }); },
      cancel() { if (racePending) racePending({ value: undefined, done: true }); return Promise.resolve(); },
    };
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => raceReader } });
      return originalFetch(url, options);
    };
    const runA = submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'ownership-race-A' });
    const turnIdA = currentVoiceTurnId;
    await waitFor(() => decodeStarted, 'ownership-race:decode-start', 3000);
    bargeIn('ownership race replacement');
    setOracleStageState('speaking', 'REPLACEMENT_OWNS_UI', 'replacement turn active');
    const presenceAfterBarge = oracleStage.dataset.presence;
    const turnIdAfterBarge = currentVoiceTurnId;
    releaseGate();
    await Promise.race([ runA, delay(3000).then(() => { throw new Error('ownership-race turn A did not settle'); }) ]);
    await delay(40);
    const presenceAfterStale = oracleStage.dataset.presence;
    const voiceStatusAfterStale = voiceStatusLine.textContent;
    const staleSnapshot = lastVoiceTurnAcceptanceState;
    ctx.decodeAudioData = origDecode;
    window.fetch = originalFetch;
    setOracleStageState('idle', 'Ready.', '');
    return {
      decodeStarted, turnIdA, turnIdAfterBarge, turnAdvancedByBarge: turnIdAfterBarge > turnIdA,
      presenceAfterBarge, presenceAfterStale, voiceStatusAfterStale,
      staleTurnDecodedChunks: staleSnapshot ? staleSnapshot.telemetry.decodedChunks : null,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(ownershipRace.decodeStarted, true);
  assert.equal(ownershipRace.turnAdvancedByBarge, true);
  assert.equal(ownershipRace.presenceAfterBarge, 'speaking');
  assert.equal(ownershipRace.presenceAfterStale, 'speaking', 'a stale failed decode must NOT overwrite the replacement turn UI');
  assert.notEqual(ownershipRace.presenceAfterStale, 'blocked');
  assert.doesNotMatch(ownershipRace.voiceStatusAfterStale, /could not be decoded|Audio chunk failed/i);
  assert.equal(ownershipRace.staleTurnDecodedChunks, 0);

  abort = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const encoder = new TextEncoder();
    let streamSignal = null, streamSignalAborted = false, readerCancelled = false, pendingReadResolve = null, readCount = 0;
    const firstFrames = encoder.encode('event: transcript\\ndata: {"text":"browser abort proof","asr_ms":1}\\n\\n' + 'event: text_delta\\ndata: {"text":"PARTIAL_STALE_REPLY"}\\n\\n');
    const staleDone = encoder.encode('event: done\\ndata: {"session_id":"stale-session","reply_text":"STALE_DONE_REPLY","dispatched_job":{"job_id":"stale-job"},"metrics":{}}\\n\\n');
    const reader = {
      read() { if (readCount++ === 0) return Promise.resolve({ value: firstFrames, done: false }); return new Promise(resolve => { pendingReadResolve = resolve; }); },
      cancel() { readerCancelled = true; if (pendingReadResolve) pendingReadResolve({ value: staleDone, done: false }); return Promise.resolve(); },
    };
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/turn/stream')) { streamSignal = options.signal; streamSignal.addEventListener('abort', () => { streamSignalAborted = true; }, { once: true }); return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => reader } }); }
      return originalFetch(url, options);
    };
    const startingTurnId = currentVoiceTurnId;
    const run = submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'browser-test' });
    await waitFor(() => activeVoiceTurnState?.assistantState?.text === 'PARTIAL_STALE_REPLY', 'abort:partial-reply', 3000);
    const interruptedTurn = activeVoiceTurnState;
    const assistantEl = interruptedTurn.assistantState.el;
    const userBubble = interruptedTurn.userBubble;
    bargeIn('browser replacement proof');
    await Promise.race([ run, delay(3000).then(() => { throw new Error('aborted voice turn did not settle'); }) ]);
    await delay(30);
    let synthSignalAborted = false, synthReadPending = false, synthReadReject = null, synthFetchCount = 0, synthReaderCancelled = false;
    window.fetch = (url, options = {}) => {
      if (String(url) === '/voice/synthesize/stream') {
        synthFetchCount += 1;
        options.signal.addEventListener('abort', () => {
          synthSignalAborted = true;
          if (synthReadReject) synthReadReject(new DOMException('Aborted', 'AbortError'));
        }, { once: true });
        const synthReader = {
          read() {
            synthReadPending = true;
            return new Promise((_resolve, reject) => { synthReadReject = reject; });
          },
          cancel() { synthReaderCancelled = true; return Promise.resolve(); },
        };
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          body: { getReader: () => synthReader },
        });
      }
      return originalFetch(url, options);
    };
    const synthRun = speakAnnouncement('stale synthesis proof', currentVoiceTurnId);
    await waitFor(() => activeOracleAudioFetchControllers.size === 1 && synthReadPending, 'abort:synth-controller', 3000);
    bargeIn('browser synthesis replacement proof');
    const synthResult = await synthRun;
    await delay(20);
    window.fetch = originalFetch;
    return {
      streamSignalAborted, readerCancelled, streamSignalIsAborted: Boolean(streamSignal && streamSignal.aborted),
      turnAdvanced: currentVoiceTurnId > startingTurnId, activeVoiceControllerCleared: activeVoiceStreamController === null,
      activeVoiceTurnCleared: activeVoiceTurnState === null, assistantRemoved: !assistantEl.isConnected,
      userMarkedInterrupted: userBubble?.dataset?.voiceTurnState === 'interrupted',
      staleDoneSuppressed: !messages.textContent.includes('STALE_DONE_REPLY') && sessionId !== 'stale-session',
      staleJobSuppressed: !messages.textContent.includes('stale-job'), playbackSources: activeAudioSources.length,
      prebufferedChunks: prebufferQueue.length, playbackRunStarted, audioFetchControllers: activeOracleAudioFetchControllers.size,
      synthSignalAborted, synthFetchCount, synthReaderCancelled, synthResult, oraclePresence: oracleStage.dataset.presence, oracleTranscript: oracleTranscriptLine.textContent, voiceStatus: voiceStatusLine.textContent,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(abort.streamSignalAborted, true);
  assert.equal(abort.readerCancelled, true);
  assert.equal(abort.turnAdvanced, true);
  assert.equal(abort.activeVoiceControllerCleared, true);
  assert.equal(abort.activeVoiceTurnCleared, true);
  assert.equal(abort.assistantRemoved, true);
  assert.equal(abort.userMarkedInterrupted, true);
  assert.equal(abort.staleDoneSuppressed, true);
  assert.equal(abort.staleJobSuppressed, true);
  assert.equal(abort.playbackSources, 0);
  assert.equal(abort.synthFetchCount, 1);
  assert.equal(abort.synthSignalAborted, true);
  assert.equal(abort.synthReaderCancelled, true);
  assert.equal(abort.synthResult, false);
  assert.equal(abort.oraclePresence, 'listening');
  assert.match(abort.voiceStatus, /Interrupted/);

  // ---- Finding 3: decoded silence and good+audio_error are NOT success. -----
  audibleVerdict = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const encoder = new TextEncoder();
    const makeReader = text => { const bytes = encoder.encode(text); let sent = false; return { read() { if (!sent) { sent = true; return Promise.resolve({ value: bytes, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } }; };
    let runSawSpeaking = false;
    const runOne = async (frames, label) => {
      window.fetch = (url, options = {}) => { if (String(url).startsWith('/voice/turn/stream')) return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => makeReader(frames) } }); return originalFetch(url, options); };
      const pre = currentVoiceTurnId;
      runSawSpeaking = false;
      await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: label });
      const s = await waitFor(() => {
        if (oracleStage.dataset.presence === 'speaking') runSawSpeaking = true;
        const st = window.__ms4OracleVoiceAcceptanceState();
        const t = st.lastVoiceTurn;
        const term = t && t.terminalKind;
        return t && t.turnId >= pre && term && term !== 'awaiting_audible' ? st : null;
      }, 'audible:' + label, 5000);
      return s.lastVoiceTurn;
    };
    // Valid all-zero WAV: decodes and schedules, never crosses the onset RMS.
    const silentTurn = await runOne(
      'event: transcript\\ndata: {"text":"silent proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Silent reply."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${silentAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"silent","reply_text":"Silent reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"total_ms":5}}\\n\\n',
      'silent',
    );
    const silentPresence = oracleStage.dataset.presence;
    const silentStatus = voiceStatusLine.textContent;
    // Good audible chunk followed by an audio_error: must be degraded, not done.
    const mixedTurn = await runOne(
      'event: transcript\\ndata: {"text":"mixed proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Partly failed reply."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: audio_error\\ndata: {"index":1,"error":"TTS chunk failed"}\\n\\n' +
      'event: done\\ndata: {"session_id":"mixed","reply_text":"Partly failed reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":1,"total_ms":5}}\\n\\n',
      'mixed',
    );
    // 2026-07-05 PTT terminal-state: mixed is degraded_audio_error at 'done'.
    // WITHOUT starting the next turn, wait for THIS SAME turn's late non-silent
    // onset to record AND all its owned sources to drain, tracking any re-entry
    // into 'speaking' after the degraded terminal (the bug: a late onset flipped
    // a blocked turn back to speaking and stuck there until the recorder timeout).
    const mixedSawSpeakingBeforeTerminal = runSawSpeaking;
    let mixedReentered = oracleStage.dataset.presence === 'speaking';
    const mixedDeadline = Date.now() + 8000;
    while (Date.now() < mixedDeadline) {
      if (oracleStage.dataset.presence === 'speaking') mixedReentered = true;
      const st = window.__ms4OracleVoiceAcceptanceState();
      const t = st.lastVoiceTurn;
      if (t && t.turnId === mixedTurn.turnId && t.telemetry.firstNonSilentOnsetMs != null && st.activeGeneratedSources === 0) break;
      await delay(16);
    }
    if (oracleStage.dataset.presence === 'speaking') mixedReentered = true;
    await delay(120);
    if (oracleStage.dataset.presence === 'speaking') mixedReentered = true;
    const mixedAfter = window.__ms4OracleVoiceAcceptanceState();
    const mixedFinal = mixedAfter.lastVoiceTurn;
    const mixed = {
      terminalKind: mixedFinal.terminalKind,
      audioErrors: mixedFinal.telemetry.audioErrors,
      onsetMs: mixedFinal.telemetry.firstNonSilentOnsetMs,
      completionMs: mixedFinal.telemetry.audibleCompletionMs ?? null,
      clientCompletionMs: mixedFinal.clientMetrics?.client_full_utterance_completion_ms ?? null,
      presence: oracleStage.dataset.presence,
      turnPhase: oracleStage.dataset.turnPhase,
      status: voiceStatusLine.textContent,
      ownedSources: mixedAfter.activeGeneratedSources,
      onsetMonitorActive: mixedFinal.onsetMonitorActive,
      sawSpeakingBeforeTerminal: mixedSawSpeakingBeforeTerminal,
      reenteredSpeaking: mixedReentered,
    };

    // A duration-gated chunk may carry the full source text in metadata while
    // its WAV was physically shortened. It must be degraded, never accepted as
    // complete audible delivery.
    const gatedTurn = await runOne(
      'event: transcript\\ndata: {"text":"duration gate proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Potentially clipped reply."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"runtime_gated":true,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"gated","reply_text":"Potentially clipped reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"runtime_gated":0,"total_ms":5}}\\n\\n',
      'duration-gated',
    );
    const gatedPresence = oracleStage.dataset.presence;
    const gatedStatus = voiceStatusLine.textContent;

    // Healthy control: a SILENT chunk then an AUDIBLE chunk in ONE turn keeps
    // onset monitoring across the chunk boundary and, after FULL drain, reaches
    // speaking then finalizes 'done' (never degraded_silent).
    const multiTurn = await runOne(
      'event: transcript\\ndata: {"text":"multi proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Silent then audible."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${silentAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":1,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"multi","reply_text":"Silent then audible.","metrics":{"audio_chunks":2,"audio_client_written":2,"audio_errors":0,"total_ms":5}}\\n\\n',
      'multi',
    );
    const multiBecameSpeaking = runSawSpeaking;
    const multiAfter = window.__ms4OracleVoiceAcceptanceState();
    window.fetch = originalFetch;
    setOracleStageState('idle', 'Ready.', '');
    return {
      silent: { terminalKind: silentTurn.terminalKind, onsetMs: silentTurn.telemetry.firstNonSilentOnsetMs, scheduled: silentTurn.telemetry.scheduledChunks, presence: silentPresence, status: silentStatus },
      mixed,
      durationGated: {terminalKind: gatedTurn.terminalKind, presence: gatedPresence, status: gatedStatus},
      multiSource: {
        terminalKind: multiTurn.terminalKind,
        onsetMs: multiTurn.telemetry.firstNonSilentOnsetMs,
        completionMs: multiTurn.telemetry.audibleCompletionMs,
        clientCompletionMs: multiTurn.clientMetrics?.client_full_utterance_completion_ms ?? null,
        scheduled: multiTurn.telemetry.scheduledChunks,
        becameSpeaking: multiBecameSpeaking,
        ownedSources: multiAfter.activeGeneratedSources,
      },
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(audibleVerdict.silent.terminalKind, 'degraded_silent', 'decoded silence must not be ordinary done');
  assert.equal(audibleVerdict.silent.onsetMs, null, 'silent audio must never register a non-silent onset');
  assert.ok(audibleVerdict.silent.scheduled >= 1, 'the silent WAV must actually decode and schedule');
  assert.equal(audibleVerdict.silent.presence, 'blocked');
  assert.doesNotMatch(audibleVerdict.silent.status, /Voice turn complete/i);
  // mixed: good chunk + audio_error is degraded_audio_error, and a LATE onset
  // (recorded after the terminal) must NOT re-enter speaking — the stage stays
  // blocked, owned sources fully drain, and the onset monitor RAF is cleared.
  assert.equal(audibleVerdict.mixed.terminalKind, 'degraded_audio_error', 'good chunk + audio_error must be degraded, not done');
  assert.ok(audibleVerdict.mixed.audioErrors >= 1);
  assert.ok(audibleVerdict.mixed.onsetMs != null, 'the late non-silent onset must be recorded in retained diagnostics');
  assert.equal(audibleVerdict.mixed.reenteredSpeaking, false, 'a late onset must never re-enter speaking after the degraded terminal');
  assert.equal(audibleVerdict.mixed.presence, 'blocked', 'the stage must stay blocked after the late onset drains');
  assert.equal(audibleVerdict.mixed.turnPhase, 'blocked');
  assert.match(audibleVerdict.mixed.status, /degraded/i);
  assert.equal(audibleVerdict.mixed.ownedSources, 0, 'all owned sources must drain');
  assert.equal(audibleVerdict.mixed.onsetMonitorActive, false, 'the onset monitor RAF/analyser must be cleared after drain');
  assert.equal(audibleVerdict.mixed.completionMs, null, 'a degraded partial utterance must not claim full playback completion');
  assert.equal(audibleVerdict.mixed.clientCompletionMs, null, 'degraded audio must not expose client_full_utterance_completion_ms');
  assert.equal(audibleVerdict.durationGated.terminalKind, 'degraded_audio_error', 'duration-gated speech must not be ordinary done');
  assert.equal(audibleVerdict.durationGated.presence, 'blocked');
  assert.match(audibleVerdict.durationGated.status, /incomplete|degraded/i);
  // Findings 4 + 6 (R4) + healthy PTT control: the silent-then-audible turn keeps
  // onset monitoring across the chunk boundary, REACHES speaking on the audible
  // chunk, and finalizes 'done' only after both owned sources fully drain.
  assert.ok(audibleVerdict.multiSource.scheduled >= 2, 'both chunks must decode and schedule');
  assert.ok(audibleVerdict.multiSource.onsetMs != null, 'onset must be observed on the later audible chunk');
  assert.equal(audibleVerdict.multiSource.becameSpeaking, true, 'a clean awaiting_audible turn must reach speaking on the real onset');
  assert.equal(audibleVerdict.multiSource.terminalKind, 'done', 'a silent-then-audible multi-source turn drains to done, not degraded_silent');
  assert.equal(audibleVerdict.multiSource.ownedSources, 0, 'the healthy turn must fully drain');
  assert.ok(Number.isFinite(audibleVerdict.multiSource.completionMs), 'a clean fully drained turn must expose finite playback completion');
  assert.ok(audibleVerdict.multiSource.completionMs >= audibleVerdict.multiSource.onsetMs, 'playback completion must be at/after measured response onset');
  assert.equal(audibleVerdict.multiSource.clientCompletionMs, audibleVerdict.multiSource.completionMs, 'client_full_utterance_completion_ms must mirror the exact drain telemetry');
  assert.notEqual(audibleVerdict.multiSource.completionMs, audibleVerdict.multiSource.onsetMs, 'response onset and playback completion must remain distinct measurements');

  retained = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const encoder = new TextEncoder();
    const originalFetch = window.fetch;
    const UNDECODABLE = ${undecodableFlag};
    const sentinel = { id: 'ora13-retained-page' };
    window.__ora13_run = sentinel;
    const startingPageState = window.__ms4OracleVoiceAcceptanceState();
    let voiceCall = 0;
    const makeReader = text => { const bytes = encoder.encode(text); let sent = false; return { read() { if (!sent) { sent = true; return Promise.resolve({ value: bytes, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } }; };
    const goodFrames = session => (
      'event: transcript\\ndata: {"text":"retained page proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Audible retained page reply."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"' + session + '","reply_text":"Audible retained page reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"total_ms":5}}\\n\\n'
    );
    const blockedFrames = UNDECODABLE ? (
      'event: transcript\\ndata: {"text":"undecodable audio proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"This output is unavailable."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"Tk9UX0FVRElP","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"blocked-session","reply_text":"This output is unavailable.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"total_ms":5}}\\n\\n'
    ) : (
      'event: transcript\\ndata: {"text":"zero audio proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"This output is unavailable."}\\n\\n' +
      'event: audio_error\\ndata: {"index":0,"error":"TTS unavailable"}\\n\\n' +
      'event: done\\ndata: {"session_id":"blocked-session","reply_text":"This output is unavailable.","metrics":{"audio_chunks":0,"audio_client_written":0,"audio_errors":1,"total_ms":5}}\\n\\n'
    );
    window.fetch = (url, options = {}) => { if (String(url).startsWith('/voice/turn/stream')) { voiceCall += 1; const frames = voiceCall === 2 ? blockedFrames : goodFrames('good-session-' + voiceCall); return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => makeReader(frames) } }); } return originalFetch(url, options); };
    const voiceBlob = new Blob([new Uint8Array(44)], { type: 'audio/wav' });
    await submitWavBlobAsVoiceTurn(voiceBlob, { durationSec: 0.1, sourceLabel: 'retained-one' });
    const firstState = await waitFor(() => {
      const s = window.__ms4OracleVoiceAcceptanceState();
      const t = s.lastVoiceTurn;
      return t?.telemetry?.firstNonSilentOnsetMs != null && s.activeGeneratedSources > 0 ? s : null;
    }, 'retained:first-onset-before-drain', 5000);
    await submitWavBlobAsVoiceTurn(voiceBlob, { durationSec: 0.1, sourceLabel: 'retained-zero' });
    const blockedState = window.__ms4OracleVoiceAcceptanceState();
    const blockedPresence = oracleStage.dataset.presence;
    const blockedStatus = voiceStatusLine.textContent;
    await submitWavBlobAsVoiceTurn(voiceBlob, { durationSec: 0.1, sourceLabel: 'retained-recovery' });
    const recoveredState = await waitFor(() => { const s = window.__ms4OracleVoiceAcceptanceState(); const t = s.lastVoiceTurn; return t?.turnId > blockedState.lastVoiceTurn.turnId && t?.telemetry?.firstNonSilentOnsetMs != null ? s : null; }, 'retained:recovery-onset', 5000);
    await waitFor(() => { const s = window.__ms4OracleVoiceAcceptanceState(); return s.activeGeneratedSources === 0 && oracleStage.dataset.presence === 'idle'; }, 'retained:drain-idle', 5000);
    const drainedState = window.__ms4OracleVoiceAcceptanceState();
    const drainedStatus = voiceStatusLine.textContent;
    window.fetch = originalFetch;
    return {
      sentinelStable: window.__ora13_run === sentinel,
      pageInstanceStable: drainedState.pageInstanceId === startingPageState.pageInstanceId,
      voiceOnsetRmsThreshold: VOICE_ONSET_RMS_THRESHOLD, voiceCall, firstState, blockedState, blockedPresence, blockedStatus, recoveredState, drainedState, drainedStatus,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(retained.sentinelStable, true);
  assert.equal(retained.pageInstanceStable, true);
  assert.equal(retained.voiceCall, 3);
  const ft = retained.firstState.lastVoiceTurn.telemetry;
  const fm = retained.firstState.lastVoiceTurn.clientMetrics;
  assert.ok(ft.firstSseReceivedMs != null);
  assert.ok(ft.firstDecodedMs >= ft.firstSseReceivedMs);
  assert.ok(ft.firstScheduledMs >= ft.firstDecodedMs);
  assert.ok(ft.firstNonSilentOnsetMs > ft.firstSseReceivedMs);
  assert.ok(ft.onsetRms >= retained.voiceOnsetRmsThreshold, 'captured onset RMS must be at/above VOICE_ONSET_RMS_THRESHOLD');
  assert.ok(ft.firstNonSilentOnsetMs >= ft.firstScheduledMs, 'audible onset must be at/after the scheduled AudioContext start');
  assert.equal(fm.client_first_audible_ms, ft.firstNonSilentOnsetMs);
  assert.notEqual(fm.client_first_audible_ms, ft.firstSseReceivedMs);
  assert.equal(fm.client_full_utterance_completion_ms, null, 'the onset snapshot must not claim playback completion before drain');
  if (variant === 'undecodable') {
    assert.equal(retained.blockedState.lastVoiceTurn.telemetry.sseAudioChunks, 1);
    assert.equal(retained.blockedState.lastVoiceTurn.telemetry.decodedChunks, 0);
    assert.equal(retained.blockedState.lastVoiceTurn.telemetry.scheduledChunks, 0);
    assert.equal(retained.blockedState.lastVoiceTurn.telemetry.decodeErrors, 1);
  } else {
    assert.equal(retained.blockedState.lastVoiceTurn.telemetry.sseAudioChunks, 0);
  }
  assert.equal(retained.blockedState.lastVoiceTurn.terminalKind, 'blocked_zero_audio');
  assert.equal(retained.blockedPresence, 'blocked');
  assert.ok(retained.recoveredState.lastVoiceTurn.turnId > retained.blockedState.lastVoiceTurn.turnId);
  assert.equal(retained.recoveredState.lastVoiceTurn.terminalKind, 'awaiting_audible', 'measured onset must remain distinct from full playback completion');
  const drainedTurn = retained.drainedState.lastVoiceTurn;
  assert.equal(drainedTurn.terminalKind, 'done', 'the clean turn becomes done only after every owned source drains');
  assert.ok(Number.isFinite(drainedTurn.telemetry.audibleCompletionMs), 'the retained clean turn must expose exact drain completion telemetry');
  assert.ok(drainedTurn.telemetry.audibleCompletionMs >= drainedTurn.telemetry.firstNonSilentOnsetMs, 'retained playback completion must be at/after onset');
  assert.equal(drainedTurn.clientMetrics.client_full_utterance_completion_ms, drainedTurn.telemetry.audibleCompletionMs);
  assert.notEqual(drainedTurn.clientMetrics.client_full_utterance_completion_ms, drainedTurn.clientMetrics.client_first_audible_ms, 'onset compatibility timing must not be overwritten with completion');
  assert.match(retained.drainedStatus, /Voice turn complete/i);

  // Repair76 causal browser gate: input health must not mask a latched,
  // fail-closed output error. A newer accepted audible turn is the only event
  // in this chronology allowed to restore undifferentiated readiness.
  voiceReadinessState = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const encoder = new TextEncoder();
    const makeReader = text => {
      const bytes = encoder.encode(text);
      let sent = false;
      return {
        read() {
          if (!sent) {
            sent = true;
            return Promise.resolve({value: bytes, done: false});
          }
          return Promise.resolve({value: undefined, done: true});
        },
        cancel() { return Promise.resolve(); },
      };
    };
    const failClosedFrames =
      'event: transcript\\ndata: {"text":"fail closed readiness proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"This output is blocked."}\\n\\n' +
      'event: error\\ndata: {"error":"Voice renderer rejected output.","diagnostics":"voice_output_fail_closed","fail_closed":true}\\n\\n';
    const successfulFrames =
      'event: transcript\\ndata: {"text":"newer success readiness proof","asr_ms":1}\\n\\n' +
      'event: text_delta\\ndata: {"text":"Audible recovery reply."}\\n\\n' +
      'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n' +
      'event: done\\ndata: {"session_id":"readiness-recovered","reply_text":"Audible recovery reply.","metrics":{"audio_chunks":1,"audio_client_written":1,"audio_errors":0,"total_ms":5}}\\n\\n';
    let voiceCall = 0;
    window.fetch = (url, options = {}) => {
      const target = String(url);
      if (target.startsWith('/voice/turn/stream')) {
        voiceCall += 1;
        const frames = voiceCall === 1 ? failClosedFrames : successfulFrames;
        return Promise.resolve({
          ok: true,
          statusText: 'OK',
          body: {getReader: () => makeReader(frames)},
        });
      }
      if (target === '/voice/status') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            voice_input_ready: true,
            asr: {service: 'ASR', healthy: true, provisioning_state: 'ready', detail: 'ASR input healthy'},
          }),
        });
      }
      return originalFetch(url, options);
    };
    try {
      const voiceBlob = new Blob([new Uint8Array(44)], {type: 'audio/wav'});
      await submitWavBlobAsVoiceTurn(
        voiceBlob,
        {durationSec: 0.1, sourceLabel: 'readiness-fail-closed'},
      );
      const failedState = window.__ms4OracleVoiceAcceptanceState();
      const failedTurnId = failedState.lastVoiceTurn && failedState.lastVoiceTurn.turnId;
      await refreshVoice();
      const afterHealthyInput = window.__ms4OracleVoiceAcceptanceState();
      const plainReadySurfaceIds = [...document.querySelectorAll('body *')]
        .filter(element => {
          if (element.children.length || !/^\\s*voice ready\\s*$/i.test(element.textContent || '')) return false;
          const style = getComputedStyle(element);
          return !element.hidden && style.display !== 'none' && style.visibility !== 'hidden';
        })
        .map(element => element.id || element.tagName.toLowerCase());
      const latched = {
        terminalKind: afterHealthyInput.lastVoiceTurn && afterHealthyInput.lastVoiceTurn.terminalKind,
        turnId: afterHealthyInput.lastVoiceTurn && afterHealthyInput.lastVoiceTurn.turnId,
        failedTurnId,
        presence: oracleStage.dataset.presence,
        subtitle: oracleSubtitle.textContent,
        topText: voiceEl.textContent,
        topTitle: voiceEl.title,
        plainReadySurfaceIds,
        liveText: voiceStatusLine.textContent,
        liveRole: voiceStatusLine.getAttribute('role'),
        liveMode: voiceStatusLine.getAttribute('aria-live'),
        liveAtomic: voiceStatusLine.getAttribute('aria-atomic'),
      };

      await submitWavBlobAsVoiceTurn(
        voiceBlob,
        {durationSec: 0.1, sourceLabel: 'readiness-newer-success'},
      );
      const recoveredState = await waitFor(() => {
        const state = window.__ms4OracleVoiceAcceptanceState();
        const turn = state.lastVoiceTurn;
        return turn && turn.turnId > failedTurnId && turn.terminalKind === 'done' ? state : null;
      }, 'readiness:newer-success', 5000);
      await waitFor(
        () => window.__ms4OracleVoiceAcceptanceState().activeGeneratedSources === 0,
        'readiness:newer-success-drain',
        5000,
      );
      await refreshVoice();
      const recovered = {
        terminalKind: recoveredState.lastVoiceTurn.terminalKind,
        turnId: recoveredState.lastVoiceTurn.turnId,
        topText: voiceEl.textContent,
        topTitle: voiceEl.title,
        liveText: voiceStatusLine.textContent,
      };
      return {voiceCall, latched, recovered};
    } finally {
      window.fetch = originalFetch;
    }
  })()`, {timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS});
  const readinessVerdict = {
    failClosedTerminalRetained:
      voiceReadinessState.latched.terminalKind === 'error_fail_closed'
      && voiceReadinessState.latched.turnId === voiceReadinessState.latched.failedTurnId,
    noPlainReadyWhileOutputBlocked:
      voiceReadinessState.latched.plainReadySurfaceIds.length === 0,
    splitInputOutputTruth:
      /input ready/i.test(voiceReadinessState.latched.topText)
      && /output (?:blocked|failed)/i.test(voiceReadinessState.latched.topText),
    outputFailureRemainsVisible:
      voiceReadinessState.latched.presence === 'blocked'
      && /output blocked/i.test(voiceReadinessState.latched.subtitle)
      && /voice turn failed/i.test(voiceReadinessState.latched.liveText),
    ariaLivePreserved:
      voiceReadinessState.latched.liveRole === 'status'
      && voiceReadinessState.latched.liveMode === 'polite'
      && voiceReadinessState.latched.liveAtomic === 'true',
    newerSuccessfulTurn:
      voiceReadinessState.recovered.turnId > voiceReadinessState.latched.failedTurnId
      && voiceReadinessState.recovered.terminalKind === 'done',
    truthfulReadyRestored:
      /^voice ready$/i.test(voiceReadinessState.recovered.topText),
  };
  assert.deepEqual(readinessVerdict, {
    failClosedTerminalRetained: true,
    noPlainReadyWhileOutputBlocked: true,
    splitInputOutputTruth: true,
    outputFailureRemainsVisible: true,
    ariaLivePreserved: true,
    newerSuccessfulTurn: true,
    truthfulReadyRestored: true,
  }, `Repair76 voice readiness contradiction: ${JSON.stringify(voiceReadinessState)}`);

  // ---- Finding 5: monotonic latest-turn acceptance-snapshot ownership. ------
  snapshotOwnership = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const encoder = new TextEncoder();
    const ctx = getPlaybackCtx();
    await ctx.resume();
    const origDecode = ctx.decodeAudioData.bind(ctx);
    let decodeStartedA = false, gatedOnce = false, releaseGate = null;
    const gate = new Promise(res => { releaseGate = res; });
    ctx.decodeAudioData = function (buf) { if (!gatedOnce) { gatedOnce = true; decodeStartedA = true; return gate.then(() => origDecode(buf)); } return origDecode(buf); };
    const aFrames = encoder.encode('event: transcript\\ndata: {"text":"snapshot A","asr_ms":1}\\n\\n' + 'event: text_delta\\ndata: {"text":"A_REPLY"}\\n\\n' + 'event: audio_chunk\\ndata: {"index":0,"audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n');
    let aRead = 0, aPending = null;
    const aReader = { read() { if (aRead++ === 0) return Promise.resolve({ value: aFrames, done: false }); return new Promise(res => { aPending = res; }); }, cancel() { if (aPending) aPending({ value: undefined, done: true }); return Promise.resolve(); } };
    const bFrames = encoder.encode('event: transcript\\ndata: {"text":"snapshot B","asr_ms":1}\\n\\n' + 'event: text_delta\\ndata: {"text":"B_REPLY"}\\n\\n' + 'event: done\\ndata: {"session_id":"snap-B","reply_text":"B_REPLY","metrics":{"audio_chunks":0,"audio_client_written":0,"audio_errors":0,"total_ms":5}}\\n\\n');
    const bReader = () => { let sent = false; return { read() { if (!sent) { sent = true; return Promise.resolve({ value: bFrames, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } }; };
    let useB = false;
    window.fetch = (url, options = {}) => { if (String(url).startsWith('/voice/turn/stream')) { if (useB) return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: bReader } }); return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => aReader } }); } return originalFetch(url, options); };
    const runA = submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'snapshot-A' });
    const turnIdA = currentVoiceTurnId;
    await waitFor(() => decodeStartedA, 'snapshot:decodeA', 3000);
    // Turn B starts (barges A) and completes with a higher turn id.
    useB = true;
    await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'snapshot-B' });
    const turnIdB = currentVoiceTurnId;
    const afterB = window.__ms4OracleVoiceAcceptanceState();
    // Now release A's held decode; its stale cleanup must NOT overwrite B.
    releaseGate();
    await Promise.race([ runA, delay(3000).then(() => { throw new Error('snapshot A did not settle'); }) ]);
    await delay(40);
    const afterAReleased = window.__ms4OracleVoiceAcceptanceState();
    ctx.decodeAudioData = origDecode;
    window.fetch = originalFetch;
    setOracleStageState('idle', 'Ready.', '');
    return {
      turnIdA, turnIdB,
      lastTurnIdAfterB: afterB.lastVoiceTurn && afterB.lastVoiceTurn.turnId,
      lastTurnIdAfterARelease: afterAReleased.lastVoiceTurn && afterAReleased.lastVoiceTurn.turnId,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.ok(snapshotOwnership.turnIdB > snapshotOwnership.turnIdA);
  assert.equal(snapshotOwnership.lastTurnIdAfterB, snapshotOwnership.turnIdB, 'B is the newest snapshot after it completes');
  assert.equal(snapshotOwnership.lastTurnIdAfterARelease, snapshotOwnership.turnIdB, 'a stale turn A release must not overwrite the newer B snapshot');

  // ---- Bounded multi-minute input integration (served production code). ----
  // First drive the real VAD onset/offset path through an ordinary 1.5 second
  // sentence pause. The pre-integration page submits immediately at offset; the
  // accepted adapter keeps one session alive until the core's 3 second final
  // silence. Then drive three virtual minutes through the production adapter,
  // real WAV encoder, /voice/transcribe fetch seam, and JSON turn-stream seam.
  longInputSession = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const originalReadOraclePlaybackSignal = readOraclePlaybackSignal;
    const originalVadAck = localStorage.getItem('ms4_voice_vad_ack');
    const encoder = new TextEncoder();
    let turnRequests = 0;
    let transcribeRequests = 0;
    let capturedTurn = null;
    let transcriptionText = 'ordinary pause sentinel';
    let finalSilenceRearmedAt = null;
    let bridgeAtTranscribeStart = null;
    let transcribeAfterFinalSilenceMs = null;

    const headerValue = (headers, name) => {
      if (!headers) return '';
      if (typeof headers.get === 'function') return headers.get(name) || '';
      return headers[name] || headers[name.toLowerCase()] || '';
    };
    const streamResponse = transcript => {
      const frames = encoder.encode(
        'event: transcript\\ndata: ' + JSON.stringify({
          text: transcript, raw_text: transcript, asr_ms: 0, model: 'bounded-session',
        }) + '\\n\\n' +
        'event: done\\ndata: ' + JSON.stringify({
          session_id: 'bounded-browser-session',
          transcript,
          reply_text: '',
          transcription_model: 'bounded-session',
          metrics: {audio_chunks: 0, audio_client_written: 0, audio_errors: 0, total_ms: 1},
        }) + '\\n\\n'
      );
      let sent = false;
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => '',
        body: {
          getReader: () => ({
            read() {
              if (!sent) { sent = true; return Promise.resolve({value: frames, done: false}); }
              return Promise.resolve({value: undefined, done: true});
            },
            cancel() { return Promise.resolve(); },
          }),
        },
      };
    };
    const captureTurnRequest = (options = {}) => {
      const contentType = headerValue(options.headers, 'Content-Type');
      let envelope = null;
      if (typeof options.body === 'string') {
        try { envelope = JSON.parse(options.body); } catch (_error) { envelope = null; }
      }
      capturedTurn = {
        contentType,
        envelope,
        bodyKeys: envelope ? Object.keys(envelope).sort() : [],
      };
      turnRequests += 1;
      return streamResponse(envelope && envelope.transcript ? envelope.transcript : 'legacy multipart');
    };

    window.fetch = (url, options = {}) => {
      const target = String(url);
      if (target.startsWith('/voice/transcribe')) {
        bridgeAtTranscribeStart = oracleThinkingBridgeState();
        transcribeAfterFinalSilenceMs = finalSilenceRearmedAt == null
          ? null
          : Math.round(performance.now() - finalSilenceRearmedAt);
        transcribeRequests += 1;
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: async () => '',
          json: async () => ({
            schema: 'Ms4VoiceTranscription.v1',
            text: transcriptionText,
            model: 'browser-fixture-asr',
          }),
        });
      }
      if (target.startsWith('/voice/turn/stream')) {
        return Promise.resolve(captureTurnRequest(options));
      }
      return originalFetch(url, options);
    };
    readOraclePlaybackSignal = () => ({active: false, level: 0, voice: 0, air: 0});
    localStorage.setItem('ms4_voice_vad_ack', 'false');
    if (typeof cancelActiveOracleVoiceInputSession === 'function') {
      await cancelActiveOracleVoiceInputSession('browser-long-input-preflight');
    }

    vadState.enabled = true;
    vadState.phase = 'speech';
    vadState.sourceSampleRate = 48000;
    vadState.smoothedRms = 0.08;
    vadState.speechStartedAt = Date.now() - 4000;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.08;
    vadState.uttFrames = 20;
    vadState.uttPlaybackFrames = 0;
    vadState.uttPlaybackLevelSum = 0;
    vadPreRoll = [new Float32Array(480).fill(0.04)];
    const pauseTurnBefore = currentVoiceTurnId;
    await onVadSpeechOnset();
    for (let i = 0; i < 8; i += 1) vadPushAudio(new Float32Array(4800).fill(0.05));
    const pauseMs = 1500;
    vadState.lastTransitionAt = Date.now() - pauseMs;
    await onVadSpeechOffset();
    await delay(80);
    const requestsAfterOrdinaryPause = turnRequests;
    const bridgeAfterSpeechEnd = oracleThinkingBridgeState();
    const pauseSession = activeOracleVoiceInputSession;
    const continuationResumed = Boolean(pauseSession && pauseSession.speechStart());
    await delay(20);
    const bridgeAfterResume = oracleThinkingBridgeState();
    finalSilenceRearmedAt = performance.now();
    const finalSilenceRearmed = Boolean(pauseSession && pauseSession.speechEnd());
    const bridgeAfterFinalSpeechEnd = oracleThinkingBridgeState();
    const loaderPresent = !!(
      window.MS4VoiceInputSession
      && typeof window.MS4VoiceInputSession.createVoiceInputSession === 'function'
    );
    const adapterPresent = typeof createOracleVoiceInputSession === 'function';
    if (loaderPresent && adapterPresent) {
      await waitFor(() => turnRequests === 1, 'bounded-input:final-silence-submit', 4500);
      await delay(60);
    }
    const ordinaryPause = {
      pauseMs,
      requestsAfterOrdinaryPause,
      requestsAfterFinalSilence: turnRequests,
      turnIdBefore: pauseTurnBefore,
      turnIdDelta: currentVoiceTurnId - pauseTurnBefore,
      contentType: capturedTurn ? capturedTurn.contentType : '',
      bodyKeys: capturedTurn ? capturedTurn.bodyKeys : [],
      transcribeRequests,
      bridgeAfterSpeechEnd,
      continuationResumed,
      bridgeAfterResume,
      finalSilenceRearmed,
      bridgeAfterFinalSpeechEnd,
      bridgeAtTranscribeStart,
      transcribeAfterFinalSilenceMs,
    };

    if (typeof cancelActiveOracleVoiceInputSession === 'function') {
      await cancelActiveOracleVoiceInputSession('browser-long-input-pause-cleanup');
    }
    vadState.enabled = false;
    vadState.phase = 'silence';
    vadPreRoll = [];

    // Causal long/run-on regression: one accepted phrase arms the bounded
    // session, a continuation arrives after 1.5 s, and that continuation ends
    // as a short/weak fragment. Rejecting the later fragment must not cancel
    // the prior accepted speech; the original 3 s final-silence owner submits.
    turnRequests = 0;
    transcribeRequests = 0;
    capturedTurn = null;
    transcriptionText = 'accepted phrase sentinel';
    vadState.enabled = true;
    vadState.phase = 'speech';
    vadState.sourceSampleRate = 48000;
    vadState.smoothedRms = 0.08;
    vadState.speechStartedAt = Date.now() - 4000;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.08;
    vadState.uttFrames = 20;
    vadState.uttPlaybackFrames = 0;
    vadState.uttPlaybackLevelSum = 0;
    vadPreRoll = [new Float32Array(480).fill(0.04)];
    const rejectedTailTurnBefore = currentVoiceTurnId;
    await onVadSpeechOnset();
    const rejectedTailSession = activeOracleVoiceInputSession;
    for (let i = 0; i < 8; i += 1) vadPushAudio(new Float32Array(4800).fill(0.05));
    vadState.speechStartedAt = Date.now() - 4000;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.08;
    vadState.uttFrames = 20;
    await onVadSpeechOffset();
    const acceptedPhraseArmed = !!(
      rejectedTailSession
      && rejectedTailSession.inspect().state === 'active'
      && rejectedTailSession.isFinalSilenceArmed()
    );

    const continuationPauseMs = 1500;
    await delay(continuationPauseMs);
    vadState.phase = 'speech';
    vadState.smoothedRms = 0.01;
    vadState.speechStartedAt = Date.now() - 100;
    vadState.lastSpeechSampleAt = Date.now();
    vadPreRoll = [new Float32Array(480).fill(0.01)];
    await onVadSpeechOnset();
    vadPushAudio(new Float32Array(480).fill(0.01));
    vadState.speechStartedAt = Date.now() - 100;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.01;
    vadState.uttFrames = 2;
    vadState.uttPlaybackFrames = 0;
    vadState.uttPlaybackLevelSum = 0;
    const trailingDecision = window.MS4DSP.shouldSubmitUtterance({
      speechMs: 100,
      peakRms: 0.01,
      onsetThreshold: vadAdaptiveThresholds().onset,
      playbackActiveFraction: 0,
      playbackMeanLevel: 0,
    });
    rejectedTailSession.speechEnd();
    await onVadSpeechOffset();
    const requestsAfterRejectedTail = turnRequests;
    const pcmSamplesAfterRejectedTail = rejectedTailSession.inspect().pcmSamples;
    const finalSilenceMs = 3000;
    await delay(finalSilenceMs + 300);
    await rejectedTailSession.settled();
    await delay(60);
    const rejectedTailInspect = rejectedTailSession.inspect();
    const rejectedTailContinuation = {
      continuationPauseMs,
      finalSilenceMs,
      acceptedPhraseArmed,
      trailingDecision,
      requestsAfterRejectedTail,
      requestsAfterFinalSilence: turnRequests,
      pcmSamplesAfterRejectedTail,
      turnIdDelta: currentVoiceTurnId - rejectedTailTurnBefore,
      transcript: capturedTurn && capturedTurn.envelope
        ? capturedTurn.envelope.transcript
        : '',
      inspect: rejectedTailInspect,
    };
    if (typeof cancelActiveOracleVoiceInputSession === 'function') {
      await cancelActiveOracleVoiceInputSession('browser-long-input-rejected-tail-cleanup');
    }
    vadState.enabled = false;
    vadState.phase = 'silence';
    vadPreRoll = [];

    // Negative control: the same weak fragment as the first phrase has no
    // accepted session content to preserve and must still cancel fail-closed.
    turnRequests = 0;
    transcribeRequests = 0;
    capturedTurn = null;
    transcriptionText = 'unexpected first-fragment transcript';
    vadState.enabled = true;
    vadState.phase = 'speech';
    vadState.sourceSampleRate = 48000;
    vadState.smoothedRms = 0.01;
    vadState.speechStartedAt = Date.now() - 100;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.01;
    vadState.uttFrames = 2;
    vadState.uttPlaybackFrames = 0;
    vadState.uttPlaybackLevelSum = 0;
    vadPreRoll = [new Float32Array(480).fill(0.01)];
    const firstFragmentTurnBefore = currentVoiceTurnId;
    await onVadSpeechOnset();
    const firstFragmentSession = activeOracleVoiceInputSession;
    vadPushAudio(new Float32Array(480).fill(0.01));
    vadState.speechStartedAt = Date.now() - 100;
    vadState.lastSpeechSampleAt = Date.now();
    vadState.uttPeakRms = 0.01;
    vadState.uttFrames = 2;
    const firstFragmentDecision = window.MS4DSP.shouldSubmitUtterance({
      speechMs: 100,
      peakRms: 0.01,
      onsetThreshold: vadAdaptiveThresholds().onset,
      playbackActiveFraction: 0,
      playbackMeanLevel: 0,
    });
    firstFragmentSession.speechEnd();
    await onVadSpeechOffset();
    await delay(60);
    const firstPhraseRejection = {
      decision: firstFragmentDecision,
      requests: turnRequests,
      turnIdDelta: currentVoiceTurnId - firstFragmentTurnBefore,
      activeSessionCleared: activeOracleVoiceInputSession === null,
      inspect: firstFragmentSession.inspect(),
    };
    vadState.enabled = false;
    vadState.phase = 'silence';
    vadPreRoll = [];

    let virtualLongRun = null;
    if (loaderPresent && adapterPresent) {
      turnRequests = 0;
      transcribeRequests = 0;
      capturedTurn = null;
      let asrInFlight = 0;
      let maxAsrInFlight = 0;
      let asrCalls = 0;
      const pendingAsr = [];

      window.fetch = (url, options = {}) => {
        const target = String(url);
        if (target.startsWith('/voice/transcribe')) {
          asrCalls += 1;
          transcribeRequests += 1;
          asrInFlight += 1;
          maxAsrInFlight = Math.max(maxAsrInFlight, asrInFlight);
          return new Promise((resolve, reject) => {
            let settled = false;
            const finish = text => {
              if (settled) return;
              settled = true;
              asrInFlight -= 1;
              resolve({
                ok: true,
                status: 200,
                statusText: 'OK',
                text: async () => '',
                json: async () => ({
                  schema: 'Ms4VoiceTranscription.v1',
                  text,
                  model: 'browser-fixture-asr',
                }),
              });
            };
            if (options.signal) {
              options.signal.addEventListener('abort', () => {
                if (settled) return;
                settled = true;
                asrInFlight -= 1;
                reject(new DOMException('Aborted', 'AbortError'));
              }, {once: true});
            }
            pendingAsr.push({finish});
          });
        }
        if (target.startsWith('/voice/turn/stream')) {
          return Promise.resolve(captureTurnRequest(options));
        }
        return originalFetch(url, options);
      };

      const makeVirtualClock = () => {
        let now = 0;
        let nextId = 1;
        const timers = new Map();
        const clock = {
          now: () => now,
          setTimeout(callback, delayMs) {
            const id = nextId++;
            timers.set(id, {at: now + delayMs, callback});
            return id;
          },
          clearTimeout(id) { timers.delete(id); },
        };
        const advance = ms => {
          now += ms;
          while (true) {
            const due = [...timers.entries()]
              .filter(([, timer]) => timer.at <= now)
              .sort((a, b) => a[1].at - b[1].at || a[0] - b[0]);
            if (!due.length) break;
            const [id, timer] = due[0];
            timers.delete(id);
            timer.callback();
          }
        };
        return {clock, advance, now: () => now};
      };
      const settle = async () => {
        await Promise.resolve();
        await Promise.resolve();
        await delay(0);
      };
      const resolveNextAsr = async text => {
        const pending = pendingAsr.shift();
        if (!pending) throw new Error('no pending ASR request to resolve');
        pending.finish(text);
        await settle();
      };
      const hypothesis = call => {
        if (call === 1) return 'alpha sentinel repeated phrase';
        if (call === 2) return 'repeated phrase beta sentinel repeated phrase';
        return 'repeated phrase segment ' + call;
      };

      const virtual = makeVirtualClock();
      const turnBefore = currentVoiceTurnId;
      const session = createOracleVoiceInputSession({
        mode: 'browser-test',
        sourceLabel: 'browser-long-input',
        clock: virtual.clock,
      });
      const started = session.start();
      const block = new Float32Array(48_000 * 5);
      block.fill(0.04);
      let virtualDurationMs = 0;

      session.appendPcm(block);
      virtualDurationMs += 5000;
      virtual.advance(5000);
      await settle();
      for (let i = 0; i < 4; i += 1) {
        session.appendPcm(block);
        virtualDurationMs += 5000;
        virtual.advance(5000);
        await settle();
      }
      const slowAsrSerialized = asrCalls === 1 && pendingAsr.length === 1 && maxAsrInFlight === 1;
      await resolveNextAsr(hypothesis(1));
      await waitFor(() => pendingAsr.length === 1, 'bounded-input:coalesced-asr', 1000);
      await resolveNextAsr(hypothesis(2));

      while (virtualDurationMs < 3 * 60_000) {
        session.appendPcm(block);
        virtualDurationMs += 5000;
        virtual.advance(5000);
        await settle();
        if (pendingAsr.length) await resolveNextAsr(hypothesis(asrCalls));
      }
      while (pendingAsr.length) await resolveNextAsr(hypothesis(asrCalls));

      const stopPromise = session.stop();
      await waitFor(() => pendingAsr.length === 1, 'bounded-input:final-asr', 1000);
      await resolveNextAsr('repeated phrase omega sentinel');
      await stopPromise;
      await waitFor(() => turnRequests === 1, 'bounded-input:json-turn', 1000);
      await delay(60);
      const inspect = session.inspect();
      virtualLongRun = {
        started,
        virtualDurationMs,
        slowAsrSerialized,
        asrCalls,
        maxAsrInFlight,
        turnRequests,
        turnIdDelta: currentVoiceTurnId - turnBefore,
        bodyKeys: capturedTurn ? capturedTurn.bodyKeys : [],
        envelope: capturedTurn ? capturedTurn.envelope : null,
        contentType: capturedTurn ? capturedTurn.contentType : '',
        inspect,
      };
    }

    window.fetch = originalFetch;
    readOraclePlaybackSignal = originalReadOraclePlaybackSignal;
    if (originalVadAck === null) localStorage.removeItem('ms4_voice_vad_ack');
    else localStorage.setItem('ms4_voice_vad_ack', originalVadAck);
    if (typeof cancelActiveOracleVoiceInputSession === 'function') {
      await cancelActiveOracleVoiceInputSession('browser-long-input-final-cleanup');
    }
    bargeIn('bounded input browser probe cleanup', {
      halfContext: false, playAck: false,
    });
    setOracleStageState('idle', 'Ready.', '');
    return {
      loaderPresent,
      adapterPresent,
      legacyRecordedChunksPresent: (
        typeof recordedChunks !== 'undefined' || Object.prototype.hasOwnProperty.call(vadState, 'recordedChunks')
      ),
      ordinaryPause,
      rejectedTailContinuation,
      firstPhraseRejection,
      virtualLongRun,
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS + 10_000 });
  longInputSession.servedHtmlSha256 = servedHtmlSha256;
  longInputSession.voiceInputSessionSha256 = voiceInputSessionSha256;
  assert.equal(
    longInputSession.ordinaryPause.requestsAfterOrdinaryPause,
    0,
    'an ordinary 1.5 second sentence pause must not finalize or submit the served VAD turn',
  );
  assert.equal(longInputSession.loaderPresent, true, 'served page must load the accepted bounded session core');
  assert.equal(longInputSession.adapterPresent, true, 'served page must expose the production bounded adapter');
  assert.equal(longInputSession.legacyRecordedChunksPresent, false, 'served capture must not retain unbounded recordedChunks');
  assert.equal(longInputSession.ordinaryPause.requestsAfterFinalSilence, 1);
  assert.equal(longInputSession.ordinaryPause.bridgeAfterSpeechEnd, null,
    'VAD speechEnd must not arm feedback while continuation is still possible');
  assert.equal(longInputSession.ordinaryPause.continuationResumed, true);
  assert.equal(longInputSession.ordinaryPause.bridgeAfterResume, null,
    'resuming within final silence must leave the thinking bridge unarmed');
  assert.equal(longInputSession.ordinaryPause.finalSilenceRearmed, true);
  assert.equal(longInputSession.ordinaryPause.bridgeAfterFinalSpeechEnd, null,
    're-armed silence remains recoverable until the full 3 second timer commits');
  assert.ok(longInputSession.ordinaryPause.bridgeAtTranscribeStart,
    'committed VAD finalization must arm the bridge before final ASR fetch begins');
  assert.ok(
    ['armed', 'active'].includes(longInputSession.ordinaryPause.bridgeAtTranscribeStart.phase),
    'the final-ASR fetch must observe a live, turn-owned wait bridge',
  );
  assert.equal(
    longInputSession.ordinaryPause.bridgeAtTranscribeStart.turnId,
    longInputSession.ordinaryPause.turnIdBefore + 1,
    'the final-ASR bridge must belong to the VAD turn that crossed final silence',
  );
  assert.ok(longInputSession.ordinaryPause.transcribeAfterFinalSilenceMs >= 2_900,
    'final ASR must not begin before the 3 second continuation window commits');
  assert.equal(longInputSession.ordinaryPause.turnIdDelta, 1, 'one bounded session owns one onset barge');
  assert.equal(longInputSession.ordinaryPause.contentType, 'application/json');
  assert.deepEqual(longInputSession.ordinaryPause.bodyKeys, [
    'schema', 'source', 'transcript', 'transcription_model',
  ]);
  assert.equal(longInputSession.rejectedTailContinuation.continuationPauseMs, 1500);
  assert.equal(longInputSession.rejectedTailContinuation.finalSilenceMs, 3000);
  assert.equal(longInputSession.rejectedTailContinuation.acceptedPhraseArmed, true);
  assert.equal(longInputSession.rejectedTailContinuation.trailingDecision.submit, false);
  assert.equal(longInputSession.rejectedTailContinuation.requestsAfterRejectedTail, 0);
  assert.equal(
    longInputSession.rejectedTailContinuation.requestsAfterFinalSilence,
    1,
    'rejecting a later short/weak continuation must retain and submit prior accepted speech',
  );
  assert.equal(longInputSession.rejectedTailContinuation.turnIdDelta, 1);
  assert.equal(longInputSession.rejectedTailContinuation.inspect.bargeCount, 1);
  assert.equal(longInputSession.rejectedTailContinuation.inspect.submissions, 1);
  assert.equal(longInputSession.rejectedTailContinuation.transcript, 'accepted phrase sentinel');
  assert.equal(longInputSession.firstPhraseRejection.decision.submit, false);
  assert.equal(longInputSession.firstPhraseRejection.requests, 0);
  assert.equal(longInputSession.firstPhraseRejection.turnIdDelta, 1);
  assert.equal(longInputSession.firstPhraseRejection.activeSessionCleared, true);
  assert.equal(longInputSession.firstPhraseRejection.inspect.state, 'cancelled');
  assert.equal(longInputSession.firstPhraseRejection.inspect.submissions, 0);
  assert.equal(longInputSession.virtualLongRun.slowAsrSerialized, true);
  assert.equal(longInputSession.virtualLongRun.maxAsrInFlight, 1);
  assert.equal(longInputSession.virtualLongRun.turnRequests, 1);
  assert.equal(longInputSession.virtualLongRun.turnIdDelta, 1);
  assert.equal(longInputSession.virtualLongRun.inspect.submissions, 1);
  assert.equal(longInputSession.virtualLongRun.inspect.timerCount, 0);
  assert.ok(longInputSession.virtualLongRun.inspect.maxObservedPcmSamples <= 48_000 * 30);
  assert.ok(longInputSession.virtualLongRun.inspect.maxObservedTranscriptChars <= 65_536);

  vad = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const originalFetch = window.fetch;
    const originalReadOraclePlaybackSignal = readOraclePlaybackSignal;
    readOraclePlaybackSignal = () => ({active: false, level: 0, voice: 0, air: 0});
    const encoder = new TextEncoder();
    let voiceStreamRequests = 0;
    const doneFrames = encoder.encode('event: transcript\\ndata: {"text":"vad offset","asr_ms":1}\\n\\n' + 'event: text_delta\\ndata: {"text":"VAD offset reply."}\\n\\n' + 'event: done\\ndata: {"session_id":"vad-offset","reply_text":"VAD offset reply.","metrics":{"audio_chunks":0,"audio_client_written":0,"audio_errors":0,"total_ms":5}}\\n\\n');
    const makeReader = bytes => { let sent = false; return { read() { if (!sent) { sent = true; return Promise.resolve({ value: bytes, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } }; };
    window.fetch = (url, options = {}) => {
      if (String(url).startsWith('/voice/transcribe')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: async () => '',
          json: async () => ({
            schema: 'Ms4VoiceTranscription.v1',
            text: 'vad offset',
            model: 'browser-fixture-asr',
          }),
        });
      }
      if (String(url).startsWith('/voice/turn/stream')) {
        voiceStreamRequests += 1;
        return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => makeReader(doneFrames) } });
      }
      return originalFetch(url, options);
    };
    localStorage.setItem('ms4_voice_vad_ack', 'false');
    localStorage.setItem('ms4_voice_vad_adaptive', 'true');
    const analyserStub = { fftSize: 2048, __amp: 0, getFloatTimeDomainData(buf) { buf.fill(this.__amp); } };
    let seq = 1;
    const resetVad = async () => {
      if (typeof cancelActiveOracleVoiceInputSession === 'function') {
        await cancelActiveOracleVoiceInputSession('vad-lifecycle-reset');
      }
      vadState.enabled = true;
      vadState.analyser = analyserStub;
      vadState.phase = 'silence';
      vadState.smoothedRms = 0;
      vadState.frameClipped = false;
      vadState.recordedChunks = [];
      vadState.speechStartedAt = 0;
      vadState.lastSpeechSampleAt = 0;
      vadState.sourceSampleRate = 48000;
      vadState.lastTransitionAt = Date.now();
      vadState.noiseFloor = 0.0002;
      vadState.uttPeakRms = 0;
      vadState.uttFrames = 0;
      vadState.uttPlaybackFrames = 0;
      vadState.uttPlaybackLevelSum = 0;
      vadState.echoReference = null;
      vadState.lastPlaybackAt = 0;
      vadPreRoll = [];
      resetOraclePlaybackSignal();
    };
    const capturedSamples = () => {
      if (typeof inspectActiveOracleVoiceInputSession === 'function') {
        const inspected = inspectActiveOracleVoiceInputSession();
        if (inspected && Number.isFinite(inspected.pcmSamples)) return inspected.pcmSamples;
      }
      return (vadState.recordedChunks || []).length;
    };
    const pushTick = () => { const f = new Float32Array(160); f.fill(analyserStub.__amp >= 0.02 ? 0.05 : 0.0); f[0] = seq++; vadPushAudio(f); vadTick(); };
    const runUntil = (want, maxTicks) => { let n = 0; while (vadState.phase !== want && n++ < maxTicks) pushTick(); return vadState.phase === want; };
    const confirmSpeech = () => { analyserStub.__amp = 0.08; const reached = runUntil('maybe_speech', 40); for (let i = 0; i < 4; i += 1) pushTick(); vadState.lastTransitionAt = Date.now() - (vadMinSpeechMs() + 80); pushTick(); return reached && vadState.phase === 'speech'; };
    await resetVad();
    const aOnset = confirmSpeech();
    const aTurnAtSpeech = currentVoiceTurnId;
    for (let i = 0; i < 3; i += 1) pushTick();
    const aChunksSpeech = capturedSamples();
    analyserStub.__amp = 0.0;
    const aSawMaybeSilence = runUntil('maybe_silence', 60);
    for (let i = 0; i < 2; i += 1) pushTick();
    const aChunksPause = capturedSamples();
    analyserStub.__amp = 0.08;
    const aResumed = runUntil('speech', 60);
    const aTurnAtResume = currentVoiceTurnId;
    const aChunksResume = capturedSamples();
    const aRequests = voiceStreamRequests;
    voiceStreamRequests = 0;
    await resetVad();
    const bTurnBeforeOnset = currentVoiceTurnId;
    const bOnset = confirmSpeech();
    for (let i = 0; i < 3; i += 1) pushTick();
    vadState.speechStartedAt = Date.now() - 500;
    vadState.lastSpeechSampleAt = Date.now();
    analyserStub.__amp = 0.0;
    const bSawMaybeSilence = runUntil('maybe_silence', 60);
    const bTurnBeforeOffset = currentVoiceTurnId;
    const bHangover = vadEffectiveHangoverMs();
    vadState.lastTransitionAt = Date.now() - (bHangover + 120);
    pushTick();
    await delay(80);
    const bRequestsAfterHangover = voiceStreamRequests;
    const bTurnAfterHangover = currentVoiceTurnId;
    const bPhaseAfterOffset = vadState.phase;
    if (typeof createOracleVoiceInputSession === 'function') {
      await waitFor(() => voiceStreamRequests >= 1, 'vad:final-silence-request', 4500);
    }
    const bTurnAfterFinalSilence = currentVoiceTurnId;
    for (let i = 0; i < 10; i += 1) pushTick();
    const bRequests = voiceStreamRequests;
    const bTurnAfterExtra = currentVoiceTurnId;
    const base = vadHangoverMs(); const floor = VAD_ADAPTIVE_FLOOR_MS; const longGate = VAD_ADAPTIVE_LONG_SPEECH_MS;
    vadState.speechStartedAt = 0; vadState.lastSpeechSampleAt = 0; const adaptiveShort = vadEffectiveHangoverMs();
    vadState.speechStartedAt = 0; vadState.lastSpeechSampleAt = longGate + 400; const adaptiveLong = vadEffectiveHangoverMs();
    window.fetch = originalFetch;
    readOraclePlaybackSignal = originalReadOraclePlaybackSignal;
    vadState.enabled = false; vadState.analyser = null; vadState.recordedChunks = []; vadPreRoll = []; vadState.phase = 'silence';
    if (typeof cancelActiveOracleVoiceInputSession === 'function') {
      await cancelActiveOracleVoiceInputSession('vad-lifecycle-cleanup');
    }
    bargeIn('vad lifecycle cleanup');
    setOracleStageState('idle', 'Ready.', '');
    return {
      resume: { onset: aOnset, sawMaybeSilence: aSawMaybeSilence, resumed: aResumed, turnAtSpeech: aTurnAtSpeech, turnAtResume: aTurnAtResume, requests: aRequests, chunksSpeech: aChunksSpeech, chunksPause: aChunksPause, chunksResume: aChunksResume },
      offset: { onset: bOnset, sawMaybeSilence: bSawMaybeSilence, hangoverMs: bHangover, turnBeforeOnset: bTurnBeforeOnset, turnBeforeOffset: bTurnBeforeOffset, turnAfterHangover: bTurnAfterHangover, turnAfterFinalSilence: bTurnAfterFinalSilence, turnAfterExtra: bTurnAfterExtra, requestsAfterHangover: bRequestsAfterHangover, requests: bRequests, phaseAfterOffset: bPhaseAfterOffset },
      adaptive: { base, floor, longGate, adaptiveShort, adaptiveLong },
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(vad.resume.requests, 0, 'a pre-offset continuation must not open a request');
  assert.equal(vad.resume.turnAtResume, vad.resume.turnAtSpeech, 'a mid-utterance resume must not advance the turn id');
  assert.ok(vad.resume.chunksResume > vad.resume.chunksPause);
  assert.equal(vad.offset.requestsAfterHangover, 0, 'VAD hangover alone must not submit before bounded final silence');
  assert.equal(vad.offset.requests, 1, 'bounded final silence must open exactly one request');
  assert.equal(vad.offset.turnAfterFinalSilence - vad.offset.turnBeforeOnset, 1, 'the whole session must advance the turn id exactly once at onset');
  assert.equal(vad.offset.turnAfterExtra, vad.offset.turnAfterFinalSilence);
  assert.equal(vad.offset.phaseAfterOffset, 'silence');
  assert.ok(vad.adaptive.adaptiveLong < vad.adaptive.base);
  assert.equal(vad.adaptive.adaptiveLong, Math.max(vad.adaptive.floor, Math.round(vad.adaptive.base * 0.65)));

  // ---- Finding 4: full-duplex enable/disable generation ownership. ----------
  fullDuplexRace = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const scratch = new (window.AudioContext || window.webkitAudioContext)();
    const realStream = () => scratch.createMediaStreamDestination().stream;
    const origAcquire = window.acquireMicStream;
    let deferred = [];
    window.acquireMicStream = () => new Promise(res => { deferred.push(() => res(realStream())); });
    if (vadState.enabled) disableFullDuplex({ silent: true });
    // (a) disable during a pending enable -> stale enable acquires nothing live.
    const pEnable1 = enableFullDuplex();
    disableFullDuplex({ silent: true });
    deferred[0]();
    await pEnable1;
    await delay(20);
    const staleCase = { enabled: vadState.enabled === true, pollTimer: vadState.pollTimer !== null };
    // (b) two concurrent enables -> exactly one owned stream/context.
    deferred = [];
    const capturedStreams = [];
    window.acquireMicStream = () => new Promise(res => { const s = realStream(); capturedStreams.push(s); deferred.push(() => res(s)); });
    const pEnableA = enableFullDuplex();
    const pEnableB = enableFullDuplex();
    deferred[0]();   // A resolves first, but B's generation already superseded it
    deferred[1]();   // B wins
    await Promise.all([pEnableA, pEnableB]);
    await delay(20);
    const winner = { enabled: vadState.enabled === true, hasContext: vadState.audioContext !== null, pollTimer: vadState.pollTimer !== null };
    const loserStreamEnded = capturedStreams.length === 2 && capturedStreams[0].getAudioTracks().every(t => t.readyState === 'ended');
    const winnerStreamLive = vadState.mediaStream && vadState.mediaStream.getAudioTracks().some(t => t.readyState === 'live');
    // Cleanup.
    disableFullDuplex({ silent: true });
    window.acquireMicStream = origAcquire;
    try { scratch.close(); } catch (e) {}
    return { staleCase, winner, loserStreamEnded, winnerStreamLive, disabledAfterCleanup: vadState.enabled === false };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(fullDuplexRace.staleCase.enabled, false, 'a disable during a pending enable must leave full-duplex OFF');
  assert.equal(fullDuplexRace.staleCase.pollTimer, false, 'a stale enable must install no poll timer');
  assert.equal(fullDuplexRace.winner.enabled, true, 'the winning enable must be active');
  assert.equal(fullDuplexRace.winner.pollTimer, true);
  assert.equal(fullDuplexRace.loserStreamEnded, true, 'the losing concurrent enable must close its acquired tracks');
  assert.equal(fullDuplexRace.winnerStreamLive, true, 'exactly one owned live stream must remain');
  assert.equal(fullDuplexRace.disabledAfterCleanup, true, 'disable must return full-duplex to OFF');

  // ---- Finding 5 (R4): deferred auto-arm cannot reopen the mic after an -----
  // explicit disable. The page-load auto-arm is bound to the enable generation
  // captured when it was scheduled; an explicit disable before it fires bumps
  // that generation, so the deferred arm is a permanent no-op (no getUserMedia).
  deferredAutoArm = await cdp.evaluate(`(async () => {
    ${PAGE_HELPERS}
    const FULL_DUPLEX_KEY = 'ms4_voice_full_duplex';
    disableFullDuplex({ silent: true });   // known-off baseline
    const origAcquire = window.acquireMicStream;
    let getUserMediaCalls = 0;
    const scratch = new (window.AudioContext || window.webkitAudioContext)();
    const realStream = () => scratch.createMediaStreamDestination().stream;
    window.acquireMicStream = () => { getUserMediaCalls += 1; return Promise.resolve(realStream()); };

    // (a) explicit DISABLE before the deferred timer fires wins permanently.
    localStorage.setItem(FULL_DUPLEX_KEY, 'true');   // persisted intent at schedule time
    const armGenA = vadState.enableGeneration;        // generation captured when the arm was scheduled
    localStorage.setItem(FULL_DUPLEX_KEY, 'false');  // user turns it off (settings toggle path)
    disableFullDuplex({ silent: true });              // explicit disable bumps the generation
    const ranA = window.__ms4RunDeferredFullDuplexAutoArm(armGenA);
    await delay(25);
    const disableWins = {
      ran: ranA,
      getUserMediaCalls,
      enabled: vadState.enabled === true,
      pollTimer: vadState.pollTimer !== null,
      hasContext: vadState.audioContext !== null,
      persisted: localStorage.getItem(FULL_DUPLEX_KEY),
    };

    // (b) intent still on and no intervening disable: the SAME guard permits a
    // legitimate deferred arm (proves the guard is not a blanket refusal).
    localStorage.setItem(FULL_DUPLEX_KEY, 'true');
    const armGenB = vadState.enableGeneration;
    const ranB = window.__ms4RunDeferredFullDuplexAutoArm(armGenB);
    await delay(40);
    const armWins = { ran: ranB, getUserMediaCalls, enabled: vadState.enabled === true };

    // Cleanup back to a known-off state.
    disableFullDuplex({ silent: true });
    localStorage.setItem(FULL_DUPLEX_KEY, 'false');
    window.acquireMicStream = origAcquire;
    try { scratch.close(); } catch (e) {}
    return { disableWins, armWins };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(deferredAutoArm.disableWins.ran, false, 'a deferred auto-arm after an explicit disable must be a no-op');
  assert.equal(deferredAutoArm.disableWins.getUserMediaCalls, 0, 'no getUserMedia after explicit disable');
  assert.equal(deferredAutoArm.disableWins.enabled, false, 'full-duplex must stay OFF after explicit disable');
  assert.equal(deferredAutoArm.disableWins.pollTimer, false, 'no poll timer after a suppressed deferred arm');
  assert.equal(deferredAutoArm.disableWins.hasContext, false, 'no audio context after a suppressed deferred arm');
  assert.equal(deferredAutoArm.disableWins.persisted, 'false', 'persisted intent stays off');
  assert.equal(deferredAutoArm.armWins.ran, true, 'a legitimate deferred arm (intent on, same generation) still arms');
  assert.equal(deferredAutoArm.armWins.getUserMediaCalls, 1, 'the legitimate arm acquires exactly one mic stream');
  assert.equal(deferredAutoArm.armWins.enabled, true);

  // ---- 2026-07-06 audio-stutter product repair: browser engine migration +
  // playback-underflow telemetry. The one-time v3 migration clears a pinned
  // ws_super so unpinned sessions land on the server REST default; a deliberate
  // post-migration opt-in is preserved. The underflow recorder reproduces the
  // measured production stutter cadence (10 gaps >= 40 ms, 4420.834 ms, 751 ms).
  productRepair = await cdp.evaluate(`(async () => {
    const migrateFn = window.__ms4MigrateTtsEnginePinV3;
    const underflowFn = window.__ms4RecordPlaybackUnderflow;
    const queryFn = window.__ms4VoiceQueryParams;
    if (!migrateFn || !underflowFn || !queryFn) return { available: false };
    const ENGINE_KEY = 'ms4_tts_engine';
    const V3 = 'ms4_engine_migrated_v3';
    const saveEngine = localStorage.getItem(ENGINE_KEY);
    const saveV3 = localStorage.getItem(V3);
    // (1) one-time migration clears a pinned ws_super and sets the marker.
    localStorage.setItem(ENGINE_KEY, 'ws_super');
    localStorage.removeItem(V3);
    const clearedWsSuper = migrateFn();
    const afterMigratePin = localStorage.getItem(ENGINE_KEY);
    const markerSet = localStorage.getItem(V3);
    const queryUnpinnedHasEngine = /(^|&)engine=/.test(queryFn());
    // (2) idempotent: a second run does not clear again.
    const secondRunCleared = migrateFn();
    // (3) deliberate post-migration opt-in is preserved (marker already set).
    localStorage.setItem(ENGINE_KEY, 'ws_super');
    const optInCleared = migrateFn();
    const optInPin = localStorage.getItem(ENGINE_KEY);
    const queryOptInHasWsSuper = /(^|&)engine=ws_super(&|$)/.test(queryFn());
    // restore prior localStorage.
    if (saveEngine === null) localStorage.removeItem(ENGINE_KEY); else localStorage.setItem(ENGINE_KEY, saveEngine);
    if (saveV3 === null) localStorage.removeItem(V3); else localStorage.setItem(V3, saveV3);
    // (4) underflow telemetry reproduces the production stutter cadence.
    const t = { scheduledChunks: 0, underflowGapCount: 0, underflowGapTotalMs: 0, underflowGapMaxMs: 0 };
    underflowFn(t, 0, 5, 0);                // first chunk (prior=0) -> excluded
    underflowFn(t, 10, 10 + 0.024292, 1);   // 24.292 ms (< 40) -> excluded
    const prodGaps = [438, 345, 500, 751, 329, 584.542, 347.292, 360, 422, 344];
    for (const g of prodGaps) underflowFn(t, 100, 100 + g / 1000, 5);
    // (5) settingsApplyToForm must PRESERVE a deliberate post-migration ws_super
    // opt-in when the server default is rest (the old generic-mismatch clear
    // erased it on every refresh).
    let optInSurvivesRefresh = 'unavailable';
    let queryAfterRefreshHasWsSuper = false;
    if (typeof settingsApplyToForm === 'function') {
      localStorage.setItem(ENGINE_KEY, 'ws_super');
      localStorage.setItem(V3, 'true');  // already migrated -> this is a deliberate opt-in
      settingsApplyToForm({ voice: { tts_engine_default: 'rest', tts_voice_known: [], tts_voice_default: 'alloy', tts_model_default: 'tts-1' } });
      optInSurvivesRefresh = localStorage.getItem(ENGINE_KEY);
      queryAfterRefreshHasWsSuper = /(^|&)engine=ws_super(&|$)/.test(queryFn());
      if (saveEngine === null) localStorage.removeItem(ENGINE_KEY); else localStorage.setItem(ENGINE_KEY, saveEngine);
      if (saveV3 === null) localStorage.removeItem(V3); else localStorage.setItem(V3, saveV3);
    }
    // (6) A measured playback underflow (>=40 ms) must GATE the terminal:
    // finalizeAudibleVerdict must NOT return 'done' for an audible, error-free
    // turn that stuttered; it must return an explicit degraded_underflow while
    // retaining the count/total/max telemetry.
    let underflowTerminal = 'unavailable';
    let cleanTerminal = 'unavailable';
    let stutterTelemetryRetained = false;
    if (typeof finalizeAudibleVerdict === 'function') {
      const mkTurn = (gapCount, total, max) => ({
        completed: true, audibleFinalized: false, terminalKind: 'awaiting_audible',
        turnId: -987654,  // never current -> no UI side effects, not recorded
        telemetry: { firstNonSilentOnsetMs: 100, onsetRms: 0.5, audioErrors: 0, decodeErrors: 0,
          scheduledChunks: 2, underflowGapCount: gapCount, underflowGapTotalMs: total, underflowGapMaxMs: max },
      });
      const stutter = mkTurn(1, 500, 500);
      finalizeAudibleVerdict(stutter);
      underflowTerminal = stutter.terminalKind;
      stutterTelemetryRetained = (stutter.telemetry.underflowGapCount === 1
        && stutter.telemetry.underflowGapTotalMs === 500 && stutter.telemetry.underflowGapMaxMs === 500);
      const clean = mkTurn(0, 0, 0);
      finalizeAudibleVerdict(clean);
      cleanTerminal = clean.terminalKind;
    }
    return {
      available: true,
      clearedWsSuper, afterMigratePin, markerSet, queryUnpinnedHasEngine,
      secondRunCleared, optInCleared, optInPin, queryOptInHasWsSuper,
      optInSurvivesRefresh, queryAfterRefreshHasWsSuper,
      underflowTerminal, cleanTerminal, stutterTelemetryRetained,
      underflow: {
        count: t.underflowGapCount,
        totalMs: Math.round(t.underflowGapTotalMs * 1000) / 1000,
        maxMs: Math.round(t.underflowGapMaxMs * 1000) / 1000,
      },
    };
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  assert.equal(productRepair.available, true, 'index.html must expose the v3 migration, underflow recorder, and query helper');
  assert.equal(productRepair.clearedWsSuper, true, 'the one-time v3 migration must clear a pinned ws_super');
  assert.equal(productRepair.afterMigratePin, null, 'after migration the ws_super pin is cleared');
  assert.equal(productRepair.markerSet, 'true', 'the v3 migration marker is set so it runs once');
  assert.equal(productRepair.queryUnpinnedHasEngine, false, 'an unpinned session sends no engine param -> server REST default');
  assert.equal(productRepair.secondRunCleared, false, 'the migration is idempotent (does not re-clear)');
  assert.equal(productRepair.optInCleared, false, 'a deliberate post-migration ws_super opt-in is preserved by the migration');
  assert.equal(productRepair.optInPin, 'ws_super', 'the deliberate opt-in pin survives the migration');
  assert.equal(productRepair.queryOptInHasWsSuper, true, 'a deliberate ws_super opt-in is still sent as engine=ws_super');
  assert.equal(productRepair.optInSurvivesRefresh, 'ws_super', 'settingsApplyToForm must NOT erase a deliberate ws_super opt-in on refresh');
  assert.equal(productRepair.queryAfterRefreshHasWsSuper, true, 'after a settings refresh the ws_super opt-in still ships as engine=ws_super');
  assert.equal(productRepair.underflowTerminal, 'degraded_underflow', 'a >=40 ms playback underflow must block clean done -> degraded_underflow');
  assert.equal(productRepair.cleanTerminal, 'done', 'a gapless audible turn still finalizes done');
  assert.equal(productRepair.stutterTelemetryRetained, true, 'count/total/max underflow telemetry is retained on the degraded_underflow turn');
  assert.equal(productRepair.underflow.count, 10, 'underflow telemetry counts the 10 production audible gaps (>=40 ms)');
  assert.ok(Math.abs(productRepair.underflow.totalMs - 4420.834) < 1.0, 'underflow total ms matches the measured 4420.834 ms deficit');
  assert.equal(productRepair.underflow.maxMs, 751, 'underflow largest gap matches the measured 751 ms');

  // ---- 2026-07-06 browser-audio REST-batch prebuffer fix. REST batch/parallel
  // TTS synthesizes sibling chunks in parallel; they can arrive out of order. A
  // lone REST first chunk must NOT flush on the target duration or the 400ms
  // escape hatch while a sibling is still outstanding (it would play and END
  // before the sibling arrives -> the measured underflow). It must hold until the
  // SECOND chunk (both scheduled contiguous, zero gap) or a one-chunk 'done'. WS
  // Super keeps its low-latency start; post-terminal late media is rejected; a
  // barge clears the held queue; an abnormal REST end (SSE error) must not strand
  // the held chunk or let it play after the turn terminal.
  {
    restBatch = await cdp.evaluate(`(async () => {
      const delay = ms => new Promise(r => setTimeout(r, ms));
      const waitFor = async (fn, label, timeout) => { const start = Date.now(); for (;;) { let v; try { v = fn(); } catch (e) { v = null; } if (v) return v; if (Date.now() - start > (timeout || 4000)) throw new Error('timeout waiting for ' + label); await delay(30); } };
      if (typeof enqueueAudioChunk !== 'function' || typeof flushPrebuffer !== 'function' || typeof submitWavBlobAsVoiceTurn !== 'function') return { available: false };
      const PROOF = '${proofAudioBase64}';
      const reset = () => { try { if (typeof prebufferTimer !== 'undefined' && prebufferTimer) clearTimeout(prebufferTimer); } catch (e) {} playbackRunStarted = false; prebufferQueue = []; try { prebufferTimer = null; } catch (e) {} nextChunkStartTime = 0; };
      const mkTurn = () => ({ turnId: currentVoiceTurnId, completed: false, terminalKind: null, audibleFinalized: false, startedAtPerf: performance.now(), serverMetrics: {}, ttsEngine: null, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0, idleAfterPlayback: null, telemetry: { firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null, firstScheduledMs: null, firstScheduledContextTime: null, firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 0, sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0, audioErrors: 0, decodeErrors: 0, underflowGapCount: 0, underflowGapTotalMs: 0, underflowGapMaxMs: 0 } });

      // (A) Delayed REST pair (~600ms inter-chunk): lone chunk must NOT start;
      // chunk 2 arms at most the bounded startup guard, then both play contiguously.
      reset(); const rt = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', rt.turnId, rt, 'rest');
      const restStartedAfterChunk1 = playbackRunStarted; const restDepthAfterChunk1 = prebufferQueue.length;
      await delay(600);
      const restStartedAfterWait = playbackRunStarted;
      await enqueueAudioChunk(PROOF, 'audio/wav', rt.turnId, rt, 'rest');
      const restStartedImmediatelyAfterChunk2 = playbackRunStarted;
      await waitFor(() => playbackRunStarted && rt.telemetry.scheduledChunks === 2, 'rest:bounded-pair-release', 1000);
      const restStartedAfterChunk2 = playbackRunStarted; const restScheduled = rt.telemetry.scheduledChunks;
      const restUnderflow = { count: rt.telemetry.underflowGapCount, total: rt.telemetry.underflowGapTotalMs, max: rt.telemetry.underflowGapMaxMs };

      // (A2) Adaptive REST runway: keep a short first fragment held, but start a
      // long first fragment immediately because it already masks the measured
      // sibling-arrival delay. Drive the real decode/scheduler path with real
      // AudioBuffers whose durations match the physical Oracle evidence.
      const adaptiveCtx = getPlaybackCtx();
      const originalAdaptiveDecode = adaptiveCtx.decodeAudioData.bind(adaptiveCtx);
      const enqueueWithDuration = async (durationSec, turn, index) => {
        adaptiveCtx.decodeAudioData = () => Promise.resolve(
          adaptiveCtx.createBuffer(1, Math.ceil(durationSec * adaptiveCtx.sampleRate), adaptiveCtx.sampleRate)
        );
        try {
          return await enqueueAudioChunk(PROOF, 'audio/wav', turn.turnId, turn, 'rest', index);
        } finally {
          adaptiveCtx.decodeAudioData = originalAdaptiveDecode;
        }
      };
      reset(); const shortRunwayTurn = mkTurn();
      await enqueueWithDuration(1.485, shortRunwayTurn, 0);
      const shortRunwayHeld = !playbackRunStarted && prebufferQueue.length === 1;
      try { bargeIn(null, { halfContext: false, playAck: false }); } catch (e) {}

      reset(); const longRunwayTurn = mkTurn();
      await enqueueWithDuration(9.5, longRunwayTurn, 0);
      const longRunwayStarted = playbackRunStarted;
      const longRunwayDepth = prebufferQueue.length;
      const longRunwayScheduledAfterFirst = longRunwayTurn.telemetry.scheduledChunks;
      await delay(2200);
      await enqueueWithDuration(4.736, longRunwayTurn, 1);
      const longRunwayScheduledAfterSecond = longRunwayTurn.telemetry.scheduledChunks;
      const longRunwayUnderflow = longRunwayTurn.telemetry.underflowGapCount;
      try { bargeIn(null, { halfContext: false, playAck: false }); } catch (e) {}

      // (B) One-chunk REST flushes once on 'done'.
      reset(); const ot = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', ot.turnId, ot, 'rest');
      const oneStartedBeforeDone = playbackRunStarted; const oneDepthBeforeDone = prebufferQueue.length;
      if (!playbackRunStarted && prebufferQueue.length) await flushPrebuffer();
      const oneStartedAfterDone = playbackRunStarted; const oneScheduled = ot.telemetry.scheduledChunks;

      // (D) Post-terminal late media rejected.
      reset(); const tt = mkTurn(); tt.terminalKind = 'done';
      const lateAccepted = await enqueueAudioChunk(PROOF, 'audio/wav', tt.turnId, tt, 'rest');
      const lateDepth = prebufferQueue.length; const lateScheduled = tt.telemetry.scheduledChunks;

      // (Abort) barge clears the held REST queue.
      reset(); const at = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', at.turnId, at, 'rest');
      const abortDepthBefore = prebufferQueue.length;
      try { bargeIn(null); } catch (e) {}
      const abortDepthAfter = prebufferQueue.length;

      // (E) WS Super early-start baseline unchanged: lone chunk flushes <=450ms.
      reset(); const wt = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', wt.turnId, wt, 'ws_super');
      await delay(450);
      const wsStartedAfterWait = playbackRunStarted;

      // (C) REAL REST error turn: chunk 0 held, then event:error (no chunk 2, no done). finally must clear + never play.
      reset();
      const originalFetch = window.fetch; const encoder = new TextEncoder();
      const makeReader = text => { const bytes = encoder.encode(text); let sent = false; return { read(){ if (!sent){ sent = true; return Promise.resolve({ value: bytes, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel(){ return Promise.resolve(); } }; };
      const errSse = 'event: transcript\\ndata: {"text":"err","asr_ms":1}\\n\\n' + 'event: text_delta\\ndata: {"text":"partial"}\\n\\n' + 'event: audio_chunk\\ndata: {"index":0,"audio_base64":"' + PROOF + '","audio_mime":"audio/wav"}\\n\\n' + 'event: error\\ndata: {"error":"synthetic REST failure"}\\n\\n';
      window.fetch = (url, options = {}) => { if (String(url).startsWith('/voice/turn/stream')) return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => makeReader(errSse) } }); return originalFetch(url, options); };
      const preErrTurn = currentVoiceTurnId; let errThrew = false;
      try { await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'rest-error' }); } catch (e) { errThrew = true; }
      let errState = null; try { errState = await waitFor(() => { const st = window.__ms4OracleVoiceAcceptanceState ? window.__ms4OracleVoiceAcceptanceState() : null; const t = st && st.lastVoiceTurn; return t && t.turnId >= preErrTurn && t.terminalKind && t.terminalKind !== 'awaiting_audible' ? st : null; }, 'rest:error-terminal', 6000); } catch (e) { errState = null; }
      await delay(500);
      window.fetch = originalFetch;
      const errClearedQueue = prebufferQueue.length === 0;
      const errTerminal = errState && errState.lastVoiceTurn ? errState.lastVoiceTurn.terminalKind : null;
      const errScheduled = errState && errState.lastVoiceTurn ? errState.lastVoiceTurn.telemetry.scheduledChunks : null;

      // (Issue 1) terminalKind flips DURING decodeAudioData -> post-decode recheck rejects.
      reset(); const r1 = mkTurn();
      const ctxR = getPlaybackCtx();
      const origDecodeR = ctxR.decodeAudioData.bind(ctxR);
      ctxR.decodeAudioData = (ab) => { r1.terminalKind = 'done'; return origDecodeR(ab); };
      try { await enqueueAudioChunk(PROOF, 'audio/wav', r1.turnId, r1, 'rest', 0); } finally { ctxR.decodeAudioData = origDecodeR; }
      const raceScheduled = r1.telemetry.scheduledChunks;
      const raceDepth = prebufferQueue.length;

      // (Issue 2a) duplicate index dropped (not a 2nd chunk, no early flush).
      reset(); const d2 = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', d2.turnId, d2, 'rest', 0);
      const dupDepth1 = prebufferQueue.length;
      await enqueueAudioChunk(PROOF, 'audio/wav', d2.turnId, d2, 'rest', 0);
      const dupDepth2 = prebufferQueue.length;
      const dupStarted = playbackRunStarted;
      const dupScheduled = d2.telemetry.scheduledChunks;

      // (Issue 2b) out-of-order held; contiguous release when the gap fills.
      reset(); const o2 = mkTurn();
      await enqueueAudioChunk(PROOF, 'audio/wav', o2.turnId, o2, 'rest', 1);
      const oooDepth1 = prebufferQueue.length;
      const oooStarted1 = playbackRunStarted;
      await enqueueAudioChunk(PROOF, 'audio/wav', o2.turnId, o2, 'rest', 0);
      await waitFor(() => o2.telemetry.scheduledChunks === 2, 'rest:out-of-order-bounded-release', 1000);
      const oooScheduled = o2.telemetry.scheduledChunks;
      const oooUnderflow = o2.telemetry.underflowGapCount;

      // (Issue 3) WS->REST fallback revokes the armed WS 400ms escape-hatch timer.
      reset(); const f3 = mkTurn(); f3.ttsEngine = 'ws_super';
      const i3HasHelper = typeof applyStatusTtsEngine === 'function';
      prebufferTimer = setTimeout(() => { playbackRunStarted = true; }, 20);
      let i3Engine = null, i3TimerRevoked = false;
      if (i3HasHelper) { applyStatusTtsEngine(f3, 'tts_engine_fallback_after_zero_audio', { to: 'rest' }); i3Engine = f3.ttsEngine; i3TimerRevoked = (prebufferTimer === null); }
      await delay(60);
      const i3StaleFired = playbackRunStarted;
      if (prebufferTimer) { clearTimeout(prebufferTimer); prebufferTimer = null; }

      reset();
      return { available: true, restStartedAfterChunk1, restDepthAfterChunk1, restStartedAfterWait, restStartedImmediatelyAfterChunk2, restStartedAfterChunk2, restScheduled, restUnderflow, shortRunwayHeld, longRunwayStarted, longRunwayDepth, longRunwayScheduledAfterFirst, longRunwayScheduledAfterSecond, longRunwayUnderflow, oneStartedBeforeDone, oneDepthBeforeDone, oneStartedAfterDone, oneScheduled, lateAccepted, lateDepth, lateScheduled, abortDepthBefore, abortDepthAfter, wsStartedAfterWait, errThrew, errClearedQueue, errTerminal, errScheduled, raceScheduled, raceDepth, dupDepth1, dupDepth2, dupStarted, dupScheduled, oooDepth1, oooStarted1, oooScheduled, oooUnderflow, i3HasHelper, i3Engine, i3TimerRevoked, i3StaleFired };
    })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
    assert.ok(restBatch, 'restBatch evaluate returned');
    assert.equal(restBatch.available, true, 'restBatch: prebuffer + turn API available');
    assert.equal(restBatch.restStartedAfterChunk1, false, 'REST: lone chunk 1 does not start playback');
    assert.equal(restBatch.restDepthAfterChunk1, 1, 'REST: chunk 1 held in prebuffer');
    assert.equal(restBatch.restStartedAfterWait, false, 'REST: still not started after 600ms (no 400ms escape hatch / target flush)');
    assert.equal(restBatch.restStartedImmediatelyAfterChunk2, false, 'REST: deficient pair receives the bounded startup guard');
    assert.equal(restBatch.restStartedAfterChunk2, true, 'REST: deficient pair flushes within the bounded startup guard');
    assert.equal(restBatch.restScheduled, 2, 'REST: both chunks scheduled');
    assert.equal(restBatch.restUnderflow.count, 0, 'REST: contiguous schedule, zero underflow gaps');
    assert.equal(restBatch.restUnderflow.total, 0, 'REST: zero total underflow ms');
    assert.equal(restBatch.restUnderflow.max, 0, 'REST: zero max underflow ms');
    assert.equal(restBatch.shortRunwayHeld, true, 'REST adaptive: 1.485s first fragment remains held');
    assert.equal(restBatch.longRunwayStarted, true, 'REST adaptive: 9.5s first fragment starts immediately');
    assert.equal(restBatch.longRunwayDepth, 0, 'REST adaptive: long first fragment is not stranded in prebuffer');
    assert.equal(restBatch.longRunwayScheduledAfterFirst, 1, 'REST adaptive: long first fragment is scheduled once');
    assert.equal(restBatch.longRunwayScheduledAfterSecond, 2, 'REST adaptive: delayed sibling schedules behind the runway');
    assert.equal(restBatch.longRunwayUnderflow, 0, 'REST adaptive: delayed sibling produces zero underflow');
    assert.equal(restBatch.oneStartedBeforeDone, false, 'REST one-chunk: not started before done');
    assert.equal(restBatch.oneDepthBeforeDone, 1, 'REST one-chunk: held before done');
    assert.equal(restBatch.oneStartedAfterDone, true, 'REST one-chunk: flushes on done');
    assert.equal(restBatch.oneScheduled, 1, 'REST one-chunk: exactly one scheduled');
    assert.equal(restBatch.lateAccepted, true, 'post-terminal late media dropped (true = handled, not a decode failure)');
    assert.equal(restBatch.lateDepth, 0, 'post-terminal late media not enqueued');
    assert.equal(restBatch.lateScheduled, 0, 'post-terminal late media not scheduled');
    assert.equal(restBatch.abortDepthBefore, 1, 'abort: chunk held before barge');
    assert.equal(restBatch.abortDepthAfter, 0, 'abort: barge clears held REST queue');
    assert.equal(restBatch.wsStartedAfterWait, true, 'WS Super: lone chunk flushes promptly (<=450ms), baseline unchanged');
    assert.equal(restBatch.errClearedQueue, true, 'error: held queue/timer cleared, not stranded');
    assert.equal(restBatch.errTerminal, 'error', 'error: turn terminal is error');
    assert.equal(restBatch.errScheduled, 0, 'error: held chunk never scheduled/played');
    assert.equal(restBatch.raceScheduled, 0, 'Issue1: terminal-during-decode -> not scheduled');
    assert.equal(restBatch.raceDepth, 0, 'Issue1: terminal-during-decode -> not held');
    assert.equal(restBatch.dupDepth1, 1, 'Issue2: first indexed chunk held');
    assert.equal(restBatch.dupDepth2, 1, 'Issue2: duplicate index dropped (still 1, not flushed)');
    assert.equal(restBatch.dupStarted, false, 'Issue2: duplicate did not start playback');
    assert.equal(restBatch.dupScheduled, 0, 'Issue2: duplicate not scheduled');
    assert.equal(restBatch.oooDepth1, 0, 'Issue2: out-of-order chunk held as pending, not queued');
    assert.equal(restBatch.oooStarted1, false, 'Issue2: out-of-order lone chunk did not start');
    assert.equal(restBatch.oooScheduled, 2, 'Issue2: contiguous release schedules both in order');
    assert.equal(restBatch.oooUnderflow, 0, 'Issue2: contiguous release, zero underflow');
    assert.equal(restBatch.i3HasHelper, true, 'Issue3: applyStatusTtsEngine fallback/timer-revocation helper exists');
    assert.equal(restBatch.i3Engine, 'rest', 'Issue3: fallback sets ttsEngine=rest');
    assert.equal(restBatch.i3TimerRevoked, true, 'Issue3: fallback revokes the armed WS prebuffer timer');
    assert.equal(restBatch.i3StaleFired, false, 'Issue3: revoked timer never flushed the lone chunk');
  }

  // ---- 2026-07-07 R4 observability probe (ADDITIVE, NON-THROWING). -----------
  // Exercises index.html's REAL enqueueAudioChunk/flushPrebuffer on the exact
  // delayed-REST cadence (chunk 0 held, ~600ms gap, chunk 2 flushes both) and
  // REPORTS which additive per-chunk attribution fields the production turn
  // telemetry actually populates. It NEVER asserts, so the shared acceptance
  // harness stays green and still commits; the RED/GREEN assertions live in the
  // Python runner (test_oracle_browser_runtime.py) reading result.restObservability.
  // The turn is seeded with EMPTY candidate attribution fields so any populated
  // value can ONLY come from index.html production code (faithful absence proof).
  //
  // R2 P1 finding 1 REPAIR (R3): the correlation sub-probe NO LONGER mints the
  // turn/chunk ids. It loads a PRODUCER/RELAY FIXTURE -- built in Python by driving
  // the real audio_chunk payload shape through the REAL server.py _sse_event relay
  // and written to MS4_BROWSER_CORR_FIXTURE -- and feeds those EXACT server-relayed
  // SSE frames into the real submitWavBlobAsVoiceTurn/fetch/SSE parser path, then
  // reads the ids back off the browser turn telemetry. The harness reports the
  // fixture-provided ids (ECHOED, never minted here) and the browser-observed ids;
  // the Python runner asserts telemetry == fixture. Production ignores payload
  // turn_id/chunk_id, so the telemetry stays empty (RED). When the fixture env is
  // unset (standalone canonical run) the probe degrades to driven:false and never
  // throws, so the shared acceptance harness stays green.
  let corrFixture = null;
  try {
    const corrFixturePath = process.env.MS4_BROWSER_CORR_FIXTURE;
    if (corrFixturePath) {
      const parsed = JSON.parse(await readFile(corrFixturePath, 'utf8'));
      if (parsed && typeof parsed.sse_frames === 'string' && parsed.sse_frames.length) {
        corrFixture = {
          turnId: parsed.turn_id != null ? String(parsed.turn_id) : null,
          chunkIds: Array.isArray(parsed.chunk_ids) ? parsed.chunk_ids.map(String) : [],
          sseFrames: parsed.sse_frames,
          source: corrFixturePath,
        };
      }
    }
  } catch (e) {
    corrFixture = null;  // reported as fixtureProvided:false below; never throws
  }
  try {
    restObservability = await cdp.evaluate(`(async () => {
      const delay = ms => new Promise(r => setTimeout(r, ms));
      const waitForScheduler = async (fn, label, timeout) => { const start = Date.now(); for (;;) { let v; try { v = fn(); } catch (e) { v = null; } if (v) return v; if (Date.now() - start > (timeout || 1000)) throw new Error('timeout waiting for ' + label); await delay(10); } };
      if (typeof enqueueAudioChunk !== 'function' || typeof flushPrebuffer !== 'function') return { available: false, reason: 'prebuffer API missing' };
      const reset = () => { try { if (typeof prebufferTimer !== 'undefined' && prebufferTimer) clearTimeout(prebufferTimer); } catch (e) {} playbackRunStarted = false; prebufferQueue = []; try { prebufferTimer = null; } catch (e) {} nextChunkStartTime = 0; };
      const mkTurn = () => ({ turnId: currentVoiceTurnId, completed: false, terminalKind: null, audibleFinalized: false, startedAtPerf: performance.now(), serverMetrics: {}, ttsEngine: null, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0, serverTurnId: null, telemetry: { firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null, firstScheduledMs: null, firstScheduledContextTime: null, firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 0, sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0, audioErrors: 0, decodeErrors: 0, underflowGapCount: 0, underflowGapTotalMs: 0, underflowGapMaxMs: 0, chunkArrivalsMs: [], chunkDecodeMs: [], chunkDecodedDurationsSec: [], firstChunkHoldMs: null, holdReason: null, startupGuardRequestedMs: null, chunkServerIds: [], chunkRecords: {}, duplicateChunks: 0 } });

      // Exact delayed-REST cadence (mirrors restBatch A): lone chunk 1 held, ~600ms
      // gap, chunk 2 flushes both contiguous (zero underflow).
      reset(); const rt = mkTurn();
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', rt.turnId, rt, 'rest');
      const startedAfterChunk1 = playbackRunStarted; const depthAfterChunk1 = prebufferQueue.length;
      await delay(600);
      const startedAfterWait = playbackRunStarted;
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', rt.turnId, rt, 'rest');
      const startedImmediatelyAfterChunk2 = playbackRunStarted;
      await waitForScheduler(() => playbackRunStarted && rt.telemetry.scheduledChunks === 2, 'observability bounded pair release');
      const startedAfterChunk2 = playbackRunStarted; const scheduled = rt.telemetry.scheduledChunks;
      const t = rt.telemetry;

      // GREEN control values (existing accepted no-gap behavior).
      const noGap = {
        startedAfterChunk1, depthAfterChunk1, startedAfterWait,
        startedImmediatelyAfterChunk2, startedAfterChunk2, scheduled,
        decodedChunks: t.decodedChunks,
        underflow: { count: t.underflowGapCount, total: t.underflowGapTotalMs, max: t.underflowGapMaxMs },
      };
      // Additive attribution: did index.html populate per-chunk arrival/decode
      // timestamps, per-chunk decoded audio duration, the first-chunk hold
      // duration, or the hold reason? (All seeded empty above; only index.html
      // production code could fill them.)
      const attribution = {
        arrivalCount: Array.isArray(t.chunkArrivalsMs) ? t.chunkArrivalsMs.length : null,
        decodeCount: Array.isArray(t.chunkDecodeMs) ? t.chunkDecodeMs.length : null,
        decodedDurationCount: Array.isArray(t.chunkDecodedDurationsSec) ? t.chunkDecodedDurationsSec.length : null,
        firstChunkHoldMs: (typeof t.firstChunkHoldMs === 'number') ? t.firstChunkHoldMs : null,
        holdReason: (typeof t.holdReason === 'string') ? t.holdReason : null,
        startupGuardRequestedMs: (typeof t.startupGuardRequestedMs === 'number')
          ? t.startupGuardRequestedMs : null,
      };
      // Additive correlation (R2 P1 finding 1 REPAIR): the KNOWN ids come from the
      // PRODUCER/RELAY FIXTURE (server.py-relayed frames), loaded by Node above and
      // injected here -- the harness does NOT create them. Feed the EXACT fixture
      // frames through the REAL audio_chunk SSE path, then read the ids back off the
      // browser turn telemetry. A correct index.html impl reads payload.turn_id /
      // payload.chunk_id in the audio_chunk handler and surfaces them onto
      // turnState.telemetry (serverTurnId + chunkServerIds); production ignores them,
      // so the telemetry stays empty (RED). We report fixture ids (echoed) AND
      // observed ids; the Python runner asserts telemetry == fixture (never "ids
      // appearing from no input").
      const fixtureProvided = ${corrFixture ? 'true' : 'false'};
      const fixtureTurnId = ${JSON.stringify(corrFixture ? corrFixture.turnId : null)};
      const fixtureChunkIds = ${JSON.stringify(corrFixture ? corrFixture.chunkIds : [])};
      const fixtureSource = ${JSON.stringify(corrFixture ? corrFixture.source : null)};
      const corrFramesFromFixture = ${JSON.stringify(corrFixture ? corrFixture.sseFrames : null)};
      const correlation = {
        driven: false, error: null, sseAudioChunks: null,
        fixtureProvided, fixtureTurnId, fixtureChunkIds, fixtureSource,
        turnServerId: null, chunkServerIds: [], chunkServerIdCount: 0,
      };
      if (!fixtureProvided || corrFramesFromFixture == null) {
        correlation.error = 'no producer/relay fixture provided (MS4_BROWSER_CORR_FIXTURE unset or empty)';
      } else if (typeof submitWavBlobAsVoiceTurn === 'function'
          && typeof window.__ms4OracleVoiceAcceptanceState === 'function') {
        const waitFor = async (fn, label, timeout) => {
          const start = Date.now();
          for (;;) {
            let v; try { v = fn(); } catch (e) { v = null; }
            if (v) return v;
            if (Date.now() - start > (timeout || 8000)) throw new Error('timeout waiting for ' + label);
            await delay(30);
          }
        };
        const encoder = new TextEncoder();
        const makeReader = text => { const bytes = encoder.encode(text); let sent = false; return { read() { if (!sent) { sent = true; return Promise.resolve({ value: bytes, done: false }); } return Promise.resolve({ value: undefined, done: true }); }, cancel() { return Promise.resolve(); } }; };
        // The frames are the EXACT server.py-relayed producer fixture loaded by Node
        // from MS4_BROWSER_CORR_FIXTURE; the harness does not synthesize ids here.
        const corrFrames = corrFramesFromFixture;
        const originalFetch = window.fetch;
        window.fetch = (url, options = {}) => { if (String(url).startsWith('/voice/turn/stream')) return Promise.resolve({ ok: true, statusText: 'OK', body: { getReader: () => makeReader(corrFrames) } }); return originalFetch(url, options); };
        const startId = currentVoiceTurnId;
        try {
          await submitWavBlobAsVoiceTurn(new Blob([new Uint8Array(44)], { type: 'audio/wav' }), { durationSec: 0.1, sourceLabel: 'r4-correlation' });
          // Wait on production-existing telemetry (sseAudioChunks) so this never
          // hangs against production; the ids are read AFTER, not waited on.
          const s = await waitFor(() => {
            const st = window.__ms4OracleVoiceAcceptanceState();
            const lt = st.lastVoiceTurn;
            return (lt && lt.turnId > startId && lt.telemetry && lt.telemetry.sseAudioChunks >= 2) ? st : null;
          }, 'r4-correlation-turn', 9000);
          const lt = s.lastVoiceTurn || {};
          const ltel = lt.telemetry || {};
          const firstArr = (...cands) => { for (const c of cands) { if (Array.isArray(c)) return c; } return []; };
          // Accept a correct impl that surfaces the ids on telemetry OR turn-level.
          const observedTurnServerId = (typeof ltel.serverTurnId === 'string' && ltel.serverTurnId)
            ? ltel.serverTurnId
            : ((typeof lt.serverTurnId === 'string' && lt.serverTurnId) ? lt.serverTurnId : null);
          const observedChunkIds = firstArr(ltel.chunkServerIds, lt.chunkServerIds)
            .map(x => (x == null ? x : String(x)));
          correlation.driven = true;
          correlation.sseAudioChunks = (typeof ltel.sseAudioChunks === 'number') ? ltel.sseAudioChunks : null;
          correlation.turnServerId = observedTurnServerId;
          correlation.chunkServerIds = observedChunkIds;
          correlation.chunkServerIdCount = observedChunkIds.length;
        } catch (e) {
          correlation.error = String((e && e.message) || e);
        }
        window.fetch = originalFetch;
      } else {
        correlation.error = 'submitWavBlobAsVoiceTurn/acceptance API missing';
      }
      // ---- R5 adversarial per-chunk attribution probes (ADDITIVE, non-throwing). --
      // Drive the REAL enqueueAudioChunk with duplicate / decode-failure / cancel-
      // during-decode / late-post-terminal / out-of-order / second-turn scenarios and
      // report the resulting per-chunk telemetry. Turns are seeded with EMPTY chunk
      // records (mkTurn) so any populated record/status can ONLY come from index.html
      // production code. Frozen index.html has no per-chunk record and inflates
      // decoded/duration counts on a duplicate, so these are RED against production;
      // the Python runner owns the assertions.
      const advStartTurnId = currentVoiceTurnId;
      const PROOFA = '${proofAudioBase64}';
      const recCount = ts => Object.keys((ts.telemetry && ts.telemetry.chunkRecords) || {}).length;
      const recStatuses = ts => Object.values((ts.telemetry && ts.telemetry.chunkRecords) || {}).map(r => (r && r.status != null) ? r.status : null);
      const scenario = async fn => { try { return await fn(); } catch (e) { return { error: String((e && e.message) || e) }; } };

      // (1) Duplicate: same producer chunk_id at the same index twice; and separately a
      // same-index (no chunk_id) redelivery. A duplicate must NOT inflate arrival/
      // decode/duration counts and must be counted as a duplicate drop.
      const duplicate = await scenario(async () => {
        reset(); const dc = mkTurn();
        await enqueueAudioChunk(PROOFA, 'audio/wav', dc.turnId, dc, 'rest', 0, 'r5-dup-0');
        await enqueueAudioChunk(PROOFA, 'audio/wav', dc.turnId, dc, 'rest', 0, 'r5-dup-0');
        reset(); const di = mkTurn();
        await enqueueAudioChunk(PROOFA, 'audio/wav', di.turnId, di, 'rest', 0);
        await enqueueAudioChunk(PROOFA, 'audio/wav', di.turnId, di, 'rest', 0);
        return {
          byChunkId: {
            decodedChunks: dc.telemetry.decodedChunks,
            arrivalCount: dc.telemetry.chunkArrivalsMs.length,
            decodeCount: dc.telemetry.chunkDecodeMs.length,
            durationCount: dc.telemetry.chunkDecodedDurationsSec.length,
            duplicateChunks: dc.telemetry.duplicateChunks || 0,
            recordCount: recCount(dc), statuses: recStatuses(dc),
          },
          byIndex: {
            decodedChunks: di.telemetry.decodedChunks,
            decodeCount: di.telemetry.chunkDecodeMs.length,
            durationCount: di.telemetry.chunkDecodedDurationsSec.length,
            duplicateChunks: di.telemetry.duplicateChunks || 0,
            recordCount: recCount(di),
          },
        };
      });

      // (2) Decode failure: decodeAudioData rejects. Arrival counted; decoded/duration
      // NOT counted; a decodeError; record terminal status 'decode_error'.
      const decodeFailure = await scenario(async () => {
        reset(); const df = mkTurn();
        const ctxd = getPlaybackCtx(); const od = ctxd.decodeAudioData.bind(ctxd);
        ctxd.decodeAudioData = () => Promise.reject(new Error('r5 synthetic decode failure'));
        let ret = null;
        try { ret = await enqueueAudioChunk(PROOFA, 'audio/wav', df.turnId, df, 'rest', 0, 'r5-fail-0'); }
        finally { ctxd.decodeAudioData = od; }
        return {
          returned: ret, decodeErrors: df.telemetry.decodeErrors,
          decodedChunks: df.telemetry.decodedChunks,
          arrivalCount: df.telemetry.chunkArrivalsMs.length,
          decodeCount: df.telemetry.chunkDecodeMs.length,
          durationCount: df.telemetry.chunkDecodedDurationsSec.length,
          recordCount: recCount(df), statuses: recStatuses(df),
        };
      });

      // (3) Cancellation during decode: the global turn id advances (barge) WHILE
      // decodeAudioData is in flight. Arrival counted; decoded/duration NOT counted;
      // record terminal status 'cancelled'; no decode inflation; no cross-turn write.
      const cancelMidDecode = await scenario(async () => {
        reset(); const cd = mkTurn(); const saved = currentVoiceTurnId;
        const ctxc = getPlaybackCtx(); const oc = ctxc.decodeAudioData.bind(ctxc);
        ctxc.decodeAudioData = (ab) => { currentVoiceTurnId = saved + 7; return oc(ab); };
        let ret = null;
        try { ret = await enqueueAudioChunk(PROOFA, 'audio/wav', cd.turnId, cd, 'rest', 0, 'r5-canc-0'); }
        finally { ctxc.decodeAudioData = oc; currentVoiceTurnId = saved; }
        return {
          returned: ret, decodedChunks: cd.telemetry.decodedChunks,
          arrivalCount: cd.telemetry.chunkArrivalsMs.length,
          decodeCount: cd.telemetry.chunkDecodeMs.length,
          durationCount: cd.telemetry.chunkDecodedDurationsSec.length,
          recordCount: recCount(cd), statuses: recStatuses(cd),
        };
      });

      // (4) Late post-terminal: the turn is already 'done'; a late chunk must be
      // dropped (return true), NOT counted as an arrival, and recorded terminal 'late'.
      const latePostTerminal = await scenario(async () => {
        reset(); const ltt = mkTurn(); ltt.terminalKind = 'done';
        const ret = await enqueueAudioChunk(PROOFA, 'audio/wav', ltt.turnId, ltt, 'rest', 0, 'r5-late-0');
        return {
          returned: ret, arrivalCount: ltt.telemetry.chunkArrivalsMs.length,
          decodedChunks: ltt.telemetry.decodedChunks,
          decodeCount: ltt.telemetry.chunkDecodeMs.length,
          recordCount: recCount(ltt), statuses: recStatuses(ltt),
        };
      });

      // (5) Out-of-order: index 1 arrives before index 0 (distinct chunk_ids). Both are
      // unique -> both arrive/decode once; contiguous release schedules 2; zero
      // duplicates; two 'decoded' records.
      const outOfOrder = await scenario(async () => {
        reset(); const oo = mkTurn();
        await enqueueAudioChunk(PROOFA, 'audio/wav', oo.turnId, oo, 'rest', 1, 'r5-ooo-1');
        const depth1 = prebufferQueue.length; const started1 = playbackRunStarted;
        await enqueueAudioChunk(PROOFA, 'audio/wav', oo.turnId, oo, 'rest', 0, 'r5-ooo-0');
        await waitForScheduler(() => oo.telemetry.scheduledChunks === 2, 'adversarial out-of-order bounded release');
        return {
          depth1, started1, scheduled: oo.telemetry.scheduledChunks,
          arrivalCount: oo.telemetry.chunkArrivalsMs.length,
          decodeCount: oo.telemetry.chunkDecodeMs.length,
          durationCount: oo.telemetry.chunkDecodedDurationsSec.length,
          decodedChunks: oo.telemetry.decodedChunks,
          duplicateChunks: oo.telemetry.duplicateChunks || 0,
          underflow: oo.telemetry.underflowGapCount,
          recordCount: recCount(oo), statuses: recStatuses(oo),
        };
      });

      // (6) Second turn: a fresh turn after the current turn id advances. Records are
      // per-turn (no cross-turn bleed); a stale chunk for the OLD turn after the new
      // one started must not mutate either turn's accepted accounting.
      const secondTurn = await scenario(async () => {
        reset(); const ta = mkTurn();
        await enqueueAudioChunk(PROOFA, 'audio/wav', ta.turnId, ta, 'rest', 0, 'r5-A-0');
        await delay(40);
        await enqueueAudioChunk(PROOFA, 'audio/wav', ta.turnId, ta, 'rest', 1, 'r5-A-1');
        await waitForScheduler(() => ta.telemetry.scheduledChunks === 2, 'adversarial turn A bounded release');
        const aScheduled = ta.telemetry.scheduledChunks;
        currentVoiceTurnId = currentVoiceTurnId + 1;
        reset(); const tb = mkTurn();
        await enqueueAudioChunk(PROOFA, 'audio/wav', tb.turnId, tb, 'rest', 0, 'r5-B-0');
        await delay(40);
        await enqueueAudioChunk(PROOFA, 'audio/wav', tb.turnId, tb, 'rest', 1, 'r5-B-1');
        await waitForScheduler(() => tb.telemetry.scheduledChunks === 2, 'adversarial turn B bounded release');
        const bScheduled = tb.telemetry.scheduledChunks;
        // Stale chunk for the OLD turn arrives after the new turn started.
        const staleRet = await enqueueAudioChunk(PROOFA, 'audio/wav', ta.turnId, ta, 'rest', 2, 'r5-A-2-stale');
        return {
          aRecordCount: recCount(ta), bRecordCount: recCount(tb),
          aDecoded: ta.telemetry.decodedChunks, bDecoded: tb.telemetry.decodedChunks,
          aDecodeCount: ta.telemetry.chunkDecodeMs.length, bDecodeCount: tb.telemetry.chunkDecodeMs.length,
          aStatuses: recStatuses(ta), bStatuses: recStatuses(tb),
          aScheduled, bScheduled,
          staleReturn: staleRet, aArrivalCount: ta.telemetry.chunkArrivalsMs.length,
        };
      });
      // (7) R6 CAP SATURATION (F1/F2): drive VOICE_CHUNK_RECORDS_MAX + 1 UNIQUE
      // chunks into ONE turn. The records map must hard-cap; the compat arrays
      // (chunkArrivalsMs/chunkDecodeMs/chunkDecodedDurationsSec), decodedChunks and
      // chunkServerIds must honor the SAME bound (F1); a WITHIN-cap redelivery must
      // still dedup; and a BEYOND-cap chunk (dropped at the cap, no record) must NOT
      // be re-accepted/decoded on redelivery (F2). Uses the tiny silent WAV so the
      // 513 real decodes stay fast. Frozen index.html caps only chunkRecords, so the
      // arrays/decodedChunks/chunkServerIds run to 513 and the beyond-cap chunk is
      // re-accepted -- the Python runner owns the RED/GREEN assertions.
      const capSaturation = await scenario(async () => {
        reset(); const ct = mkTurn();
        const CAP = (typeof VOICE_CHUNK_RECORDS_MAX !== 'undefined') ? VOICE_CHUNK_RECORDS_MAX : 512;
        const N = CAP + 1;
        const TINY = '${silentAudioBase64}';
        for (let i = 0; i < N; i++) { await enqueueAudioChunk(TINY, 'audio/wav', ct.turnId, ct, 'rest', i, 'cap-' + i); }
        const tel = ct.telemetry;
        const afterSaturate = {
          cap: CAP, unique_sent: N,
          recordCount: recCount(ct), chunkRecordsDropped: tel.chunkRecordsDropped || 0,
          arrivals: tel.chunkArrivalsMs.length, decodeMs: tel.chunkDecodeMs.length,
          durations: tel.chunkDecodedDurationsSec.length, decodedChunks: tel.decodedChunks,
          serverIds: Array.isArray(tel.chunkServerIds) ? tel.chunkServerIds.length : null,
          duplicateChunks: tel.duplicateChunks || 0,
        };
        // Within-cap redelivery (cap-5) -> must dedup (no re-count).
        const wA = tel.chunkArrivalsMs.length, wD = tel.chunkDecodeMs.length, wDup = tel.duplicateChunks || 0;
        await enqueueAudioChunk(TINY, 'audio/wav', ct.turnId, ct, 'rest', 5, 'cap-5');
        const withinCapRedeliver = {
          arrivalsDelta: tel.chunkArrivalsMs.length - wA,
          decodeDelta: tel.chunkDecodeMs.length - wD,
          dupDelta: (tel.duplicateChunks || 0) - wDup,
        };
        // Beyond-cap redelivery (cap-<CAP> was dropped at the cap; it has NO record).
        const bcName = 'cap-' + CAP;
        const bA = tel.chunkArrivalsMs.length, bD = tel.chunkDecodeMs.length;
        const bDup = tel.duplicateChunks || 0, bDrop = tel.chunkRecordsDropped || 0, bDec = tel.decodedChunks;
        await enqueueAudioChunk(TINY, 'audio/wav', ct.turnId, ct, 'rest', 90000, bcName);
        const beyondCapRedeliver = {
          arrivalsDelta: tel.chunkArrivalsMs.length - bA, decodeDelta: tel.chunkDecodeMs.length - bD,
          dupDelta: (tel.duplicateChunks || 0) - bDup, droppedDelta: (tel.chunkRecordsDropped || 0) - bDrop,
          decodedChunksDelta: tel.decodedChunks - bDec, recordCountAfter: recCount(ct),
        };
        return { afterSaturate, withinCapRedeliver, beyondCapRedeliver };
      });

      currentVoiceTurnId = advStartTurnId;
      const adversarial = { duplicate, decodeFailure, cancelMidDecode, latePostTerminal, outOfOrder, secondTurn, capSaturation };

      reset();
      return { available: true, noGap, attribution, correlation, adversarial };
    })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS });
  } catch (err) {
    // Non-throwing: a probe failure must not fail the shared acceptance harness.
    restObservability = { available: false, error: String((err && err.message) || err) };
  }

  // ---- Streaming scheduler timeline + cancellation contract. ----------------
  // Drive index.html's REAL scheduleDecodedChunk/bargeIn code with a deterministic
  // AudioContext clock. Initial audio retains the 150 ms lead; queued A->B stays
  // contiguous; a late C may add only producer lateness, never another lead.
  schedulerTiming = await cdp.evaluate(`(async () => {
    const saved = {
      playbackCtx, playbackAudioUnlocked, nextChunkStartTime, activeAudioSources,
      prebufferQueue, prebufferTimer, playbackRunStarted, currentVoiceTurnId,
      activeVoiceTurnState, activeChatTurnState, activeVoiceStreamController,
      lastVoiceTurnAcceptanceState, lastVoiceTurnAcceptanceTurnId,
      gate: oracleAwaitSinkReadyBeforeStart,
      elapsed: voiceTurnElapsedMs,
      audioControllers: [...activeOracleAudioFetchControllers],
      oracleMuted: document.documentElement.dataset.oracleMuted,
    };
    let result;
    try {
      // This block verifies the speech scheduler's exact sink-admission counts.
      // Suppress independent optional transition cues so they cannot consume a
      // controlled gate intended exclusively for generated reply chunks.
      document.documentElement.dataset.oracleMuted = 'true';
      // Exact 2026-07-11 physical REST/Vega startup-pair replay. Drive the REAL
      // enqueue -> decode -> REST prebuffer -> schedule -> underflow -> terminal
      // path on the reported turn-relative timeline. Audio decode is synthetic
      // only to pin the three decoded durations; every scheduler decision remains
      // index.html production code running in this real browser.
      const startupReceipt = {
        arrivalsMs: [15265, 15284, 19001],
        decodeReadyMs: [15284, 15288, 19008],
        durationsSec: [1.5786667, 1.8026667, 5.472],
      };
      const startupTurnId = 910001;
      let startupDecodeIndex = 0;
      const startupStarts = [];
      let startupContextNowMs = startupReceipt.arrivalsMs[0];
      let startupContextRun = null;
      const currentStartupContextMs = () => startupContextRun
        ? startupContextRun.baseMs + (performance.now() - startupContextRun.startedAtPerf)
        : startupContextNowMs;
      const waitUntilTurnMs = async (turnStartPerf, targetMs) => {
        while (performance.now() - turnStartPerf < targetMs) {
          const remaining = targetMs - (performance.now() - turnStartPerf);
          await new Promise(resolve => setTimeout(resolve, Math.max(0, Math.min(5, remaining))));
        }
      };
      const startupTurnStartPerf = performance.now() - startupReceipt.arrivalsMs[0];
      const startupCtx = {
        state: 'running',
        destination: {},
        get currentTime() { return currentStartupContextMs() / 1000; },
        async decodeAudioData() {
          const item = startupDecodeIndex++;
          await waitUntilTurnMs(startupTurnStartPerf, startupReceipt.decodeReadyMs[item]);
          startupContextNowMs = startupReceipt.decodeReadyMs[item];
          startupContextRun = null;
          return { label: 'physical-' + item, duration: startupReceipt.durationsSec[item] };
        },
        createBufferSource() {
          return {
            buffer: null, stopped: false, onended: null,
            connect() {},
            start(startAt) {
              startupStarts.push({
                label: this.buffer.label,
                scheduledAtMs: performance.now() - startupTurnStartPerf,
                startAtMs: startAt * 1000,
                durationSec: this.buffer.duration,
              });
            },
            stop() { this.stopped = true; },
            disconnect() {},
          };
        },
      };
      const startupTurn = {
        kind: 'voice', turnId: startupTurnId, controller: null, reader: null,
        assistantState: null, userBubble: null, completed: false, aborted: false,
        audibleFinalized: false, terminalKind: null, ttsEngine: 'rest',
        audioReleaseNextIndex: 0, audioPending: null, idleAfterPlayback: null,
        startedAtEpoch: Date.now(), startedAtPerf: startupTurnStartPerf,
        serverMetrics: {}, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0,
        telemetry: {
          firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null,
          firstScheduledMs: null, firstScheduledContextTime: null,
          firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 0,
          sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0,
          audioErrors: 0, decodeErrors: 0, underflowGapCount: 0,
          underflowGapTotalMs: 0, underflowGapMaxMs: 0,
          chunkArrivalsMs: [], chunkDecodeMs: [], chunkDecodedDurationsSec: [],
          firstChunkHoldMs: null, firstChunkHoldStartMs: null,
          holdReason: null, serverTurnId: null, chunkServerIds: [],
          chunkRecords: {}, duplicateChunks: 0,
        },
      };
      playbackCtx = startupCtx;
      playbackAudioUnlocked = true;
      nextChunkStartTime = 0;
      activeAudioSources = [];
      prebufferQueue = [];
      prebufferTimer = null;
      playbackRunStarted = false;
      currentVoiceTurnId = startupTurnId;
      activeVoiceTurnState = null;
      activeChatTurnState = null;
      activeVoiceStreamController = null;
      activeOracleAudioFetchControllers.clear();
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', startupTurnId, startupTurn, 'rest', 0, 'physical-0');
      const holdReasonAfterFirst = startupTurn.telemetry.holdReason;
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', startupTurnId, startupTurn, 'rest', 1, 'physical-1');
      const pairReadyDecodeMs = startupTurn.telemetry.chunkDecodeMs[1];
      const holdReasonAfterPair = startupTurn.telemetry.holdReason;
      const startupGuardRequestedMs = startupTurn.telemetry.startupGuardRequestedMs ?? null;
      const startedImmediatelyAfterPair = playbackRunStarted;
      if (!startedImmediatelyAfterPair) {
        startupContextRun = {
          baseMs: startupContextNowMs,
          startedAtPerf: performance.now(),
        };
      }
      const pairWaitStarted = performance.now();
      while (!playbackRunStarted && performance.now() - pairWaitStarted < 1000) {
        await new Promise(resolve => setTimeout(resolve, 5));
      }
      if (!playbackRunStarted) throw new Error('startup-pair replay did not release within 1000ms');
      if (startupContextRun) {
        startupContextNowMs = currentStartupContextMs();
        startupContextRun = null;
      }
      await waitUntilTurnMs(startupTurnStartPerf, startupReceipt.arrivalsMs[2]);
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', startupTurnId, startupTurn, 'rest', 2, 'physical-2');
      const startupTelemetry = startupTurn.telemetry;
      const startupPairEndMs = startupStarts[1].startAtMs
        + startupStarts[1].durationSec * 1000;
      const chunk3StartMs = startupStarts[2].startAtMs;
      const startupPair = {
        input: startupReceipt,
        observed: {
          arrivalsMs: [...startupTelemetry.chunkArrivalsMs],
          decodeMs: [...startupTelemetry.chunkDecodeMs],
          durationsSec: [...startupTelemetry.chunkDecodedDurationsSec],
          firstScheduledMs: startupTelemetry.firstScheduledMs,
          firstScheduledContextTimeMs: startupTelemetry.firstScheduledContextTime * 1000,
          firstChunkHoldMs: startupTelemetry.firstChunkHoldMs,
          holdReason: startupTelemetry.holdReason,
          holdReasonAfterFirst,
          holdReasonAfterPair,
          startupGuardRequestedMs,
          scheduledChunks: startupTelemetry.scheduledChunks,
          audioErrors: startupTelemetry.audioErrors,
          decodeErrors: startupTelemetry.decodeErrors,
        },
        startedImmediatelyAfterPair,
        pairReadyDecodeMs,
        guardCapMs: REST_STARTUP_PAIR_HOLD_MAX_MS,
        addedFirstScheduleLatencyMs: startupTelemetry.firstScheduledMs - pairReadyDecodeMs,
        pairEndMs: startupPairEndMs,
        chunk3StartMs,
        pairEndToChunk3StartMs: chunk3StartMs - startupPairEndMs,
        starts: startupStarts,
        underflow: {
          count: startupTelemetry.underflowGapCount,
          totalMs: startupTelemetry.underflowGapTotalMs,
          maxMs: startupTelemetry.underflowGapMaxMs,
        },
        terminalKind: null,
      };
      startupTurn.completed = true;
      startupTurn.terminalKind = 'awaiting_audible';
      startupTurn.telemetry.firstNonSilentOnsetMs = startupTurn.telemetry.firstScheduledMs;
      activeAudioSources = [];
      currentVoiceTurnId = startupTurnId + 1;
      finalizeAudibleVerdict(startupTurn);
      startupPair.terminalKind = startupTurn.terminalKind;

      // Exact post-server-fix physical cadence. Turn time and AudioContext time
      // are independently controlled so the served scheduler sees the measured
      // arrivals/durations while raw continuity remains deterministic.
      const slowReceipt = {
        arrivalsMs: [16844, 20155, 25119, 29397, 29411, 29424],
        decodeReadyMs: [16847, 20175, 25155, 29409, 29424, 29431],
        durationsSec: [
          1.4826666666666666, 2.2186666666666666, 5.0986666666666665,
          8.309333333333333, 10.538666666666666, 5.290666666666667,
        ],
      };
      const slowContextAtDecodeMs = [0, 0, 4890.333333333333, 9234, 9249, 9256];
      const slowTurnId = 915001;
      let slowTurnMs = slowReceipt.arrivalsMs[0];
      let slowContextMs = 0;
      let slowDecodeIndex = 0;
      const slowStarts = [];
      const slowCtx = {
        state: 'running',
        destination: {},
        get currentTime() { return slowContextMs / 1000; },
        decodeAudioData() {
          const index = slowDecodeIndex++;
          slowTurnMs = slowReceipt.decodeReadyMs[index];
          return Promise.resolve({
            label: 'slow-physical-' + index,
            duration: slowReceipt.durationsSec[index],
          });
        },
        createBufferSource() {
          return {
            buffer: null, stopped: false, onended: null,
            connect() {},
            start(startAt) {
              slowStarts.push({
                label: this.buffer.label,
                scheduledAtMs: slowTurnMs,
                startAtMs: startAt * 1000,
                durationSec: this.buffer.duration,
              });
            },
            stop() { this.stopped = true; },
            disconnect() {},
          };
        },
      };
      const slowTurn = {
        kind: 'voice', turnId: slowTurnId, controller: null, reader: null,
        assistantState: null, userBubble: null, completed: false, aborted: false,
        audibleFinalized: false, terminalKind: null, ttsEngine: 'rest',
        audioReleaseNextIndex: 0, audioPending: null, prebufferFlushPromise: null,
        idleAfterPlayback: null, startedAtEpoch: Date.now(), startedAtPerf: 0,
        serverMetrics: {}, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0,
        telemetry: {
          firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null,
          firstScheduledMs: null, firstScheduledContextTime: null,
          firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 6,
          sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0,
          audioErrors: 0, decodeErrors: 0, underflowGapCount: 0,
          underflowGapTotalMs: 0, underflowGapMaxMs: 0,
          chunkArrivalsMs: [], chunkDecodeMs: [], chunkDecodedDurationsSec: [],
          firstChunkHoldMs: null, firstChunkHoldStartMs: null, holdReason: null,
          startupGuardRequestedMs: null, serverTurnId: null, chunkServerIds: [],
          chunkRecords: {}, duplicateChunks: 0,
        },
      };
      voiceTurnElapsedMs = () => slowTurnMs;
      oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();
      playbackCtx = slowCtx;
      playbackAudioUnlocked = true;
      nextChunkStartTime = 0;
      activeAudioSources = [];
      prebufferQueue = [];
      prebufferTimer = null;
      playbackRunStarted = false;
      currentVoiceTurnId = slowTurnId;
      activeVoiceTurnState = null;
      activeChatTurnState = null;
      activeVoiceStreamController = null;
      activeOracleAudioFetchControllers.clear();

      slowTurnMs = slowReceipt.arrivalsMs[0];
      await enqueueAudioChunk(
        '${proofAudioBase64}', 'audio/wav', slowTurnId, slowTurn,
        'rest', 0, 'slow-physical-0');
      slowTurnMs = slowReceipt.arrivalsMs[1];
      await enqueueAudioChunk(
        '${proofAudioBase64}', 'audio/wav', slowTurnId, slowTurn,
        'rest', 1, 'slow-physical-1');
      const slowPairReadyDecodeMs = slowTurn.telemetry.chunkDecodeMs[1];
      const slowReasonAfterPair = slowTurn.telemetry.holdReason;
      const slowRequestedAfterPair = slowTurn.telemetry.startupGuardRequestedMs;
      const slowTimerArmedAfterPair = prebufferTimer != null;
      const slowScheduledAfterPair = slowTurn.telemetry.scheduledChunks;
      let slowFirstScheduledBeforeChunk2 = slowTurn.telemetry.firstScheduledMs;
      if (slowTimerArmedAfterPair) {
        slowTurnMs = 20356;
        slowContextMs = 149;
        const timerStarted = performance.now();
        while (slowTurn.telemetry.scheduledChunks < 2
            && performance.now() - timerStarted < 1000) {
          await new Promise(resolve => setTimeout(resolve, 5));
        }
        if (slowTurn.telemetry.scheduledChunks < 2) {
          throw new Error('slow cadence pair timer did not release within 1000ms');
        }
        slowFirstScheduledBeforeChunk2 = slowTurn.telemetry.firstScheduledMs;
      }
      for (let index = 2; index < slowReceipt.arrivalsMs.length; index += 1) {
        slowTurnMs = slowReceipt.arrivalsMs[index];
        slowContextMs = slowContextAtDecodeMs[index];
        await enqueueAudioChunk(
          '${proofAudioBase64}', 'audio/wav', slowTurnId, slowTurn,
          'rest', index, 'slow-physical-' + index);
      }
      const slowTelemetry = slowTurn.telemetry;
      const slowPairEndMs = slowStarts[1].startAtMs
        + slowStarts[1].durationSec * 1000;
      const slowChunk2StartMs = slowStarts[2].startAtMs;
      const slowCadencePair = {
        input: slowReceipt,
        observed: {
          arrivalsMs: [...slowTelemetry.chunkArrivalsMs],
          decodeMs: [...slowTelemetry.chunkDecodeMs],
          durationsSec: [...slowTelemetry.chunkDecodedDurationsSec],
          firstScheduledMs: slowTelemetry.firstScheduledMs,
          firstChunkHoldMs: slowTelemetry.firstChunkHoldMs,
          holdReason: slowTelemetry.holdReason,
          holdReasonAfterPair: slowReasonAfterPair,
          startupGuardRequestedMs: slowTelemetry.startupGuardRequestedMs,
          startupGuardRequestedAfterPairMs: slowRequestedAfterPair,
          scheduledChunks: slowTelemetry.scheduledChunks,
          audioErrors: slowTelemetry.audioErrors,
          decodeErrors: slowTelemetry.decodeErrors,
        },
        pairReadyDecodeMs: slowPairReadyDecodeMs,
        pairRunwayMs: (
          slowReceipt.durationsSec[0] + slowReceipt.durationsSec[1]) * 1000,
        firstSiblingArrivalGapMs:
          slowReceipt.arrivalsMs[1] - slowReceipt.arrivalsMs[0],
        nextSiblingArrivalGapMs:
          slowReceipt.arrivalsMs[2] - slowReceipt.arrivalsMs[1],
        timerArmedAfterPair: slowTimerArmedAfterPair,
        scheduledAfterPair: slowScheduledAfterPair,
        firstScheduledBeforeChunk2: slowFirstScheduledBeforeChunk2,
        addedFirstScheduleLatencyMs:
          slowTelemetry.firstScheduledMs - slowPairReadyDecodeMs,
        pairEndMs: slowPairEndMs,
        chunk2StartMs: slowChunk2StartMs,
        pairEndToChunk2StartMs: slowChunk2StartMs - slowPairEndMs,
        starts: slowStarts,
        underflow: {
          count: slowTelemetry.underflowGapCount,
          totalMs: slowTelemetry.underflowGapTotalMs,
          maxMs: slowTelemetry.underflowGapMaxMs,
        },
        terminalKind: null,
      };
      slowTurn.completed = true;
      slowTurn.terminalKind = 'awaiting_audible';
      slowTurn.telemetry.firstNonSilentOnsetMs = slowTurn.telemetry.firstScheduledMs;
      activeAudioSources = [];
      currentVoiceTurnId = slowTurnId + 1;
      finalizeAudibleVerdict(slowTurn);
      slowCadencePair.terminalKind = slowTurn.terminalKind;
      voiceTurnElapsedMs = saved.elapsed;

      // Two-sided cadence boundary probes with real outstanding metadata. These
      // stop before any guard timer fires and inspect the served scheduler's
      // admission decision directly.
      let cadenceCaseSeq = 0;
      const runCadenceCase = async spec => {
        cadenceCaseSeq += 1;
        const caseTurnId = 916000 + cadenceCaseSeq;
        let caseTurnMs = 0;
        let nextDecodeIndex = 0;
        let nextDecodeMs = 0;
        let pairCheckpoint = null;
        const caseDurations = new Map(
          spec.events.map(event => [
            event.index,
            event.durationSec != null
              ? event.durationSec : event.frames / event.sampleRate,
          ]));
        const caseStarts = [];
        const caseCtx = {
          state: 'running',
          destination: {},
          currentTime: 0,
          decodeAudioData() {
            const index = nextDecodeIndex;
            caseTurnMs = nextDecodeMs;
            return Promise.resolve({
              label: spec.name + '-' + index,
              duration: caseDurations.get(index),
            });
          },
          createBufferSource() {
            return {
              buffer: null, stopped: false, onended: null,
              connect() {},
              start(startAt) {
                caseStarts.push({
                  label: this.buffer.label,
                  scheduledContextNowMs: caseCtx.currentTime * 1000,
                  startAtMs: startAt * 1000,
                  durationSec: this.buffer.duration,
                });
              },
              stop() { this.stopped = true; },
              disconnect() {},
            };
          },
        };
        const caseTurn = {
          kind: 'voice', turnId: caseTurnId, controller: null, reader: null,
          assistantState: null, userBubble: null, completed: false, aborted: false,
          audibleFinalized: false, terminalKind: null, ttsEngine: 'rest',
          audioReleaseNextIndex: 0, audioPending: null, prebufferFlushPromise: null,
          idleAfterPlayback: null, startedAtEpoch: Date.now(), startedAtPerf: 0,
          serverMetrics: {}, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0,
          telemetry: {
            firstAcknowledgementMs: null, firstSseReceivedMs: null,
            firstDecodedMs: null, firstScheduledMs: null,
            firstScheduledContextTime: null, firstNonSilentOnsetMs: null,
            onsetRms: null, serverScheduledChunks: spec.serverScheduledChunks,
            sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0,
            audioErrors: 0, decodeErrors: 0, underflowGapCount: 0,
            underflowGapTotalMs: 0, underflowGapMaxMs: 0,
            chunkArrivalsMs: [], chunkDecodeMs: [], chunkDecodedDurationsSec: [],
            firstChunkHoldMs: null, firstChunkHoldStartMs: null, holdReason: null,
            startupGuardRequestedMs: null, serverTurnId: null, chunkServerIds: [],
            chunkRecords: {}, duplicateChunks: 0,
          },
        };
        voiceTurnElapsedMs = () => caseTurnMs;
        oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();
        playbackCtx = caseCtx;
        playbackAudioUnlocked = true;
        nextChunkStartTime = 0;
        activeAudioSources = [];
        prebufferQueue = [];
        if (prebufferTimer) { clearTimeout(prebufferTimer); prebufferTimer = null; }
        playbackRunStarted = false;
        currentVoiceTurnId = caseTurnId;
        activeVoiceTurnState = null;
        activeChatTurnState = null;
        activeVoiceStreamController = null;
        activeOracleAudioFetchControllers.clear();
        for (const event of spec.events) {
          if (event.contextDuringWallDelayMs != null) {
            caseCtx.currentTime = event.contextDuringWallDelayMs / 1000;
          }
          if (event.wallDelayBeforeMs) {
            await new Promise(resolve => setTimeout(resolve, event.wallDelayBeforeMs));
          }
          if (event.serverScheduledChunks != null) {
            caseTurn.telemetry.serverScheduledChunks = event.serverScheduledChunks;
          }
          caseTurnMs = event.arrivalMs;
          nextDecodeIndex = event.index;
          nextDecodeMs = event.decodeMs ?? event.arrivalMs;
          caseCtx.currentTime = (event.contextMs ?? 0) / 1000;
          await enqueueAudioChunk(
            '${proofAudioBase64}', 'audio/wav', caseTurnId, caseTurn,
            'rest', event.index, spec.name + '-' + event.index);
          if (spec.captureAfterPair && event.index === 1
              && caseTurn.audioReleaseNextIndex >= 2) {
            pairCheckpoint = {
              timerArmed: prebufferTimer != null,
              playbackRunStarted,
              scheduledChunks: caseTurn.telemetry.scheduledChunks,
              prebufferDepth: prebufferQueue.length,
              serverScheduledChunks: caseTurn.telemetry.serverScheduledChunks,
              decodedChunks: caseTurn.telemetry.decodedChunks,
              holdReason: caseTurn.telemetry.holdReason,
              requestedMs: caseTurn.telemetry.startupGuardRequestedMs,
              firstScheduledMs: caseTurn.telemetry.firstScheduledMs,
              firstChunkHoldMs: caseTurn.telemetry.firstChunkHoldMs,
            };
          }
        }
        const records = Object.values(caseTurn.telemetry.chunkRecords || {});
        const record0 = records.find(record => record.index === 0);
        const record1 = records.find(record => record.index === 1);
        const indexedCadenceMs = record0 && record1
          ? Math.abs(record1.arrivalMs - record0.arrivalMs) : null;
        const arrivals = caseTurn.telemetry.chunkArrivalsMs;
        const acceptedCadenceMs = arrivals.length >= 2
          ? Math.max(0, arrivals[1] - arrivals[0]) : null;
        const pairRunwayMs = (
          caseDurations.get(0) + caseDurations.get(1)) * 1000;
        const sourceGapsMs = caseStarts.slice(1).map((start, index) => {
          const previous = caseStarts[index];
          return start.startAtMs
            - (previous.startAtMs + previous.durationSec * 1000);
        });
        const result = {
          name: spec.name,
          timerArmed: prebufferTimer != null,
          playbackRunStarted,
          scheduledChunks: caseTurn.telemetry.scheduledChunks,
          prebufferDepth: prebufferQueue.length,
          holdReason: caseTurn.telemetry.holdReason,
          requestedMs: caseTurn.telemetry.startupGuardRequestedMs,
          serverScheduledChunks: caseTurn.telemetry.serverScheduledChunks,
          decodedChunks: caseTurn.telemetry.decodedChunks,
          acceptedArrivalsMs: [...arrivals],
          decodeMs: [...caseTurn.telemetry.chunkDecodeMs],
          durationsSec: [...caseTurn.telemetry.chunkDecodedDurationsSec],
          contextAtDecodeMs: spec.events.map(event => event.contextMs ?? 0),
          firstScheduledMs: caseTurn.telemetry.firstScheduledMs,
          firstChunkHoldMs: caseTurn.telemetry.firstChunkHoldMs,
          pairReadyDecodeMs: record1 ? record1.decodeMs : null,
          addedFirstScheduleLatencyMs: (
            record1 && caseTurn.telemetry.firstScheduledMs != null)
            ? caseTurn.telemetry.firstScheduledMs - record1.decodeMs : null,
          acceptedCadenceMs,
          indexedCadenceMs,
          pairRunwayMs,
          availableRunwayMs: pairRunwayMs + AUDIO_LEAD_TIME_MS,
          cadenceTargetMs: indexedCadenceMs == null
            ? null : indexedCadenceMs * REST_SLOW_CADENCE_RUNWAY_FACTOR,
          orderReleasedThrough: caseTurn.audioReleaseNextIndex,
          pairCheckpoint,
          sourceOrder: caseStarts.map(start => start.label),
          sourceGapsMs,
          underflow: {
            count: caseTurn.telemetry.underflowGapCount,
            totalMs: caseTurn.telemetry.underflowGapTotalMs,
            maxMs: caseTurn.telemetry.underflowGapMaxMs,
          },
          audioErrors: caseTurn.telemetry.audioErrors,
          decodeErrors: caseTurn.telemetry.decodeErrors,
          duplicateChunks: caseTurn.telemetry.duplicateChunks,
          terminalKind: null,
          durationFrames: spec.events
            .filter(event => event.index === 0 || event.index === 1)
            .map(event => event.frames ?? null),
          sampleRate: spec.events.find(event => event.sampleRate)?.sampleRate ?? null,
          starts: caseStarts,
        };
        if (spec.complete) {
          caseTurn.completed = true;
          caseTurn.terminalKind = 'awaiting_audible';
          caseTurn.telemetry.firstNonSilentOnsetMs =
            caseTurn.telemetry.firstScheduledMs;
          activeAudioSources = [];
          currentVoiceTurnId = caseTurnId + 1;
          finalizeAudibleVerdict(caseTurn);
          result.terminalKind = caseTurn.terminalKind;
        }
        if (prebufferTimer) { clearTimeout(prebufferTimer); prebufferTimer = null; }
        activeAudioSources = [];
        prebufferQueue = [];
        playbackRunStarted = false;
        return result;
      };
      const cadenceBoundary = {
        meaningfulBelow: await runCadenceCase({
          name: 'meaningful-below',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 0.600},
            {index: 1, arrivalMs: 1000, durationSec: 0.690},
          ],
        }),
        deficit39: await runCadenceCase({
          name: 'deficit-39',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, frames: 28800, sampleRate: 48000},
            {index: 1, arrivalMs: 1000, frames: 34128, sampleRate: 48000},
          ],
        }),
        deficit40: await runCadenceCase({
          name: 'deficit-40',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, frames: 28800, sampleRate: 48000},
            {index: 1, arrivalMs: 1000, frames: 34080, sampleRate: 48000},
          ],
        }),
        deficit41: await runCadenceCase({
          name: 'deficit-41',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, frames: 28800, sampleRate: 48000},
            {index: 1, arrivalMs: 1000, frames: 34032, sampleRate: 48000},
          ],
        }),
        tinyBelow: await runCadenceCase({
          name: 'tiny-below',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 0.650},
            {index: 1, arrivalMs: 1000, durationSec: 0.699},
          ],
        }),
        equality: await runCadenceCase({
          name: 'equality',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 0.650},
            {index: 1, arrivalMs: 1000, durationSec: 0.700},
          ],
        }),
        tinyAbove: await runCadenceCase({
          name: 'tiny-above',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 0.650},
            {index: 1, arrivalMs: 1000, durationSec: 0.701},
          ],
        }),
        nearSimultaneousOutstanding: await runCadenceCase({
          name: 'near-simultaneous-outstanding',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 1.5786667},
            {index: 1, arrivalMs: 19, durationSec: 1.8026667},
          ],
        }),
        adequateOutstanding: await runCadenceCase({
          name: 'adequate-outstanding',
          serverScheduledChunks: 3,
          events: [
            {index: 0, arrivalMs: 0, durationSec: 2.000},
            {index: 1, arrivalMs: 1000, durationSec: 1.850},
          ],
        }),
        indexedInterposed: await runCadenceCase({
          name: 'indexed-interposed',
          serverScheduledChunks: 4,
          events: [
            {index: 1, arrivalMs: 0, durationSec: 0.690},
            {index: 2, arrivalMs: 10, durationSec: 0.001},
            {index: 0, arrivalMs: 1000, durationSec: 0.600},
          ],
        }),
        indexedInterposedCompletion: await runCadenceCase({
          name: 'indexed-interposed-completion',
          serverScheduledChunks: 4,
          complete: true,
          events: [
            {index: 1, arrivalMs: 0, durationSec: 0.690},
            {index: 2, arrivalMs: 10, durationSec: 0.001},
            {index: 0, arrivalMs: 1000, durationSec: 0.600},
            {index: 3, arrivalMs: 1600, durationSec: 3.000},
          ],
        }),
        latestPhysical: await runCadenceCase({
          name: 'latest-physical',
          serverScheduledChunks: 7,
          complete: true,
          captureAfterPair: true,
          events: [
            {
              index: 0, arrivalMs: 15151, decodeMs: 15174,
              durationSec: 1.4293333333333333, contextMs: 0,
            },
            {
              index: 1, arrivalMs: 21314, decodeMs: 21344,
              durationSec: 2.592, contextMs: 0,
            },
            {
              index: 2, arrivalMs: 26935, decodeMs: 26941,
              durationSec: 4.405333333333333, contextMs: 5530.666666666666,
            },
            {
              index: 3, arrivalMs: 26941, decodeMs: 26949,
              durationSec: 5.290666666666667, contextMs: 5538.666666666666,
            },
            {
              index: 4, arrivalMs: 31392, decodeMs: 31400,
              durationSec: 6.912, contextMs: 9989,
            },
            {
              index: 5, arrivalMs: 31400, decodeMs: 31406,
              durationSec: 5.802666666666667, contextMs: 9995,
            },
            {
              index: 6, arrivalMs: 31406, decodeMs: 31409,
              durationSec: 2.8266666666666667, contextMs: 9998,
            },
          ],
        }),
        q20LiveTiming: await runCadenceCase({
          name: 'q20-live-timing',
          serverScheduledChunks: 19,
          complete: true,
          captureAfterPair: true,
          events: [
            {
              index: 0, arrivalMs: 10561, decodeMs: 10565,
              durationSec: 1.856, contextMs: 0, serverScheduledChunks: 3,
            },
            {
              index: 1, arrivalMs: 12063, decodeMs: 12070,
              durationSec: 4.64, contextMs: 0, serverScheduledChunks: 19,
              wallDelayBeforeMs: 1505,
            },
            {
              index: 2, arrivalMs: 19961, decodeMs: 19975,
              durationSec: 9.141333333333334,
              contextMs: 7751.333333333397,
            },
            {
              index: 3, arrivalMs: 19976, decodeMs: 19985,
              durationSec: 6.026666666666666,
              contextMs: 7761.333333333397,
            },
            {
              index: 4, arrivalMs: 20637, decodeMs: 20644,
              durationSec: 4.682666666666667,
              contextMs: 8420.333333333396,
            },
            {
              index: 5, arrivalMs: 25395, decodeMs: 25405,
              durationSec: 7.8933333333333335,
              contextMs: 13181.333333333396,
            },
            {
              index: 6, arrivalMs: 25405, decodeMs: 25413,
              durationSec: 6.261333333333333,
              contextMs: 13189.333333333396,
            },
            {
              index: 7, arrivalMs: 27135, decodeMs: 27142,
              durationSec: 4.224, contextMs: 14918.333333333396,
            },
            {
              index: 8, arrivalMs: 30315, decodeMs: 30324,
              durationSec: 7.616, contextMs: 18100.333333333398,
            },
            {
              index: 9, arrivalMs: 30325, decodeMs: 30328,
              durationSec: 3.2426666666666666,
              contextMs: 18104.333333333398,
            },
            {
              index: 10, arrivalMs: 35473, decodeMs: 35494,
              durationSec: 8.682666666666666,
              contextMs: 23270.3333333334,
            },
            {
              index: 11, arrivalMs: 35496, decodeMs: 35505,
              durationSec: 7.381333333333333,
              contextMs: 23281.3333333334,
            },
            {
              index: 12, arrivalMs: 39774, decodeMs: 39783,
              durationSec: 8.906666666666666,
              contextMs: 27559.3333333334,
            },
            {
              index: 13, arrivalMs: 39783, decodeMs: 39788,
              durationSec: 3.7546666666666666,
              contextMs: 27564.3333333334,
            },
            {
              index: 14, arrivalMs: 41643, decodeMs: 41647,
              durationSec: 3.989333333333333,
              contextMs: 29423.3333333334,
            },
            {
              index: 15, arrivalMs: 42531, decodeMs: 42539,
              durationSec: 6.730666666666667,
              contextMs: 30315.3333333334,
            },
            {
              index: 16, arrivalMs: 47386, decodeMs: 47414,
              durationSec: 9.098666666666666,
              contextMs: 35190.3333333334,
            },
            {
              index: 17, arrivalMs: 47454, decodeMs: 47461,
              durationSec: 7.562666666666667,
              contextMs: 35237.3333333334,
            },
            {
              index: 18, arrivalMs: 49096, decodeMs: 49099,
              durationSec: 1.664, contextMs: 36875.3333333334,
            },
          ],
        }),
        q26LiveTiming: await runCadenceCase({
          name: 'q26-live-timing',
          serverScheduledChunks: 21,
          complete: true,
          captureAfterPair: true,
          events: [
            {index: 0, arrivalMs: 6827, decodeMs: 6831, durationSec: 2.784, contextMs: 0},
            {index: 1, arrivalMs: 9983, decodeMs: 9991, durationSec: 4.405333333333333, contextMs: 0},
            {index: 2, arrivalMs: 13714, decodeMs: 13722, durationSec: 5.8453333333333335, contextMs: 0},
            {
              index: 3, arrivalMs: 21550, decodeMs: 21556,
              durationSec: 3.2426666666666666, contextMs: 7834,
              wallDelayBeforeMs: 650, contextDuringWallDelayMs: 600,
            },
            {index: 4, arrivalMs: 30595, decodeMs: 30605, durationSec: 6.8693333333333335, contextMs: 16883},
            {index: 5, arrivalMs: 33422, decodeMs: 33430, durationSec: 4.266666666666667, contextMs: 19708},
            {index: 6, arrivalMs: 41090, decodeMs: 41101, durationSec: 7.52, contextMs: 27379},
            {index: 7, arrivalMs: 48863, decodeMs: 48871, durationSec: 4.362666666666667, contextMs: 35149},
            {index: 8, arrivalMs: 52592, decodeMs: 52601, durationSec: 6.496, contextMs: 38879},
            {index: 9, arrivalMs: 55316, decodeMs: 55322, durationSec: 4.266666666666667, contextMs: 41600},
            {index: 10, arrivalMs: 59283, decodeMs: 59292, durationSec: 7.189333333333333, contextMs: 45570},
            {index: 11, arrivalMs: 62658, decodeMs: 62665, durationSec: 5.664, contextMs: 48943},
            {index: 12, arrivalMs: 65366, decodeMs: 65372, durationSec: 4.309333333333333, contextMs: 51650},
            {index: 13, arrivalMs: 68544, decodeMs: 68551, durationSec: 5.152, contextMs: 54829},
            {index: 14, arrivalMs: 70974, decodeMs: 70979, durationSec: 3.296, contextMs: 57257},
            {index: 15, arrivalMs: 74700, decodeMs: 74708, durationSec: 6.453333333333333, contextMs: 60986},
            {index: 16, arrivalMs: 77998, decodeMs: 78005, durationSec: 5.056, contextMs: 64283},
            {index: 17, arrivalMs: 82458, decodeMs: 82468, durationSec: 8.170666666666667, contextMs: 68746},
            {index: 18, arrivalMs: 86623, decodeMs: 86631, durationSec: 7.328, contextMs: 72909},
            {index: 19, arrivalMs: 91008, decodeMs: 91017, durationSec: 7.7973333333333334, contextMs: 77295},
            {index: 20, arrivalMs: 93386, decodeMs: 93390, durationSec: 3.477333333333333, contextMs: 79668},
          ],
        }),
      };
      voiceTurnElapsedMs = saved.elapsed;

      // Deterministic review probes for startup boundaries and in-flight flush
      // ownership. These invoke the served page's real enqueue/flush/barge/done
      // paths; only decode buffers, AudioContext time, and sink admission are
      // controlled so concurrency can be observed without copying the scheduler.
      const reviewDelay = ms => new Promise(resolve => setTimeout(resolve, ms));
      const reviewBound = (promise, label, timeoutMs = 3000) => Promise.race([
        promise,
        reviewDelay(timeoutMs).then(() => { throw new Error('scheduler review promise timeout: ' + label); }),
      ]);
      const reviewWait = async (predicate, label, timeoutMs = 1000) => {
        const started = performance.now();
        for (;;) {
          let value = null;
          try { value = predicate(); } catch (_error) { value = null; }
          if (value) return value;
          if (performance.now() - started > timeoutMs) {
            throw new Error('scheduler review timeout: ' + label);
          }
          await reviewDelay(5);
        }
      };
      const settleMicrotasks = async (count = 12) => {
        for (let i = 0; i < count; i += 1) await Promise.resolve();
      };
      const makeReviewTurn = turnId => ({
        kind: 'voice', turnId, controller: null, reader: null, assistantState: null,
        userBubble: null, completed: false, aborted: false, audibleFinalized: false,
        terminalKind: null, ttsEngine: 'rest', audioReleaseNextIndex: 0,
        audioPending: null, idleAfterPlayback: null, startedAtEpoch: Date.now(),
        startedAtPerf: performance.now(), serverMetrics: {}, onsetAnalyser: null,
        onsetData: null, onsetMonitorFrame: 0, prebufferFlushPromise: null,
        telemetry: {
          firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null,
          firstScheduledMs: null, firstScheduledContextTime: null,
          firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 0,
          sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0,
          audioErrors: 0, decodeErrors: 0, underflowGapCount: 0,
          underflowGapTotalMs: 0, underflowGapMaxMs: 0,
          chunkArrivalsMs: [], chunkDecodeMs: [], chunkDecodedDurationsSec: [],
          firstChunkHoldMs: null, firstChunkHoldStartMs: null, holdReason: null,
          startupGuardRequestedMs: null, serverTurnId: null, chunkServerIds: [],
          chunkRecords: {}, duplicateChunks: 0,
        },
      });
      const makeReviewContext = durations => {
        let decodeIndex = 0;
        const starts = [];
        const sources = [];
        const ctx = {
          state: 'running', destination: {}, sampleRate: 24000,
          get currentTime() { return 30; },
          decodeAudioData() {
            const index = decodeIndex++;
            return Promise.resolve({label: 'review-' + index, duration: durations[index]});
          },
          createBufferSource() {
            const source = {
              buffer: null, stopped: false, onended: null,
              connect() {},
              start(startAt) {
                starts.push({label: this.buffer.label, startAt, duration: this.buffer.duration});
              },
              stop() { this.stopped = true; },
              disconnect() {},
            };
            sources.push(source);
            return source;
          },
        };
        return {ctx, starts, sources};
      };
      const resetReview = (ctx, turnId) => {
        if (prebufferTimer) { clearTimeout(prebufferTimer); prebufferTimer = null; }
        playbackCtx = ctx;
        playbackAudioUnlocked = true;
        nextChunkStartTime = 0;
        activeAudioSources = [];
        prebufferQueue = [];
        playbackRunStarted = false;
        currentVoiceTurnId = turnId;
        activeVoiceTurnState = null;
        activeChatTurnState = null;
        activeVoiceStreamController = null;
        activeOracleAudioFetchControllers.clear();
      };
      const installDeferredSink = () => {
        const releases = [];
        oracleAwaitSinkReadyBeforeStart = () => new Promise(resolve => { releases.push(resolve); });
        return releases;
      };

      oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();
      let reviewMedia = makeReviewContext([4.0]);
      resetReview(reviewMedia.ctx, 920001);
      let reviewTurn = makeReviewTurn(currentVoiceTurnId);
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'boundary-0');
      const loneBoundary = {
        started: playbackRunStarted,
        scheduled: reviewTurn.telemetry.scheduledChunks,
        timerArmed: prebufferTimer != null,
      };

      // Physical35 decoded one 3749.125 ms REST chunk, then received neither a
      // sibling nor done. Capture the lone-first deadline without sleeping, fire
      // it deterministically, and separately prove barge-in revokes its owner.
      const loneNativeSetTimeout = window.setTimeout;
      const loneNativeClearTimeout = window.clearTimeout;
      let loneEscapeCallback = null;
      let loneEscapeDelayMs = null;
      let loneEscapeTimerCleared = false;
      let loneEscapeMedia;
      let loneEscapeTurn;
      let loneEscapeBeforeDeadline;
      let loneEscapeAfterDeadline;
      let loneCancelCallback = null;
      let loneCancelDelayMs = null;
      let loneCancelTimerCleared = false;
      let loneCancelMedia;
      let loneCancelTurn;
      let loneCancelBeforeBarge;
      let loneCancelAfterBarge;
      try {
        const escapeToken = {restLoneFirstEscapeTimer: true};
        window.setTimeout = (callback, delayMs) => {
          if (!loneEscapeCallback) {
            loneEscapeCallback = callback;
            loneEscapeDelayMs = delayMs;
          }
          return escapeToken;
        };
        window.clearTimeout = token => {
          if (token === escapeToken) loneEscapeTimerCleared = true;
        };
        loneEscapeMedia = makeReviewContext([3.749125]);
        resetReview(loneEscapeMedia.ctx, 920009);
        loneEscapeTurn = makeReviewTurn(currentVoiceTurnId);
        loneEscapeTurn.telemetry.serverScheduledChunks = 1;
        await enqueueAudioChunk(
          '${proofAudioBase64}', 'audio/wav', loneEscapeTurn.turnId,
          loneEscapeTurn, 'rest', 0, 'rest-lone-3749125-escape');
        loneEscapeBeforeDeadline = {
          started: playbackRunStarted,
          scheduled: loneEscapeTurn.telemetry.scheduledChunks,
          queueDepth: prebufferQueue.length,
          holdReason: loneEscapeTurn.telemetry.holdReason,
          timerArmed: prebufferTimer != null,
          callbackCaptured: typeof loneEscapeCallback === 'function',
          requestedMs: loneEscapeDelayMs,
        };
        if (loneEscapeCallback) {
          loneEscapeCallback();
          await settleMicrotasks(24);
        }
        loneEscapeAfterDeadline = {
          started: playbackRunStarted,
          scheduled: loneEscapeTurn.telemetry.scheduledChunks,
          starts: loneEscapeMedia.starts.length,
          queueDepth: prebufferQueue.length,
          timerCleared: prebufferTimer == null,
          serverScheduledChunks: loneEscapeTurn.telemetry.serverScheduledChunks,
          decodedChunks: loneEscapeTurn.telemetry.decodedChunks,
        };

        const cancelToken = {restLoneFirstCancelTimer: true};
        window.setTimeout = (callback, delayMs) => {
          if (!loneCancelCallback) {
            loneCancelCallback = callback;
            loneCancelDelayMs = delayMs;
          }
          return cancelToken;
        };
        window.clearTimeout = token => {
          if (token === cancelToken) loneCancelTimerCleared = true;
        };
        loneCancelMedia = makeReviewContext([3.749125]);
        resetReview(loneCancelMedia.ctx, 920010);
        loneCancelTurn = makeReviewTurn(currentVoiceTurnId);
        await enqueueAudioChunk(
          '${proofAudioBase64}', 'audio/wav', loneCancelTurn.turnId,
          loneCancelTurn, 'rest', 0, 'rest-lone-3749125-cancel');
        loneCancelBeforeBarge = {
          scheduled: loneCancelTurn.telemetry.scheduledChunks,
          queueDepth: prebufferQueue.length,
          timerArmed: prebufferTimer != null,
          callbackCaptured: typeof loneCancelCallback === 'function',
          requestedMs: loneCancelDelayMs,
        };
        const loneCancelOwner = loneCancelTurn.turnId;
        bargeIn(null, {halfContext: false, playAck: false});
        if (loneCancelCallback) loneCancelCallback();
        await settleMicrotasks();
        loneCancelAfterBarge = {
          ownerAdvanced: currentVoiceTurnId > loneCancelOwner,
          scheduled: loneCancelTurn.telemetry.scheduledChunks,
          starts: loneCancelMedia.starts.length,
          queueDepth: prebufferQueue.length,
          timerCleared: prebufferTimer == null && loneCancelTimerCleared,
          playbackRunStarted,
        };
      } finally {
        window.setTimeout = loneNativeSetTimeout;
        window.clearTimeout = loneNativeClearTimeout;
      }
      const restLoneFirstEscape = {
        durationMs: 3749.125,
        beforeDeadline: loneEscapeBeforeDeadline,
        afterDeadline: loneEscapeAfterDeadline,
        cancelBeforeBarge: loneCancelBeforeBarge,
        cancelAfterBarge: loneCancelAfterBarge,
      };

      // Exact R9 startup failure replay. The old lone-first timeout released a
      // 3.658667 s chunk at 1 s (+150 ms lead); its already-declared sibling was
      // not decoded until 4.989667 s, creating the observed 181 ms silence gap.
      // The production scheduler must now keep that first chunk held at timeout,
      // then release both contiguously when the sibling arrives.
      const r9NativeSetTimeout = window.setTimeout;
      const r9NativeClearTimeout = window.clearTimeout;
      const r9FirstDurationSec = 3.6586666666666665;
      const r9SecondDurationSec = 6.677333333333333;
      const r9LegacyTimerMs = REST_LONE_FIRST_MAX_MS;
      const r9SiblingDecodeMs = 4989.666666666667;
      const r9LegacyFirstStartMs = r9LegacyTimerMs + AUDIO_LEAD_TIME_MS;
      const r9LegacyFirstEndMs = r9LegacyFirstStartMs + r9FirstDurationSec * 1000;
      let r9ContextNowSec = 0;
      let r9DecodeIndex = 0;
      let r9TimerCallback = null;
      let r9TimerDelayMs = null;
      let r9TimerCleared = false;
      const r9Starts = [];
      const r9Sources = [];
      const r9TimerToken = {r9OutstandingSiblingTimer: true};
      const r9Ctx = {
        state: 'running', destination: {}, sampleRate: 24000,
        get currentTime() { return r9ContextNowSec; },
        decodeAudioData() {
          const durations = [r9FirstDurationSec, r9SecondDurationSec];
          const index = r9DecodeIndex++;
          return Promise.resolve({label: 'r9-' + index, duration: durations[index]});
        },
        createBufferSource() {
          const source = {
            buffer: null, stopped: false, onended: null,
            connect() {},
            start(startAt) {
              r9Starts.push({label: this.buffer.label, startAt, duration: this.buffer.duration});
            },
            stop() { this.stopped = true; },
            disconnect() {},
          };
          r9Sources.push(source);
          return source;
        },
      };
      let r9Turn;
      let r9BeforeDeadline;
      let r9AfterDeadline;
      let r9AfterSibling;
      try {
        window.setTimeout = (callback, delayMs) => {
          r9TimerCallback = callback;
          r9TimerDelayMs = delayMs;
          return r9TimerToken;
        };
        window.clearTimeout = token => {
          if (token === r9TimerToken) r9TimerCleared = true;
        };
        resetReview(r9Ctx, 920011);
        r9Turn = makeReviewTurn(currentVoiceTurnId);
        r9Turn.telemetry.serverScheduledChunks = 2;
        await enqueueAudioChunk(
          '${proofAudioBase64}', 'audio/wav', r9Turn.turnId,
          r9Turn, 'rest', 0, 'r9-outstanding-0');
        r9BeforeDeadline = {
          scheduled: r9Turn.telemetry.scheduledChunks,
          decoded: r9Turn.telemetry.decodedChunks,
          serverScheduled: r9Turn.telemetry.serverScheduledChunks,
          queueDepth: prebufferQueue.length,
          timerArmed: prebufferTimer != null,
          requestedMs: r9TimerDelayMs,
        };
        r9ContextNowSec = r9LegacyTimerMs / 1000;
        if (r9TimerCallback) {
          r9TimerCallback();
          await settleMicrotasks(24);
        }
        r9AfterDeadline = {
          scheduled: r9Turn.telemetry.scheduledChunks,
          decoded: r9Turn.telemetry.decodedChunks,
          serverScheduled: r9Turn.telemetry.serverScheduledChunks,
          queueDepth: prebufferQueue.length,
          timerArmed: prebufferTimer != null,
          timerCleared: r9TimerCleared,
          playbackRunStarted,
          holdReason: r9Turn.telemetry.holdReason,
          starts: r9Starts.length,
        };
        r9ContextNowSec = r9SiblingDecodeMs / 1000;
        await enqueueAudioChunk(
          '${proofAudioBase64}', 'audio/wav', r9Turn.turnId,
          r9Turn, 'rest', 1, 'r9-outstanding-1');
        const r9RawGapMs = r9Starts.length === 2
          ? (r9Starts[1].startAt - (r9Starts[0].startAt + r9Starts[0].duration)) * 1000
          : null;
        r9AfterSibling = {
          scheduled: r9Turn.telemetry.scheduledChunks,
          decoded: r9Turn.telemetry.decodedChunks,
          serverScheduled: r9Turn.telemetry.serverScheduledChunks,
          queueDepth: prebufferQueue.length,
          playbackRunStarted,
          starts: r9Starts.map(item => ({...item})),
          rawGapMs: r9RawGapMs,
          underflow: {
            count: r9Turn.telemetry.underflowGapCount,
            totalMs: r9Turn.telemetry.underflowGapTotalMs,
            maxMs: r9Turn.telemetry.underflowGapMaxMs,
          },
        };
      } finally {
        window.setTimeout = r9NativeSetTimeout;
        window.clearTimeout = r9NativeClearTimeout;
      }
      const r9OutstandingSibling = {
        input: {
          firstDurationSec: r9FirstDurationSec,
          secondDurationSec: r9SecondDurationSec,
          siblingDecodeMs: r9SiblingDecodeMs,
          legacyTimerMs: r9LegacyTimerMs,
          leadMs: AUDIO_LEAD_TIME_MS,
        },
        legacy: {
          firstStartMs: r9LegacyFirstStartMs,
          firstEndMs: r9LegacyFirstEndMs,
          gapMs: r9SiblingDecodeMs - r9LegacyFirstEndMs,
          wouldCountAtThreshold: r9SiblingDecodeMs - r9LegacyFirstEndMs
            >= UNDERFLOW_GAP_MIN_MS,
          thresholdMs: UNDERFLOW_GAP_MIN_MS,
        },
        beforeDeadline: r9BeforeDeadline,
        afterDeadline: r9AfterDeadline,
        afterSibling: r9AfterSibling,
      };

      reviewMedia = makeReviewContext([2.0, 1.85]);
      resetReview(reviewMedia.ctx, 920002);
      reviewTurn = makeReviewTurn(currentVoiceTurnId);
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'adequate-0');
      const adequateHeldAfterFirst = !playbackRunStarted;
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 1, 'adequate-1');
      const adequatePair = {
        heldAfterFirst: adequateHeldAfterFirst,
        startedAfterPair: playbackRunStarted,
        scheduled: reviewTurn.telemetry.scheduledChunks,
        timerArmed: prebufferTimer != null,
        order: reviewMedia.starts.map(item => item.label),
      };

      reviewMedia = makeReviewContext([1.0, 1.0, 2.0]);
      resetReview(reviewMedia.ctx, 920003);
      reviewTurn = makeReviewTurn(currentVoiceTurnId);
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'early-0');
      const earlyReasonAfterFirst = reviewTurn.telemetry.holdReason;
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 1, 'early-1');
      const earlyAfterPair = {
        started: playbackRunStarted,
        timerArmed: prebufferTimer != null,
        reason: reviewTurn.telemetry.holdReason,
        requestedMs: reviewTurn.telemetry.startupGuardRequestedMs ?? null,
      };
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 2, 'early-2');
      const earlyThird = {
        reasonAfterFirst: earlyReasonAfterFirst,
        afterPair: earlyAfterPair,
        startedAfterThird: playbackRunStarted,
        scheduled: reviewTurn.telemetry.scheduledChunks,
        timerCleared: prebufferTimer == null,
        order: reviewMedia.starts.map(item => item.label),
        underflow: reviewTurn.telemetry.underflowGapCount,
      };

      // Capture a real guard timer callback, barge the owner, then invoke the
      // already-queued callback. A stale callback must not revive run state.
      const nativeSetTimeout = window.setTimeout;
      const nativeClearTimeout = window.clearTimeout;
      let staleTimerCallback = null;
      let staleTimerDelayMs = null;
      let staleTimerCleared = false;
      try {
        window.setTimeout = (callback, delayMs) => {
          staleTimerCallback = callback;
          staleTimerDelayMs = delayMs;
          return {schedulerReviewTimer: true};
        };
        window.clearTimeout = () => { staleTimerCleared = true; };
        reviewMedia = makeReviewContext([1.0, 1.0]);
        resetReview(reviewMedia.ctx, 920004);
        reviewTurn = makeReviewTurn(currentVoiceTurnId);
        await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'stale-timer-0');
        await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 1, 'stale-timer-1');
        bargeIn(null, {halfContext: false, playAck: false});
        if (staleTimerCallback) staleTimerCallback();
        await settleMicrotasks();
      } finally {
        window.setTimeout = nativeSetTimeout;
        window.clearTimeout = nativeClearTimeout;
      }
      const staleTimer = {
        callbackCaptured: typeof staleTimerCallback === 'function',
        requestedMs: staleTimerDelayMs,
        clearedByBarge: staleTimerCleared,
        playbackRunStarted,
        scheduled: reviewTurn.telemetry.scheduledChunks,
        starts: reviewMedia.starts.length,
        queueDepth: prebufferQueue.length,
      };

      // Defer sink admission while a pair flushes, then decode chunk 3. Current
      // code exposes a second gate before the pair drain completes and permits
      // chunk 3 to schedule first; serialized code must expose gates in 0,1,2 order.
      reviewMedia = makeReviewContext([1.0, 1.0, 1.0]);
      resetReview(reviewMedia.ctx, 920005);
      reviewTurn = makeReviewTurn(currentVoiceTurnId);
      let reviewGates = installDeferredSink();
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'flush-0');
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 1, 'flush-1');
      const activeFlushPromise = flushPrebuffer();
      await reviewWait(() => reviewGates.length >= 1, 'active flush first sink gate');
      const lateChunkPromise = enqueueAudioChunk(
        '${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 2, 'flush-2');
      await settleMicrotasks();
      const concurrentGateBeforeFlushRelease = reviewGates.length >= 2;
      if (concurrentGateBeforeFlushRelease) {
        reviewGates[1]();
        await reviewBound(lateChunkPromise, 'concurrent late chunk');
        reviewGates[0]();
        await reviewWait(() => reviewGates.length >= 3, 'active flush second pair gate');
        reviewGates[2]();
        await reviewBound(activeFlushPromise, 'concurrent active flush');
      } else {
        reviewGates[0]();
        await reviewWait(() => reviewGates.length >= 2, 'serialized pair second gate');
        reviewGates[1]();
        await reviewBound(activeFlushPromise, 'serialized active flush');
        await reviewWait(() => reviewGates.length >= 3, 'serialized late chunk gate');
        reviewGates[2]();
        await reviewBound(lateChunkPromise, 'serialized late chunk');
      }
      const activeFlush = {
        concurrentGateBeforeFlushRelease,
        order: reviewMedia.starts.map(item => item.label),
        scheduled: reviewTurn.telemetry.scheduledChunks,
      };

      // Barge while the production flush and a later chunk are both sink-blocked.
      // Releasing those gates afterwards must schedule nothing for the stale turn.
      reviewMedia = makeReviewContext([1.0, 1.0, 1.0]);
      resetReview(reviewMedia.ctx, 920006);
      reviewTurn = makeReviewTurn(currentVoiceTurnId);
      reviewGates = installDeferredSink();
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 0, 'cancel-0');
      await enqueueAudioChunk('${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 1, 'cancel-1');
      const cancelFlushPromise = flushPrebuffer();
      await reviewWait(() => reviewGates.length >= 1, 'cancel active flush first gate');
      const cancelLatePromise = enqueueAudioChunk(
        '${proofAudioBase64}', 'audio/wav', reviewTurn.turnId, reviewTurn, 'rest', 2, 'cancel-2');
      await settleMicrotasks();
      const cancelOwner = reviewTurn.turnId;
      bargeIn(null, {halfContext: false, playAck: false});
      for (const release of reviewGates) release();
      await reviewBound(
        Promise.allSettled([cancelFlushPromise, cancelLatePromise]),
        'active flush cancellation');
      await settleMicrotasks();
      const activeFlushCancellation = {
        ownerAdvanced: currentVoiceTurnId > cancelOwner,
        scheduled: reviewTurn.telemetry.scheduledChunks,
        starts: reviewMedia.starts.length,
        queueDepth: prebufferQueue.length,
        timerCleared: prebufferTimer == null,
        playbackRunStarted,
        flushPromiseCleared: !reviewTurn.prebufferFlushPromise,
      };

      // Drive the actual submit/SSE done branch while an explicitly-started
      // production flush is sink-blocked. No wall-clock guard timer is involved.
      const originalFetchForDone = window.fetch;
      let doneTurn = null;
      let doneSettled = false;
      let doneError = null;
      let doneReads = [];
      let signalDoneReadReady;
      const doneReadReady = new Promise(resolve => { signalDoneReadReady = resolve; });
      try {
        reviewMedia = makeReviewContext([1.0, 1.0]);
        resetReview(reviewMedia.ctx, 920007);
        reviewGates = installDeferredSink();
        const encoder = new TextEncoder();
        const pairFrames =
          'event: transcript\\ndata: {"text":"serialization done probe","asr_ms":1}\\n\\n'
          + 'event: text_delta\\ndata: {"text":"Done serialization reply."}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":0,"chunk_id":"done-0","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":1,"chunk_id":"done-1","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n';
        const doneFrame =
          'event: done\\ndata: {"session_id":"serialization-done","reply_text":"Done serialization reply.","metrics":{"audio_chunks":2,"audio_client_written":2,"audio_errors":0,"total_ms":5}}\\n\\n';
        let firstRead = true;
        const reader = {
          read() {
            if (firstRead) {
              firstRead = false;
              return Promise.resolve({value: encoder.encode(pairFrames), done: false});
            }
            return new Promise(resolve => {
              doneReads.push(resolve);
              signalDoneReadReady();
            });
          },
          cancel() { return Promise.resolve(); },
        };
        window.fetch = (url, options = {}) => {
          if (String(url).startsWith('/voice/turn/stream')) {
            return Promise.resolve({ok: true, statusText: 'OK', body: {getReader: () => reader}});
          }
          return originalFetchForDone(url, options);
        };
        const doneSubmitPromise = submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'serialization-done'});
        doneSubmitPromise.then(
          () => { doneSettled = true; },
          error => { doneError = String((error && error.message) || error); doneSettled = true; },
        );
        await doneReadReady;
        doneTurn = activeVoiceTurnState;
        if (!doneTurn || doneTurn.telemetry.decodedChunks !== 2) {
          throw new Error('real done pair did not complete controlled decode');
        }
        const preFlushState = {
          timerArmed: prebufferTimer != null,
          playbackRunStarted,
          prebufferDepth: prebufferQueue.length,
          flushPromisePresent: doneTurn.prebufferFlushPromise != null,
          scheduledChunks: doneTurn.telemetry.scheduledChunks,
        };
        const explicitFlushPromise =
          flushPrebuffer(doneTurn.turnId, doneTurn);
        const activeFlushState = {
          gateCount: reviewGates.length,
          playbackRunStarted,
          prebufferDepth: prebufferQueue.length,
          flushPromisePresent: doneTurn.prebufferFlushPromise != null,
          scheduledChunks: doneTurn.telemetry.scheduledChunks,
        };
        if (reviewGates.length !== 1
            || activeFlushState.playbackRunStarted !== true
            || activeFlushState.prebufferDepth !== 0
            || activeFlushState.flushPromisePresent !== true
            || activeFlushState.scheduledChunks !== 0
            || doneReads.length !== 1) {
          throw new Error(
            'real done explicit flush invariant failed: '
            + JSON.stringify({activeFlushState, gateCount: reviewGates.length,
              doneReads: doneReads.length}));
        }
        doneReads.shift()({value: encoder.encode(doneFrame), done: false});
        await settleMicrotasks();
        const doneAwaitingFlushBeforeRelease = (
          doneTurn.completed === true
          && doneSettled === false
          && doneTurn.prebufferFlushPromise != null
        );
        if (!doneAwaitingFlushBeforeRelease) {
          throw new Error('real done did not await the controlled active flush');
        }
        const settledBeforeSinkRelease = doneSettled;
        const terminalBeforeSinkRelease = doneTurn.terminalKind;
        const scheduledBeforeSinkRelease = doneTurn.telemetry.scheduledChunks;
        reviewGates[0]();
        await settleMicrotasks();
        if (reviewGates.length !== 2) {
          throw new Error(
            'real done second controlled gate missing: '
            + JSON.stringify({gateCount: reviewGates.length,
              scheduled: doneTurn.telemetry.scheduledChunks}));
        }
        reviewGates[1]();
        await reviewBound(
          Promise.all([explicitFlushPromise, doneSubmitPromise]),
          'real done explicit flush');
        await settleMicrotasks();
        var realDone = {
          flushStartMode: 'explicit_production_flush',
          preFlushState,
          activeFlushState,
          doneAwaitingFlushBeforeRelease,
          settledBeforeSinkRelease,
          terminalBeforeSinkRelease,
          scheduledBeforeSinkRelease,
          finalTerminal: doneTurn.terminalKind,
          finalScheduled: doneTurn.telemetry.scheduledChunks,
          error: doneError,
          order: reviewMedia.starts.map(item => item.label),
        };
      } finally {
        window.fetch = originalFetchForDone;
        if (activeAudioSources.length) bargeIn(null, {halfContext: false, playAck: false});
      }

      // A terminal done event is the second release edge for an outstanding REST
      // startup pair. Drive the real submit/SSE branch without an explicit test
      // flush: two decoded chunks remain buffered while three are declared, then
      // done must drain both and let submit settle instead of hanging forever.
      const originalFetchForPairDone = window.fetch;
      let pairDoneTurn = null;
      let pairDoneError = null;
      let pairDoneReads = [];
      let signalPairDoneReadReady;
      const pairDoneReadReady =
        new Promise(resolve => { signalPairDoneReadReady = resolve; });
      try {
        reviewMedia = makeReviewContext([1.0, 1.0]);
        resetReview(reviewMedia.ctx, 920009);
        oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();
        const encoder = new TextEncoder();
        const pairFrames =
          'event: transcript\\ndata: {"text":"outstanding pair done probe","asr_ms":1}\\n\\n'
          + 'event: text_delta\\ndata: {"text":"Outstanding pair done reply."}\\n\\n'
          + 'event: chunk_scheduled\\ndata: {"index":0}\\n\\n'
          + 'event: chunk_scheduled\\ndata: {"index":1}\\n\\n'
          + 'event: chunk_scheduled\\ndata: {"index":2}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":0,"chunk_id":"pair-done-0","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":1,"chunk_id":"pair-done-1","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n';
        const doneFrame =
          'event: done\\ndata: {"session_id":"outstanding-pair-done","reply_text":"Outstanding pair done reply.","metrics":{"audio_chunks":2,"audio_client_written":2,"audio_errors":0,"total_ms":5}}\\n\\n';
        let readPhase = 0;
        const reader = {
          read() {
            if (readPhase === 0) {
              readPhase = 1;
              return Promise.resolve({value: encoder.encode(pairFrames), done: false});
            }
            if (readPhase === 1) {
              readPhase = 2;
              return new Promise(resolve => {
                pairDoneReads.push(resolve);
                signalPairDoneReadReady();
              });
            }
            return Promise.resolve({value: undefined, done: true});
          },
          cancel() { return Promise.resolve(); },
        };
        window.fetch = (url, options = {}) => {
          if (String(url).startsWith('/voice/turn/stream')) {
            return Promise.resolve({ok: true, statusText: 'OK', body: {getReader: () => reader}});
          }
          return originalFetchForPairDone(url, options);
        };
        const pairDoneSubmit = submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'outstanding-pair-done'});
        pairDoneSubmit.catch(error => {
          pairDoneError = String((error && error.message) || error);
        });
        await pairDoneReadReady;
        pairDoneTurn = activeVoiceTurnState;
        if (!pairDoneTurn || pairDoneTurn.telemetry.decodedChunks !== 2) {
          throw new Error('outstanding pair done did not complete controlled decode');
        }
        const heldBeforeDone = {
          serverScheduledChunks: pairDoneTurn.telemetry.serverScheduledChunks,
          decodedChunks: pairDoneTurn.telemetry.decodedChunks,
          scheduledChunks: pairDoneTurn.telemetry.scheduledChunks,
          prebufferDepth: prebufferQueue.length,
          playbackRunStarted,
          timerArmed: prebufferTimer != null,
          holdReason: pairDoneTurn.telemetry.holdReason,
        };
        pairDoneReads.shift()({value: encoder.encode(doneFrame), done: false});
        await reviewBound(pairDoneSubmit, 'outstanding pair terminal done');
        await settleMicrotasks();
        var outstandingPairDone = {
          heldBeforeDone,
          completed: pairDoneTurn.completed,
          finalTerminal: pairDoneTurn.terminalKind,
          finalScheduled: pairDoneTurn.telemetry.scheduledChunks,
          finalPrebufferDepth: prebufferQueue.length,
          playbackRunStarted,
          submitError: pairDoneError,
          order: reviewMedia.starts.map(item => item.label),
        };
      } finally {
        window.fetch = originalFetchForPairDone;
        if (activeAudioSources.length) bargeIn(null, {halfContext: false, playAck: false});
      }

      // Combined ownership race: done has marked the old turn completed and is
      // awaiting the sink-blocked pair drain when a real barge establishes a
      // replacement owner. Releasing the sink must not let old done mutate it.
      const originalFetchForDoneBarge = window.fetch;
      const priorDoneBargeUi = {
        presence: oracleStage.dataset.presence,
        subtitle: oracleSubtitle.textContent,
        detail: oracleTimingLine.textContent,
        status: voiceStatusLine.textContent,
        sessionId,
        storedSessionId: localStorage.getItem('ms4_session_id'),
        sessionText: sessionEl.textContent,
      };
      let doneBargeTurn = null;
      let doneBargeSettled = false;
      let doneBargeError = null;
      let doneBargeReads = [];
      let signalDoneBargeReadReady;
      const doneBargeReadReady =
        new Promise(resolve => { signalDoneBargeReadReady = resolve; });
      let replacementTurn = null;
      try {
        reviewMedia = makeReviewContext([1.0, 1.0]);
        resetReview(reviewMedia.ctx, 920008);
        reviewGates = installDeferredSink();
        const encoder = new TextEncoder();
        const pairFrames =
          'event: transcript\\ndata: {"text":"combined done barge probe","asr_ms":1}\\n\\n'
          + 'event: text_delta\\ndata: {"text":"Old done reply."}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":0,"chunk_id":"done-barge-0","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n'
          + 'event: audio_chunk\\ndata: {"index":1,"chunk_id":"done-barge-1","audio_base64":"${proofAudioBase64}","audio_mime":"audio/wav"}\\n\\n';
        const doneFrame =
          'event: done\\ndata: {"session_id":"old-done-session","reply_text":"Old done reply.","metrics":{"audio_chunks":2,"audio_client_written":2,"audio_errors":0,"total_ms":5}}\\n\\n';
        let firstRead = true;
        const reader = {
          read() {
            if (firstRead) {
              firstRead = false;
              return Promise.resolve({value: encoder.encode(pairFrames), done: false});
            }
            return new Promise(resolve => {
              doneBargeReads.push(resolve);
              signalDoneBargeReadReady();
            });
          },
          cancel() { return Promise.resolve(); },
        };
        window.fetch = (url, options = {}) => {
          if (String(url).startsWith('/voice/turn/stream')) {
            return Promise.resolve({ok: true, statusText: 'OK', body: {getReader: () => reader}});
          }
          return originalFetchForDoneBarge(url, options);
        };
        const submitPromise = submitWavBlobAsVoiceTurn(
          new Blob([new Uint8Array(44)], {type: 'audio/wav'}),
          {durationSec: 0.1, sourceLabel: 'combined-done-barge'});
        submitPromise.then(
          () => { doneBargeSettled = true; },
          error => {
            doneBargeError = String((error && error.message) || error);
            doneBargeSettled = true;
          },
        );
        await doneBargeReadReady;
        doneBargeTurn = activeVoiceTurnState;
        if (!doneBargeTurn || doneBargeTurn.telemetry.decodedChunks !== 2) {
          throw new Error('combined done pair did not complete controlled decode');
        }
        const doneBargeFlushPromise =
          flushPrebuffer(doneBargeTurn.turnId, doneBargeTurn);
        if (reviewGates.length !== 1
            || doneBargeTurn.prebufferFlushPromise == null
            || playbackRunStarted !== true
            || prebufferQueue.length !== 0
            || doneBargeReads.length !== 1) {
          throw new Error(
            'combined done explicit flush invariant failed: '
            + JSON.stringify({gateCount: reviewGates.length,
              promise: doneBargeTurn.prebufferFlushPromise != null,
              playbackRunStarted, prebufferDepth: prebufferQueue.length,
              doneReads: doneBargeReads.length}));
        }
        doneBargeReads.shift()({value: encoder.encode(doneFrame), done: false});
        await settleMicrotasks();
        const awaitingDrainBeforeBarge = (
          doneBargeTurn.completed === true
          && doneBargeSettled === false
          && doneBargeTurn.prebufferFlushPromise != null
          && reviewGates.length === 1
          && doneBargeTurn.telemetry.scheduledChunks === 0
        );
        if (!awaitingDrainBeforeBarge) {
          throw new Error('combined done did not await the controlled active flush');
        }

        bargeIn('combined_done_barge', {halfContext: false, playAck: false});
        const terminalAfterBarge = doneBargeTurn.terminalKind;
        const replacementTurnId = currentVoiceTurnId;
        replacementTurn = makeReviewTurn(replacementTurnId);
        replacementTurn.completed = true;
        replacementTurn.terminalKind = 'replacement_owner_sentinel';
        activeVoiceTurnState = replacementTurn;
        sessionId = 'replacement-session-sentinel';
        localStorage.setItem('ms4_session_id', sessionId);
        sessionEl.textContent = sessionId;
        setVoiceStatus('REPLACEMENT_STATUS_SENTINEL', 'ok');
        setOracleStageState(
          'speaking', 'REPLACEMENT_UI_SENTINEL', 'REPLACEMENT_DETAIL_SENTINEL');
        recordVoiceTurnAcceptance(replacementTurn);
        const replacementAcceptanceJson = JSON.stringify(lastVoiceTurnAcceptanceState);

        reviewGates[0]();
        await reviewBound(
          Promise.all([doneBargeFlushPromise, submitPromise]),
          'combined done barge explicit flush');
        await settleMicrotasks();
        var doneBargeRace = {
          awaitingDrainBeforeBarge,
          terminalAfterBarge,
          oldFinalTerminal: doneBargeTurn.terminalKind,
          oldAcceptanceTerminal: doneBargeTurn.acceptanceSnapshot
            ? doneBargeTurn.acceptanceSnapshot.terminalKind : null,
          oldScheduled: doneBargeTurn.telemetry.scheduledChunks,
          oldStarts: reviewMedia.starts.length,
          oldPlaybackRunStarted: playbackRunStarted,
          submitError: doneBargeError,
          submitSettled: doneBargeSettled,
          replacementTurnId,
          activeReplacementPreserved: activeVoiceTurnState === replacementTurn,
          replacementSessionPreserved: sessionId === 'replacement-session-sentinel',
          replacementStoredSessionPreserved:
            localStorage.getItem('ms4_session_id') === 'replacement-session-sentinel',
          replacementPresence: oracleStage.dataset.presence,
          replacementSubtitle: oracleSubtitle.textContent,
          replacementDetail: oracleTimingLine.textContent,
          replacementStatus: voiceStatusLine.textContent,
          replacementAcceptancePreserved:
            JSON.stringify(lastVoiceTurnAcceptanceState) === replacementAcceptanceJson,
          replacementAcceptanceTurnId:
            lastVoiceTurnAcceptanceState && lastVoiceTurnAcceptanceState.turnId,
        };
      } finally {
        window.fetch = originalFetchForDoneBarge;
        activeVoiceTurnState = null;
        sessionId = priorDoneBargeUi.sessionId;
        if (priorDoneBargeUi.storedSessionId == null) {
          localStorage.removeItem('ms4_session_id');
        } else {
          localStorage.setItem('ms4_session_id', priorDoneBargeUi.storedSessionId);
        }
        sessionEl.textContent = priorDoneBargeUi.sessionText;
        oracleStage.dataset.presence = priorDoneBargeUi.presence;
        oracleSubtitle.textContent = priorDoneBargeUi.subtitle;
        oracleTimingLine.textContent = priorDoneBargeUi.detail;
        voiceStatusLine.textContent = priorDoneBargeUi.status;
      }
      // Q29 exact onset regressions. The server delivered c0 at ~2.5 s on a
      // proven capacity-two REST path, but the browser held it until c2 + 600 ms.
      // Drive the production deadline timer and scheduler with the exact c0/c1
      // timings/durations from turns 5 and 6. A same-turn reflex deliberately
      // occupies the old cursor so the same probe proves generated speech
      // preempts waiting audio rather than queueing behind it.
      const runResponseStartCase = async spec => {
        const nativeSetTimeout = window.setTimeout;
        const nativeClearTimeout = window.clearTimeout;
        const priorElapsed = voiceTurnElapsedMs;
        let timerCallback = null;
        let timerDelayMs = null;
        let timerCleared = false;
        let turnMs = spec.firstDecodedMs;
        let contextNowSec = 0;
        let decodeIndex = 0;
        const starts = [];
        const sources = [];
        const timerToken = {responseStartDeadline: spec.name};
        const ctx = {
          state: 'running', destination: {}, sampleRate: 24000,
          get currentTime() { return contextNowSec; },
          decodeAudioData() {
            const duration = spec.durationsSec[decodeIndex];
            const label = spec.name + '-' + decodeIndex;
            decodeIndex += 1;
            return Promise.resolve({label, duration});
          },
          createBufferSource() {
            const source = {
              buffer: null, stopped: false, onended: null,
              connect() {}, disconnect() {},
              start(startAt) {
                starts.push({label: this.buffer.label, startAt, duration: this.buffer.duration});
              },
              stop() { this.stopped = true; },
            };
            sources.push(source);
            return source;
          },
        };
        try {
          oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();
          stopOracleThinkingBridge();
          window.setTimeout = (callback, delayMs) => {
            timerCallback = callback;
            timerDelayMs = delayMs;
            return timerToken;
          };
          window.clearTimeout = token => {
            if (token === timerToken) timerCleared = true;
          };
          resetReview(ctx, spec.turnId);
          voiceTurnElapsedMs = () => turnMs;
          const turn = makeReviewTurn(currentVoiceTurnId);
          turn.telemetry.serverScheduledChunks = spec.serverScheduledChunks;
          applyStatusTtsEngine(turn, 'thinking', {
            engine: 'rest',
            rest_capacity_effective: spec.restCapacity == null ? 2 : spec.restCapacity,
            rest_capacity_provenance:
              spec.restCapacityProvenance || 'measured_concurrency_probe',
          });
          await enqueueAudioChunk(
            '${proofAudioBase64}', 'audio/wav', turn.turnId,
            turn, 'rest', 0, spec.name + '-c0');
          const placeholderSource = {
            stopped: false, stop() { this.stopped = true; }, disconnect() {},
          };
          const placeholderEntry = {
            source: placeholderSource, startAt: 0, endAt: 5,
            turnId: turn.turnId, kind: 'reflex', audible: true,
          };
          activeAudioSources.push(placeholderEntry);
          nextChunkStartTime = 5;
          const beforeDeadline = {
            scheduled: turn.telemetry.scheduledChunks,
            prebufferDepth: prebufferQueue.length,
            timerArmed: prebufferTimer != null,
            requestedMs: timerDelayMs,
          };
          const timerFireLagMs = Number(spec.timerFireLagMs || 0);
          turnMs = spec.firstDecodedMs + timerDelayMs + timerFireLagMs;
          contextNowSec = (timerDelayMs + timerFireLagMs) / 1000;
          if (timerCallback) {
            timerCallback();
            await settleMicrotasks(24);
          }
          const afterDeadline = {
            scheduled: turn.telemetry.scheduledChunks,
            firstScheduledMs: turn.telemetry.firstScheduledMs,
            projectedOnsetMs: turn.telemetry.firstScheduledMs + AUDIO_LEAD_TIME_MS,
            holdReason: turn.telemetry.holdReason,
            sloRelease: Boolean(turn.telemetry.responseStartSloRelease),
            placeholderStopped: placeholderSource.stopped,
            placeholderPreempted: turn.telemetry.replyPlaceholderAudioPreempted,
            placeholderKinds: [...(turn.telemetry.replyPlaceholderKinds || [])],
            responseStartReleaseRunwayMs:
              turn.telemetry.responseStartReleaseRunwayMs,
            firstStartAtMs: starts.length ? starts[0].startAt * 1000 : null,
          };
          turnMs = spec.secondDecodedMs;
          contextNowSec = (spec.secondDecodedMs - spec.firstDecodedMs) / 1000;
          await enqueueAudioChunk(
            '${proofAudioBase64}', 'audio/wav', turn.turnId,
            turn, 'rest', 1, spec.name + '-c1');
          if (spec.flushAfterSecond && starts.length === 0) {
            await flushPrebuffer(turn.turnId, turn);
            await settleMicrotasks(24);
          }
          const rawGapMs = starts.length >= 2
            ? (starts[1].startAt - (starts[0].startAt + starts[0].duration)) * 1000
            : null;
          return {
            input: {...spec}, restTtsCapacity: turn.restTtsCapacity,
            restTtsCapacityProvenance: turn.restTtsCapacityProvenance,
            beforeDeadline, afterDeadline, timerCleared,
            starts: starts.map(item => ({...item})), rawGapMs,
            underflow: {
              count: turn.telemetry.underflowGapCount,
              totalMs: turn.telemetry.underflowGapTotalMs,
              maxMs: turn.telemetry.underflowGapMaxMs,
            },
          };
        } finally {
          window.setTimeout = nativeSetTimeout;
          window.clearTimeout = nativeClearTimeout;
          voiceTurnElapsedMs = priorElapsed;
          if (prebufferTimer === timerToken) prebufferTimer = null;
          activeAudioSources = [];
        }
      };
      const q29ResponseStart = {
        turn5: await runResponseStartCase({
          name: 'q29-turn5', turnId: 929005, serverScheduledChunks: 14,
          firstDecodedMs: 2514, secondDecodedMs: 4386,
          durationsSec: [3.477333333333333, 4.224],
        }),
        turn6: await runResponseStartCase({
          name: 'q29-turn6', turnId: 929006, serverScheduledChunks: 21,
          firstDecodedMs: 2417, secondDecodedMs: 4407,
          durationsSec: [3.296, 5.056],
        }),
        boundary: await runResponseStartCase({
          name: 'response-start-boundary', turnId: 929007, serverScheduledChunks: 2,
          firstDecodedMs: 3000, secondDecodedMs: 3900,
          durationsSec: [2.5, 1.2], timerFireLagMs: 0,
        }),
        lateTimer: await runResponseStartCase({
          name: 'response-start-late-timer', turnId: 929008, serverScheduledChunks: 2,
          firstDecodedMs: 3000, secondDecodedMs: 3900,
          durationsSec: [2.5, 1.2], timerFireLagMs: 1,
        }),
        shortFirstLateSibling: await runResponseStartCase({
          name: 'short-c0-late-c1', turnId: 929009, serverScheduledChunks: 2,
          firstDecodedMs: 1000, secondDecodedMs: 3000,
          durationsSec: [0.1, 1.0], restCapacity: 2, flushAfterSecond: true,
        }),
        oneObservedReplica: await runResponseStartCase({
          name: 'one-observed-replica', turnId: 929010, serverScheduledChunks: 14,
          firstDecodedMs: 2514, secondDecodedMs: 4386,
          durationsSec: [3.477333333333333, 4.224], restCapacity: 1,
          restCapacityProvenance: 'scale_endpoint',
        }),
        unverifiedTwoReplicas: await runResponseStartCase({
          name: 'two-unverified-replicas', turnId: 929011,
          serverScheduledChunks: 14,
          firstDecodedMs: 2514, secondDecodedMs: 4386,
          durationsSec: [3.477333333333333, 4.224], restCapacity: 2,
          restCapacityProvenance: 'scale_response_unverified',
        }),
      };
      const schedulerReview = {
        loneBoundary, restLoneFirstEscape, r9OutstandingSibling,
        adequatePair, earlyThird, staleTimer,
        activeFlush, activeFlushCancellation, realDone, outstandingPairDone,
        doneBargeRace, q29ResponseStart,
      };
      oracleAwaitSinkReadyBeforeStart = () => Promise.resolve();

      let now = 10.000;
      const starts = [];
      const sources = [];
      const fakeCtx = {
        state: 'running',
        destination: {},
        get currentTime() { return now; },
        createBufferSource() {
          const source = {
            buffer: null, stopped: false,
            connect() {},
            start(startAt) {
              starts.push({ label: this.buffer.label, startAt, duration: this.buffer.duration });
            },
            stop() { this.stopped = true; },
            disconnect() {},
            onended: null,
          };
          sources.push(source);
          return source;
        },
      };
      const makeTurnState = turnId => ({
        kind: 'voice', turnId, controller: null, reader: null, assistantState: null,
        userBubble: null, completed: false, aborted: false, terminalKind: null,
        ttsEngine: 'rest', audioReleaseNextIndex: 0, audioPending: null,
        idleAfterPlayback: null, startedAtEpoch: Date.now(), startedAtPerf: performance.now(),
        serverMetrics: null, onsetAnalyser: null, onsetData: null, onsetMonitorFrame: 0,
        telemetry: {
          firstAcknowledgementMs: null, firstSseReceivedMs: null, firstDecodedMs: null,
          firstScheduledMs: null, firstScheduledContextTime: null,
          firstNonSilentOnsetMs: null, onsetRms: null, serverScheduledChunks: 0,
          sseAudioChunks: 0, decodedChunks: 0, scheduledChunks: 0,
          audioErrors: 0, decodeErrors: 0, underflowGapCount: 0,
          underflowGapTotalMs: 0, underflowGapMaxMs: 0,
        },
      });
      playbackCtx = fakeCtx;
      playbackAudioUnlocked = true;
      nextChunkStartTime = 0;
      activeAudioSources = [];
      prebufferQueue = [];
      prebufferTimer = null;
      playbackRunStarted = true;
      currentVoiceTurnId = 900001;
      activeVoiceTurnState = null;
      activeChatTurnState = null;
      activeVoiceStreamController = null;
      activeOracleAudioFetchControllers.clear();

      const timingTurn = makeTurnState(currentVoiceTurnId);
      await scheduleDecodedChunk({ label: 'A', duration: 0.200 }, currentVoiceTurnId, timingTurn);
      now = 10.100;
      await scheduleDecodedChunk({ label: 'B', duration: 0.300 }, currentVoiceTurnId, timingTurn);
      now = 10.700;
      await scheduleDecodedChunk({ label: 'C', duration: 0.100 }, currentVoiceTurnId, timingTurn);

      const ms = seconds => Math.round(seconds * 1000);
      const [a, b, c] = starts;
      const aEnd = a.startAt + a.duration;
      const bEnd = b.startAt + b.duration;
      const unavoidableGapMs = ms(now - bEnd);
      const actualLateGapMs = ms(c.startAt - bEnd);

      timingTurn.controller = new AbortController();
      timingTurn.reader = { cancel() {} };
      timingTurn.audioPending = new Map([[7, { label: 'queued-old-turn', duration: 0.250 }]]);
      activeVoiceTurnState = timingTurn;
      activeVoiceStreamController = timingTurn.controller;
      prebufferQueue = [{ audioBuffer: { label: 'queued', duration: 0.250 }, turnId: currentVoiceTurnId }];
      prebufferTimer = setTimeout(() => {}, 60000);
      const oldTurnId = currentVoiceTurnId;
      _coreBargeIn(null, { playAck: false });
      const createdBeforeStale = sources.length;
      await scheduleDecodedChunk({ label: 'STALE', duration: 0.100 }, oldTurnId, timingTurn);
      const cancellation = {
        pendingDepth: timingTurn.audioPending.size,
        prebufferDepth: prebufferQueue.length,
        timerCleared: prebufferTimer === null,
        playbackRunStarted,
        activeSources: activeAudioSources.length,
        sourcesStopped: sources.every(source => source.stopped),
        controllerAborted: timingTurn.controller.signal.aborted,
        staleCreatedSources: sources.length - createdBeforeStale,
      };

      const freshTurn = makeTurnState(currentVoiceTurnId);
      await scheduleDecodedChunk({ label: 'D', duration: 0.100 }, currentVoiceTurnId, freshTurn);
      const d = starts[3];

      result = {
        startupPair,
        slowCadencePair,
        cadenceBoundary,
        schedulerReview,
        order: starts.slice(0, 3).map(item => item.label),
        scheduledChunks: timingTurn.telemetry.scheduledChunks,
        firstAudioLeadMs: ms(a.startAt - 10.000),
        abGapMs: ms(b.startAt - aEnd),
        unavoidableGapMs,
        actualLateGapMs,
        artificialRestartDelayMs: actualLateGapMs - unavoidableGapMs,
        cancellation,
        freshTurn: {
          label: d.label,
          turnIdAdvanced: freshTurn.turnId > oldTurnId,
          firstAudioLeadMs: ms(d.startAt - now),
          scheduledChunks: freshTurn.telemetry.scheduledChunks,
        },
      };
    } finally {
      if (prebufferTimer && prebufferTimer !== saved.prebufferTimer) clearTimeout(prebufferTimer);
      playbackCtx = saved.playbackCtx;
      playbackAudioUnlocked = saved.playbackAudioUnlocked;
      nextChunkStartTime = saved.nextChunkStartTime;
      activeAudioSources = saved.activeAudioSources;
      prebufferQueue = saved.prebufferQueue;
      prebufferTimer = saved.prebufferTimer;
      playbackRunStarted = saved.playbackRunStarted;
      currentVoiceTurnId = saved.currentVoiceTurnId;
      activeVoiceTurnState = saved.activeVoiceTurnState;
      activeChatTurnState = saved.activeChatTurnState;
      activeVoiceStreamController = saved.activeVoiceStreamController;
      lastVoiceTurnAcceptanceState = saved.lastVoiceTurnAcceptanceState;
      lastVoiceTurnAcceptanceTurnId = saved.lastVoiceTurnAcceptanceTurnId;
      oracleAwaitSinkReadyBeforeStart = saved.gate;
      voiceTurnElapsedMs = saved.elapsed;
      if (saved.oracleMuted === undefined) delete document.documentElement.dataset.oracleMuted;
      else document.documentElement.dataset.oracleMuted = saved.oracleMuted;
      activeOracleAudioFetchControllers.clear();
      for (const controller of saved.audioControllers) activeOracleAudioFetchControllers.add(controller);
    }
    return result;
  // This transaction intentionally replays the physical ~19 s startup receipt,
  // Q20/Q26, and both Q29 onset cases in one browser ownership interval.
  })()`, { timeoutMs: CDP_BEHAVIOR_TIMEOUT_MS + 30_000 });
  assert.deepEqual(schedulerTiming.order, ['A', 'B', 'C'], 'scheduler preserves decoded A/B/C order');
  assert.equal(schedulerTiming.scheduledChunks, 3, 'A/B/C exercise production turn telemetry');
  assert.equal(schedulerTiming.firstAudioLeadMs, 150, 'first audio retains the intentional 150 ms lead');
  assert.equal(schedulerTiming.abGapMs, 0, 'ahead-of-playback B starts exactly at A end');
  assert.equal(schedulerTiming.unavoidableGapMs, 50, 'late C has exactly 50 ms of producer lateness');
  assert.equal(
    schedulerTiming.actualLateGapMs,
    schedulerTiming.unavoidableGapMs,
    `late C adds no restart lead: unavoidable=${schedulerTiming.unavoidableGapMs}ms `
      + `actual=${schedulerTiming.actualLateGapMs}ms artificial=${schedulerTiming.artificialRestartDelayMs}ms`,
  );
  assert.equal(schedulerTiming.artificialRestartDelayMs, 0, 'late C adds zero artificial silence');
  assert.equal(schedulerTiming.cancellation.pendingDepth, 0, 'barge clears indexed pending audio');
  assert.equal(schedulerTiming.cancellation.prebufferDepth, 0, 'barge clears prebuffered audio');
  assert.equal(schedulerTiming.cancellation.timerCleared, true, 'barge clears the prebuffer timer');
  assert.equal(schedulerTiming.cancellation.playbackRunStarted, false, 'barge resets playback-run state');
  assert.equal(schedulerTiming.cancellation.activeSources, 0, 'barge removes active audio sources');
  assert.equal(schedulerTiming.cancellation.sourcesStopped, true, 'barge stops every active source');
  assert.equal(schedulerTiming.cancellation.controllerAborted, true, 'barge aborts the old turn controller');
  assert.equal(schedulerTiming.cancellation.staleCreatedSources, 0, 'late old-turn work schedules no stale source');
  assert.equal(schedulerTiming.freshTurn.label, 'D', 'fresh turn schedules only its own audio');
  assert.equal(schedulerTiming.freshTurn.turnIdAdvanced, true, 'fresh playback belongs to the new turn');
  assert.equal(schedulerTiming.freshTurn.firstAudioLeadMs, 150, 'a reset turn regains the 150 ms initial lead');
  assert.equal(schedulerTiming.freshTurn.scheduledChunks, 1, 'fresh turn starts production telemetry at one chunk');
  experiencePolish.finalEvidenceState = await cdp.evaluate(`(async () => {
    bargeIn(null, {halfContext: false, playAck: false});
    stopOracleOptionalAudio();
    await new Promise(resolve => setTimeout(resolve, 180));
    resetOraclePlaybackSignal();
    setOracleStageTranscript('');
    setOracleStageState('idle', 'Ready.', '');
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return {
      presence: oracleStage.dataset.presence,
      turnPhase: oracleStage.dataset.turnPhase,
      audioReactive: oracleStage.dataset.audioReactive,
      subtitle: oracleSubtitle.textContent,
      detail: oracleTimingLine.textContent,
      activeSpeechSources: activeAudioSources.length,
      baseVoiceCells: oracleMirrorCellSpecs().filter(([, , , , shape]) => shape === 'voice').length,
      speakingBarsOpacity: Number.parseFloat(
        getComputedStyle(oracleMirror.querySelector('.mirror-speaking-bars')).opacity),
      neuralWaveOpacity: Number.parseFloat(
        getComputedStyle(oracleMirror.querySelector('.mirror-wave')).opacity),
      optionalAudio: oracleOptionalAudioState(),
    };
  })()`);
  assert.deepEqual(experiencePolish.finalEvidenceState, {
    presence: 'idle',
    turnPhase: 'idle',
    audioReactive: 'false',
    subtitle: 'Ready.',
    detail: '',
    activeSpeechSources: 0,
    baseVoiceCells: 0,
    speakingBarsOpacity: 0,
    neuralWaveOpacity: 0,
    optionalAudio: {activeSources: 0, speechLikeSources: 0, ambienceActive: false},
  }, 'final evidence must be normalized to one coherent idle state before capture');

  // ---- Identity + result (finding 6 served-bytes binding). ------------------
  const identity = {
    variant, variantId, runId, acceptanceTarget, pageUrl, loopbackUrl, fixtureNonce, fixturePort, debugPort,
    servedHtmlSha256, servedHtmlBytes, fixtureIdentity, dspPath, dspSha256,
    voiceInputSessionPath, voiceInputSessionSha256,
    // B-01: exact executed harness + proof-audio + Node/Chrome executable digests.
    harnessPath, harnessSha256, proofAudioSha256, silentAudioSha256,
    nodeExecPath, nodeExecSha256, chromeExecSha256,
    chrome: chromeIdentity, chromePath: resolvedChromePath, chromeArgs, nodeVersion: process.version, indexPath,
    profileDir, profileOwnedByHarness: ownsProfileCleanup,
    stagingDir, stagingOwnedByHarness: ownsStagingCleanup, wrapperDriven,
    publishDir: evidenceDir ? join(evidenceDir, `${variant}__${runId}`) : null,
    env: {
      MS4_BROWSER_VARIANT: variant, MS4_BROWSER_TARGET_URL: inheritedTarget,
      MS4_BROWSER_TARGET_PINNED: process.env.MS4_BROWSER_TARGET_PINNED || null,
      MS4_CHROME_PATH: process.env.MS4_CHROME_PATH || null, MS4_BROWSER_EVIDENCE_DIR: evidenceDir,
      MS4_BROWSER_REPO_ROOT: process.env.MS4_BROWSER_REPO_ROOT || null,
    },
  };
  const result = { ok: true, identity, visual, experiencePolish, dspDelegation, staleCues, facePrewarm, streamDeadline, bargeAck, ownershipRace, abort, audibleVerdict, retained, voiceReadinessState, snapshotOwnership, longInputSession, vad, fullDuplexRace, deferredAutoArm, productRepair, restBatch, restObservability, schedulerTiming };
  const resultJson = `${JSON.stringify(result, null, 2)}\n`;
  resultJsonForSeal = resultJson;

  // ---- R6: the harness writes evidence ONLY into private staging and NEVER ---
  // publishes. Every artifact is written with exclusive-create (non-replacing)
  // writes and a completion SEAL. The CALLER (Python wrapper / direct CLI) owns
  // the final transaction: prove tree death, delete the identity-bound profile,
  // then atomically commit staging -> the accepted `<variant>__<runId>` dir, and
  // fail closed (no accepted evidence) on any cleanup failure.
  if (stagingDir) {
    const publish = async (name, data) => { await writeFile(join(stagingDir, name), data, { flag: 'wx' }); };
    await cdp.call('Page.bringToFront');
    const clip = await cdp.evaluate(`(() => { window.scrollTo(0,0); const r = oracleStage.getBoundingClientRect(); return { x: r.x+scrollX, y: r.y+scrollY, width: r.width, height: r.height, scale: 1 }; })()`);
    await cdp.evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
    const shot = await cdp.call('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip });
    const finalPng = Buffer.from(shot.data, 'base64');
    const artifacts = {};
    const bind = (name, buf) => { artifacts[name] = { sha256: createHash('sha256').update(buf).digest('hex').toUpperCase(), bytes: buf.length }; };
    if (oracleIdlePng) { await publish(`oracle_experience_idle__${variant}.png`, oracleIdlePng); bind(`oracle_experience_idle__${variant}.png`, oracleIdlePng); }
    if (oracleMobilePng) { await publish(`oracle_experience_mobile_390x844__${variant}.png`, oracleMobilePng); bind(`oracle_experience_mobile_390x844__${variant}.png`, oracleMobilePng); }
    if (hiveCellsPng) { await publish(`oracle_hive_cells__${variant}.png`, hiveCellsPng); bind(`oracle_hive_cells__${variant}.png`, hiveCellsPng); }
    await publish(`oracle_final_state__${variant}.png`, finalPng); bind(`oracle_final_state__${variant}.png`, finalPng);
    const resultBuf = Buffer.from(resultJson, 'utf8');
    await publish(`oracle_browser_runtime__${variant}.json`, resultBuf); bind(`oracle_browser_runtime__${variant}.json`, resultBuf);
    // A4: bind nonce/variant/run/source + runtime digests + the exact artifact
    // hash set into the publish manifest, and then bind the manifest's own hash
    // into the inner SEAL, so the inner seal is a complete hash graph over every
    // staged artifact. The caller re-validates this graph BEFORE the commit.
    const publishManifest = {
      schema: 'OracleR6PublishManifest.v1',
      variant, variantId, runId, fixtureNonce, servedHtmlSha256, servedHtmlBytes,
      harnessSha256, voiceInputSessionSha256, proofAudioSha256, nodeExecSha256, chromeExecSha256,
      pageUrl, chrome: chromeIdentity && chromeIdentity.Browser, artifacts,
    };
    const publishManifestBuf = Buffer.from(`${JSON.stringify(publishManifest, null, 2)}\n`, 'utf8');
    await publish('publish_manifest.json', publishManifestBuf);
    const publishManifestSha256 = createHash('sha256').update(publishManifestBuf).digest('hex').toUpperCase();
    // Completion seal, written LAST inside staging: binds EVERY artifact hash and
    // the publish-manifest hash. Its presence + validity is the atomic proof that
    // every artifact was written before the caller's commit rename.
    await publish('SEAL.json', Buffer.from(`${JSON.stringify({
      schema: 'OracleR6InnerSeal.v1', sealed: true, variant, variantId, runId,
      fixtureNonce, servedHtmlSha256, harnessSha256, voiceInputSessionSha256, proofAudioSha256,
      nodeExecSha256, chromeExecSha256,
      artifacts, publishManifestSha256,
    }, null, 2)}\n`, 'utf8'));
  }

  bodySucceeded = true;
} finally {
  // R6 ONE cleanup boundary. Reject pending CDP calls, kill the ENTIRE owned
  // Chrome TREE (children hold the profile's Windows file locks), AWAIT the main
  // process exit (CHECKED), close listeners. When WE own the profile/staging
  // (raw-node, no wrapper) remove our EXACT identity-bound dirs with retries and
  // NEVER swallow a failure. Wrapper-driven runs leave profile + staging to the
  // wrapper, which owns the handle-bound tree-death proof + atomic commit.
  if (cdp) cdp.dispose();
  // Test hook (never set by acceptance runs): simulate a taskkill failure so the
  // WRAPPER's handle-bound Job Object must be the authoritative tree reaper.
  if (process.env.MS4_BROWSER_TEST_TASKKILL_FAIL === '1') {
    treeKillStatus = { ok: false, method: 'test-hook-skip', error: 'forced taskkill failure' };
  } else {
    treeKillStatus = chrome && chrome.pid != null ? killProcessTree(chrome.pid) : { ok: true, note: 'no chrome' };
  }
  if (chrome && chrome.exitCode === null && chrome.signalCode === null) {
    try { chrome.kill('SIGKILL'); } catch (_error) { /* already gone */ }
  }
  chromeExitObserved = chrome ? await awaitChildExit(chrome, 5_000) : true;
  // Test hook (never set by acceptance runs): simulate an unobserved Chrome exit.
  if (process.env.MS4_BROWSER_TEST_CLEANUP_FAIL === '1') chromeExitObserved = false;
  if (debugServer) { try { await new Promise(res => debugServer.close(res)); } catch (_e) {} }
  if (fixtureServer) { try { await new Promise(res => fixtureServer.close(res)); } catch (_e) {} }
  if (ownsProfileCleanup) {
    profileResidue = await removeOwnedDir(profileDir, tmpdir(), PROFILE_PREFIX, profileIdentity);
  }
  if (ownsStagingCleanup && stagingDir) {
    stagingResidue = await removeOwnedDir(stagingDir, evidenceDir, STAGING_PREFIX, stagingIdentity);
  }
}

// ---- R6 fail-closed boundary (runs only on a fully successful body) ---------
// The harness never publishes. It fails closed ONLY on residue it OWNS (raw-node
// path). Wrapper-driven runs: the harness has written staging + inner SEAL; the
// Python wrapper/direct CLI performs the tree-death proof, identity-bound profile
// deletion, atomic commit, and cleanup attestation, and fails closed there.
if (!chromeExitObserved) {
  if (ownsStagingCleanup && stagingDir) await rm(stagingDir, { recursive: true, force: true }).catch(() => {});
  throw new Error('cleanup fail-closed: Chrome process did not exit within the observed cap (no accepted evidence)');
}
if (ownsProfileCleanup && profileResidue && profileResidue.removed !== true) {
  throw new Error(`cleanup fail-closed: run-owned profile could not be removed: ${JSON.stringify(profileResidue)}`);
}
if (ownsStagingCleanup && stagingResidue && stagingResidue.removed !== true) {
  throw new Error(`cleanup fail-closed: run-owned staging could not be removed: ${JSON.stringify(stagingResidue)}`);
}
// stdout is byte-identical to the persisted staging JSON (the caller cross-checks
// this against the committed artifact).
process.stdout.write(resultJsonForSeal);
