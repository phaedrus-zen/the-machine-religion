'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const pwa = require(path.join(__dirname, '..', 'machine_spirit_4', 'web', 'oracle_remote_pwa.js'));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function fakeClock() {
  let now = 100;
  let nextId = 1;
  const timers = new Map();
  return {
    now: () => now,
    advance: ms => { now += ms; },
    setTimeout(fn, ms) { const id = nextId++; timers.set(id, {fn, ms}); return id; },
    clearTimeout(id) { timers.delete(id); },
    nextDelay() { return [...timers.values()].map(item => item.ms).sort((a, b) => a - b)[0]; },
    timerCount() { return timers.size; },
    runDelay(ms) {
      const entry = [...timers.entries()].find(([, item]) => item.ms === ms);
      assert.ok(entry, `missing ${ms}ms timer`);
      timers.delete(entry[0]);
      entry[1].fn();
    },
  };
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise(resolve => setImmediate(resolve));
}

async function testSameOriginAndBackoff() {
  const loc = {origin: 'https://oracle.private.example'};
  assert.equal(
    pwa.sameOriginUrl(pwa.DEFAULT_STATUS_ENDPOINT, loc),
    'https://oracle.private.example/api/v1/ms4_gateway/status',
  );
  assert.throws(() => pwa.sameOriginUrl('//evil.example/health', loc), /same-origin/);
  assert.deepEqual([1, 2, 3, 4, 5, 99].map(n => pwa.reconnectDelay(n)), [1000, 2000, 4000, 8000, 15000, 15000]);
}

async function testBoundedReconnectAndLatency() {
  const clock = fakeClock();
  const states = [];
  let calls = 0;
  const monitor = pwa.createConnectionMonitor({
    fetchFn: async (url, options) => {
      calls += 1;
      assert.equal(url, 'https://oracle.private.example/api/v1/ms4_gateway/status');
      assert.equal(options.redirect, 'error');
      if (calls === 1) throw new Error('offline');
      clock.advance(37);
      return {
        ok: true,
        status: 200,
        async json() {
          return {ok: true, service: 'ms4-gateway', status: 'ready', runtime: 'ms4-fusion'};
        },
      };
    },
    locationLike: {origin: 'https://oracle.private.example'},
    clock,
    onState: state => states.push(state),
  });
  monitor.start();
  await settle();
  assert.equal(monitor.inspect().status, 'offline');
  assert.equal(monitor.inspect().nextRetryMs, 1000);
  assert.equal(clock.nextDelay(), 1000);
  clock.runDelay(1000);
  await settle();
  assert.equal(monitor.inspect().status, 'online');
  assert.equal(monitor.inspect().latencyMs, 37);
  assert.equal(clock.nextDelay(), 15000);
  assert.ok(states.some(state => state.status === 'reconnecting'));
  monitor.stop();
}

async function testConnectionRequiresExactGatewayIdentity() {
  const clock = fakeClock();
  let bodyReads = 0;
  const monitor = pwa.createConnectionMonitor({
    fetchFn: async () => ({
      ok: true,
      status: 200,
      async json() {
        bodyReads += 1;
        return {ok: true, service: 'wrong-backend', status: 'ready', runtime: 'not-ms4'};
      },
    }),
    locationLike: {origin: 'https://oracle.private.example'},
    clock,
  });
  monitor.start();
  await settle();
  assert.equal(bodyReads, 1);
  assert.equal(monitor.inspect().status, 'misrouted');
  assert.match(monitor.inspect().error, /expected service=ms4-gateway/);
  assert.equal(clock.nextDelay(), 1000);
  monitor.stop();

  assert.deepEqual(
    pwa.validateGatewayIdentity({ok: true, service: 'ms4-gateway', status: 'ready', runtime: 'ms4-fusion'}),
    {
      ok: true,
      identity: {service: 'ms4-gateway', status: 'ready', runtime: 'ms4-fusion'},
    },
  );
  assert.equal(pwa.validateGatewayIdentity(null).code, 'malformed_identity');
}

async function testRestartKeepsReplacementProbeOwned() {
  const clock = fakeClock();
  const requests = [];
  const monitor = pwa.createConnectionMonitor({
    fetchFn: (_url, options) => {
      const gate = deferred();
      requests.push({gate, signal: options.signal});
      return gate.promise;
    },
    locationLike: {origin: 'https://oracle.private.example'},
    clock,
  });
  monitor.start();
  await settle();
  monitor.stop();
  monitor.start();
  await settle();
  assert.equal(requests.length, 2);
  requests[0].gate.reject(new Error('old aborted request settled late'));
  await settle();
  void monitor.retryNow();
  await settle();
  assert.equal(requests.length, 2, 'stale finalizer must not admit a third probe');
  assert.equal(clock.timerCount(), 1, 'only the replacement timeout remains owned');
  monitor.stop();
  assert.equal(requests[1].signal.aborted, true, 'stop must abort the replacement probe');
  requests[1].gate.reject(new Error('replacement aborted'));
  await settle();
  assert.equal(monitor.inspect().status, 'stopped');
}

async function testWakeLockTracksHandsFreeIntent() {
  const listeners = {};
  const doc = {
    visibilityState: 'visible',
    addEventListener(name, fn) { listeners[name] = fn; },
  };
  let requests = 0;
  let releases = 0;
  let releaseListener = null;
  const nav = {wakeLock: {request: async kind => {
    assert.equal(kind, 'screen');
    requests += 1;
    return {
      addEventListener(name, fn) { if (name === 'release') releaseListener = fn; },
      async release() { releases += 1; if (releaseListener) releaseListener(); },
    };
  }}};
  const lock = pwa.createWakeLockController({navigatorLike: nav, documentLike: doc});
  await lock.setDesired(true);
  assert.deepEqual(lock.inspect(), {desired: true, held: true, supported: true});
  await lock.setDesired(false);
  assert.equal(releases, 1);
  assert.equal(lock.inspect().held, false);
  await lock.setDesired(true);
  releaseListener();
  listeners.visibilitychange();
  await settle();
  assert.equal(requests, 3);
}

async function testWakeLockDisableAndOverlapRaces() {
  const listeners = {};
  const doc = {
    visibilityState: 'visible',
    addEventListener(name, fn) { listeners[name] = fn; },
  };
  const gates = [];
  let releases = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: () => {
      const gate = deferred();
      gates.push(gate);
      return gate.promise;
    }}},
    documentLike: doc,
  });

  const staleEnable = lock.setDesired(true);
  await settle();
  await lock.setDesired(false);
  gates[0].resolve({addEventListener() {}, async release() { releases += 1; }});
  assert.deepEqual(await staleEnable, {desired: false, held: false, supported: true});
  assert.deepEqual(lock.inspect(), {desired: false, held: false, supported: true});
  assert.equal(releases, 1, 'late disabled acquisition is immediately released');

  const currentEnable = lock.setDesired(true);
  listeners.visibilitychange();
  listeners.visibilitychange();
  await settle();
  assert.equal(gates.length, 2, 'one pending acquisition owns overlapping visibility events');
  gates[1].resolve({addEventListener() {}, async release() { releases += 1; }});
  await currentEnable;
  assert.equal(lock.inspect().held, true);
  doc.visibilityState = 'hidden';
  listeners.visibilitychange();
  await settle();
  assert.deepEqual(lock.inspect(), {desired: true, held: false, supported: true});
  assert.equal(releases, 2);
}

async function testWakeLockSystemReleaseSelfHealsOnce() {
  const clock = fakeClock();
  const doc = {visibilityState: 'visible', addEventListener() {}};
  const releaseListeners = [];
  let requests = 0;
  let explicitReleases = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: async () => {
      requests += 1;
      const ordinal = requests;
      return {
        addEventListener(name, fn) {
          if (name === 'release') releaseListeners[ordinal - 1] = fn;
        },
        async release() {
          explicitReleases += 1;
          releaseListeners[ordinal - 1]();
        },
      };
    }}},
    documentLike: doc,
    clock,
  });
  await lock.setDesired(true);
  releaseListeners[0]();
  releaseListeners[0]();
  await settle();
  assert.equal(requests, 1, 'system release cannot recurse through the promise queue');
  assert.equal(clock.nextDelay(), 250);
  clock.runDelay(250);
  await settle();
  assert.equal(requests, 2, 'one system release admits one delayed replacement request');
  assert.deepEqual(lock.inspect(), {desired: true, held: true, supported: true});
  await lock.setDesired(false);
  await settle();
  assert.equal(explicitReleases, 1);
  assert.equal(requests, 2, 'explicit disable cannot reacquire');
  assert.deepEqual(lock.inspect(), {desired: false, held: false, supported: true});
}

async function testWakeLockImmediateRevocationAfterAcquireSelfHeals() {
  const clock = fakeClock();
  const doc = {visibilityState: 'visible', addEventListener() {}};
  let requests = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: async () => {
      requests += 1;
      const ordinal = requests;
      return {
        addEventListener(name, fn) {
          if (name === 'release' && ordinal === 1) fn();
        },
        async release() {},
      };
    }}},
    documentLike: doc,
    clock,
  });
  await lock.setDesired(true);
  await settle();
  assert.equal(requests, 1);
  assert.equal(clock.nextDelay(), 250);
  clock.runDelay(250);
  await settle();
  assert.equal(requests, 2);
  assert.deepEqual(lock.inspect(), {desired: true, held: true, supported: true});
  await lock.setDesired(false);
}

async function testWakeLockAlreadyReleasedSentinelSelfHeals() {
  const clock = fakeClock();
  const doc = {visibilityState: 'visible', addEventListener() {}};
  let requests = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: async () => {
      requests += 1;
      return {
        released: requests === 1,
        addEventListener() {},
        async release() {},
      };
    }}},
    documentLike: doc,
    clock,
  });
  await lock.setDesired(true);
  await settle();
  assert.equal(requests, 1);
  assert.equal(clock.nextDelay(), 250);
  clock.runDelay(250);
  await settle();
  assert.equal(requests, 2);
  assert.deepEqual(lock.inspect(), {desired: true, held: true, supported: true});
  await lock.setDesired(false);
}

async function testWakeLockPersistentRevocationUsesBoundedBackoff() {
  const clock = fakeClock();
  const doc = {visibilityState: 'visible', addEventListener() {}};
  let requests = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: async () => {
      requests += 1;
      return {released: true, addEventListener() {}, async release() {}};
    }}},
    documentLike: doc,
    clock,
    retryMs: [10, 20, 40],
  });
  await lock.setDesired(true);
  await settle();
  assert.equal(requests, 1, 'already-released sentinels cannot microtask-spin');
  assert.equal(clock.nextDelay(), 10);
  clock.runDelay(10);
  await settle();
  assert.equal(requests, 2);
  assert.equal(clock.nextDelay(), 20);
  clock.runDelay(20);
  await settle();
  assert.equal(requests, 3);
  assert.equal(clock.nextDelay(), 40);
  clock.runDelay(40);
  await settle();
  assert.equal(requests, 4);
  assert.equal(clock.nextDelay(), 40, 'retry delay is capped at the final schedule entry');
  await lock.setDesired(false);
  assert.equal(clock.timerCount(), 0, 'explicit disable cancels the owned retry');
}

async function testWakeLockRequestErrorsBackOffAndHideCancels() {
  const clock = fakeClock();
  const listeners = {};
  const doc = {
    visibilityState: 'visible',
    addEventListener(name, fn) { listeners[name] = fn; },
  };
  let requests = 0;
  const lock = pwa.createWakeLockController({
    navigatorLike: {wakeLock: {request: async () => {
      requests += 1;
      throw new Error('denied');
    }}},
    documentLike: doc,
    clock,
    retryMs: [5, 15],
  });
  await lock.setDesired(true);
  await settle();
  assert.equal(requests, 1);
  assert.equal(clock.nextDelay(), 5);
  clock.runDelay(5);
  await settle();
  assert.equal(requests, 2);
  assert.equal(clock.nextDelay(), 15);
  doc.visibilityState = 'hidden';
  listeners.visibilitychange();
  await settle();
  assert.equal(clock.timerCount(), 0, 'hidden pages cancel denied-request retry ownership');
  assert.equal(requests, 2);
}

async function testServiceWorkerReadinessAndGenerationReload() {
  const listeners = {};
  const workerListeners = {};
  const state = {textContent: '', dataset: {}};
  const expectedUrl = `https://oracle.private.example${pwa.buildAssetPath('/service-worker.js', pwa.BUILD)}`;
  const installing = {
    state: 'installing',
    scriptURL: expectedUrl,
    addEventListener(name, fn) { workerListeners[name] = fn; },
  };
  const registration = {
    scope: 'https://oracle.private.example/',
    installing,
    waiting: null,
    active: null,
  };
  let registered = null;
  let reloads = 0;
  const storage = new Map();
  const windowLike = {
    isSecureContext: true,
    __ORACLE_PWA_EXPECTED_BUILD: pwa.BUILD,
    location: {
      origin: 'https://oracle.private.example',
      reload() { reloads += 1; },
    },
    sessionStorage: {
      getItem(key) { return storage.get(key) || null; },
      setItem(key, value) { storage.set(key, value); },
    },
    navigator: {serviceWorker: {
      controller: null,
      async register(url, options) {
        registered = {url, options};
        return registration;
      },
      addEventListener(name, fn) { listeners[name] = fn; },
    }},
  };
  const documentLike = {getElementById: id => id === 'oraclePwaState' ? state : null};
  const result = await pwa.registerServiceWorker(windowLike, documentLike);
  assert.equal(result.status, 'installing');
  assert.equal(state.textContent, 'PWA installing');
  assert.deepEqual(registered, {
    url: pwa.buildAssetPath('/service-worker.js', pwa.BUILD),
    options: {scope: '/', updateViaCache: 'none'},
  });

  const active = {state: 'activated', scriptURL: expectedUrl, addEventListener() {}};
  registration.installing = null;
  registration.active = active;
  windowLike.navigator.serviceWorker.controller = active;
  listeners.controllerchange();
  assert.equal(state.textContent, 'PWA ready');
  assert.equal(reloads, 1);
  listeners.controllerchange();
  assert.equal(reloads, 1, 'controller reconciliation reload is one-shot per build');

  windowLike.navigator.serviceWorker.controller = {
    state: 'activated',
    scriptURL: 'https://oracle.private.example/service-worker.js?build=older',
  };
  assert.equal(
    pwa.inspectServiceWorkerReadiness(windowLike, registration, pwa.BUILD).status,
    'generation_mismatch',
  );
}

async function testServiceWorkerActivatesExpectedWaitingGenerationOnce() {
  const listeners = {};
  const workerListeners = {};
  const state = {textContent: '', dataset: {}};
  const expectedUrl = `https://oracle.private.example${pwa.buildAssetPath('/service-worker.js', pwa.BUILD)}`;
  const older = {
    state: 'activated',
    scriptURL: 'https://oracle.private.example/service-worker.js?build=older',
    addEventListener() {},
  };
  const messages = [];
  const waiting = {
    state: 'installed',
    scriptURL: expectedUrl,
    postMessage(value) { messages.push(value); },
    addEventListener(name, fn) { workerListeners[name] = fn; },
  };
  const registration = {
    scope: 'https://oracle.private.example/',
    installing: null,
    waiting,
    active: older,
    addEventListener() {},
  };
  let reloads = 0;
  const windowLike = {
    isSecureContext: true,
    __ORACLE_PWA_EXPECTED_BUILD: pwa.BUILD,
    location: {origin: 'https://oracle.private.example', reload() { reloads += 1; }},
    sessionStorage: {getItem() { return null; }, setItem() {}},
    navigator: {serviceWorker: {
      controller: older,
      async register() { return registration; },
      addEventListener(name, fn) { listeners[name] = fn; },
    }},
  };
  const result = await pwa.registerServiceWorker(
    windowLike,
    {getElementById: id => id === 'oraclePwaState' ? state : null},
  );
  assert.equal(result.status, 'generation_mismatch');
  assert.deepEqual(messages, [{type: 'oracle-activate-build', build: pwa.BUILD}]);
  workerListeners.statechange();
  workerListeners.statechange();
  assert.equal(messages.length, 1, 'one waiting worker owns one activation request');

  registration.waiting = null;
  registration.active = waiting;
  waiting.state = 'activated';
  windowLike.navigator.serviceWorker.controller = waiting;
  listeners.controllerchange();
  assert.equal(state.textContent, 'PWA ready');
  assert.equal(reloads, 1);
}

async function testFailedControllerReloadDoesNotConsumeFence() {
  const expectedUrl = `https://oracle.private.example${pwa.buildAssetPath('/service-worker.js', pwa.BUILD)}`;
  const storage = new Map();
  let reloads = 0;
  const windowLike = {
    location: {
      origin: 'https://oracle.private.example',
      reload() { reloads += 1; throw new Error('navigation blocked'); },
    },
    navigator: {serviceWorker: {controller: {scriptURL: expectedUrl}}},
    sessionStorage: {
      getItem(key) { return storage.get(key) || null; },
      setItem(key, value) { storage.set(key, value); },
      removeItem(key) { storage.delete(key); },
    },
  };
  assert.equal(pwa.reloadOnceForExpectedController(windowLike, pwa.BUILD), false);
  assert.equal(storage.size, 0);
  assert.notEqual(windowLike.__ORACLE_PWA_CONTROLLER_RELOAD_BUILD, pwa.BUILD);
  windowLike.location.reload = () => { reloads += 1; };
  assert.equal(pwa.reloadOnceForExpectedController(windowLike, pwa.BUILD), true);
  assert.equal(reloads, 2);
}

async function testServiceWorkerStrictStaticBoundary() {
  const workerPath = path.join(__dirname, '..', 'machine_spirit_4', 'web', 'service-worker.js');
  const source = fs.readFileSync(workerPath, 'utf8');
  const listeners = {};
  const writes = [];
  const matches = [];
  const fetches = [];
  let fetchResult = {ok: true, body: 'network root'};
  const offlineShell = {ok: true, body: 'offline shell'};
  const cache = {
    async addAll(items) { writes.push({kind: 'addAll', items: [...items]}); },
    async put(key) { writes.push({kind: 'put', key}); },
  };
  let skipWaitingCalls = 0;
  const skipWaitingGate = deferred();
  const context = vm.createContext({
    URL,
    fetch: async request => {
      fetches.push(request.url || request);
      if (fetchResult instanceof Error) throw fetchResult;
      return fetchResult;
    },
    caches: {
      open: async () => cache,
      keys: async () => [],
      delete: async () => true,
      match: async key => {
        matches.push(typeof key === 'string' ? key : key.url);
        return key === '/' ? offlineShell : null;
      },
    },
    self: {
      location: {origin: 'https://oracle.private.example'},
      clients: {claim: async () => {}},
      skipWaiting() { skipWaitingCalls += 1; return skipWaitingGate.promise; },
      addEventListener(name, handler) { listeners[name] = handler; },
    },
  });
  vm.runInContext(source, context, {filename: workerPath});

  let install = null;
  listeners.install({waitUntil(value) { install = value; }});
  let installSettled = false;
  install.then(() => { installSettled = true; });
  await settle();
  assert.equal(skipWaitingCalls, 1);
  assert.equal(installSettled, false, 'install lifetime must own skipWaiting completion');
  skipWaitingGate.resolve();
  await install;
  assert.equal(installSettled, true);
  assert.equal(writes.length, 1, 'install is the only cache write path');
  assert.equal(writes[0].kind, 'addAll');

  let messageLifetime = null;
  listeners.message({
    data: {type: 'oracle-activate-build', build: pwa.BUILD},
    waitUntil(value) { messageLifetime = value; },
  });
  await messageLifetime;
  assert.equal(skipWaitingCalls, 2, 'exact-build recovery reaches worker takeover');
  listeners.message({
    data: {type: 'oracle-activate-build', build: 'wrong-build'},
    waitUntil() { throw new Error('wrong build must not own worker activation'); },
  });
  assert.equal(skipWaitingCalls, 2);

  const dynamicPaths = [
    '/health',
    '/chat',
    '/voice/turn',
    '/api/v1/double-agent/jobs/job-1',
    '/api/v1/memory/recall',
    '/api/v1/double-agent/conversations/private',
    '/?conversation=private',
    '/static/oracle_remote_pwa.js?v=unreviewed',
  ];
  for (const pathname of dynamicPaths) {
    let intercepted = false;
    listeners.fetch({
      request: {method: 'GET', mode: 'navigate', url: `https://oracle.private.example${pathname}`},
      respondWith() { intercepted = true; },
    });
    assert.equal(intercepted, false, `must not intercept ${pathname}`);
  }
  assert.equal(fetches.length, 0);
  assert.equal(matches.length, 0);
  assert.equal(writes.length, 1, 'dynamic requests cannot poison the cache');

  let executableResponse = null;
  const executableUrl = `https://oracle.private.example/static/oracle_remote_pwa.js?build=${encodeURIComponent(pwa.BUILD)}`;
  listeners.fetch({
    request: {method: 'GET', mode: 'no-cors', url: executableUrl},
    respondWith(value) { executableResponse = value; },
  });
  assert.equal(await executableResponse, fetchResult, 'exact-build executable is network-first');
  assert.ok(fetches.includes(executableUrl));

  let rootResponse = null;
  listeners.fetch({
    request: {method: 'GET', mode: 'navigate', url: 'https://oracle.private.example/'},
    respondWith(value) { rootResponse = value; },
  });
  assert.equal((await rootResponse).body, 'network root');
  assert.equal(writes.length, 1, 'online root navigation does not rewrite the shell');

  fetchResult = new Error('offline');
  rootResponse = null;
  listeners.fetch({
    request: {method: 'GET', mode: 'navigate', url: 'https://oracle.private.example/'},
    respondWith(value) { rootResponse = value; },
  });
  assert.equal(await rootResponse, offlineShell);
  assert.ok(matches.includes('/'));
  assert.equal(writes.length, 1);
}

async function testInsecureContextFailsClosed() {
  const state = {textContent: ''};
  const result = await pwa.registerServiceWorker(
    {isSecureContext: false, navigator: {}},
    {getElementById: id => id === 'oraclePwaState' ? state : null},
  );
  assert.equal(result.status, 'blocked_insecure_context');
  assert.match(state.textContent, /HTTPS required/);
}

async function testBootstrapBuildMismatchFailsClosed() {
  const elements = new Map([
    ['oracleBuildTruth', {textContent: ''}],
    ['oraclePwaState', {textContent: '', dataset: {}}],
    ['oracleConnectionState', {textContent: '', dataset: {}}],
    ['oracleLatencyState', {textContent: ''}],
    ['messageInput', {disabled: false}],
    ['micButton', {disabled: false}],
    ['sendButton', {disabled: false}],
    ['oracleStartHandsfree', {disabled: false}],
    ['settingsFullDuplex', {disabled: false}],
  ]);
  const windowLike = {
    __ORACLE_PWA_EXPECTED_BUILD: 'oracle-pwa-stage-a/wrong',
    navigator: {},
    location: {origin: 'https://oracle.private.example'},
    fetch: async () => { throw new Error('must not probe'); },
    addEventListener() {},
  };
  const documentLike = {
    visibilityState: 'visible',
    addEventListener() {},
    getElementById(id) { return elements.get(id) || null; },
  };
  const result = pwa.bootstrap(windowLike, documentLike);
  assert.equal(result.status, 'blocked_build_mismatch');
  assert.match(elements.get('oraclePwaState').textContent, /build mismatch/);
  for (const id of [
    'messageInput',
    'micButton',
    'sendButton',
    'oracleStartHandsfree',
    'settingsFullDuplex',
  ]) {
    assert.equal(elements.get(id).disabled, true);
  }
  assert.equal(result.connection.inspect().status, 'idle');
}

(async () => {
  await testSameOriginAndBackoff();
  await testBoundedReconnectAndLatency();
  await testConnectionRequiresExactGatewayIdentity();
  await testRestartKeepsReplacementProbeOwned();
  await testWakeLockTracksHandsFreeIntent();
  await testWakeLockDisableAndOverlapRaces();
  await testWakeLockSystemReleaseSelfHealsOnce();
  await testWakeLockImmediateRevocationAfterAcquireSelfHeals();
  await testWakeLockAlreadyReleasedSentinelSelfHeals();
  await testWakeLockPersistentRevocationUsesBoundedBackoff();
  await testWakeLockRequestErrorsBackOffAndHideCancels();
  await testServiceWorkerReadinessAndGenerationReload();
  await testServiceWorkerActivatesExpectedWaitingGenerationOnce();
  await testFailedControllerReloadDoesNotConsumeFence();
  await testServiceWorkerStrictStaticBoundary();
  await testInsecureContextFailsClosed();
  await testBootstrapBuildMismatchFailsClosed();
  console.log(JSON.stringify({
    ok: true,
    build: pwa.BUILD,
    tests: [
      'same-origin',
      'bounded-reconnect-latency',
      'gateway-identity-required',
      'restart-attempt-ownership',
      'wake-lock-lifecycle',
      'wake-lock-disable-overlap-races',
      'wake-lock-system-release-self-heal',
      'wake-lock-immediate-revocation-self-heal',
      'wake-lock-already-released-sentinel-self-heal',
      'wake-lock-persistent-revocation-bounded-backoff',
      'wake-lock-request-error-backoff-hide-cancel',
      'service-worker-readiness-generation-reload',
      'service-worker-expected-waiting-activation',
      'failed-controller-reload-retryable',
      'strict-static-service-worker-boundary',
      'insecure-context-fail-closed',
      'bootstrap-build-mismatch-fail-closed',
    ],
  }));
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
