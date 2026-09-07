/* Oracle private-remote PWA lifecycle.
 * Native browser APIs only: same-origin health, bounded reconnect, service
 * worker registration, and a wake lock while hands-free voice is active.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.OracleRemotePwa = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const VERSION = 'oracle_remote_pwa/1.0.0-candidate.5';
  const BUILD = 'oracle-pwa-stage-a/2026-08-11.v5';
  const DEFAULT_STATUS_ENDPOINT = '/api/v1/ms4_gateway/status';
  const EXPECTED_GATEWAY_IDENTITY = Object.freeze({
    service: 'ms4-gateway',
    status: 'ready',
    runtime: 'ms4-fusion',
  });
  const DEFAULT_RECONNECT_MS = Object.freeze([1000, 2000, 4000, 8000, 15000]);

  function sameOriginUrl(path, locationLike) {
    const value = String(path || '');
    const origin = String(locationLike && locationLike.origin || '');
    if (!origin || !value.startsWith('/') || value.startsWith('//')) {
      throw new Error('Oracle remote URLs must be absolute same-origin paths');
    }
    const resolved = new URL(value, origin);
    if (resolved.origin !== origin) throw new Error('Oracle remote URL crossed origin');
    return resolved.toString();
  }

  function reconnectDelay(failures, schedule) {
    const delays = schedule && schedule.length ? schedule : DEFAULT_RECONNECT_MS;
    const index = Math.max(0, Math.min(delays.length - 1, Number(failures || 1) - 1));
    return Number(delays[index]);
  }

  function validateGatewayIdentity(value, expected) {
    const identity = expected || EXPECTED_GATEWAY_IDENTITY;
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return Object.freeze({ok: false, code: 'malformed_identity', reason: 'status body is not an object'});
    }
    if (value.ok !== true) {
      return Object.freeze({ok: false, code: 'malformed_identity', reason: 'status did not report ok=true'});
    }
    for (const field of ['service', 'status', 'runtime']) {
      if (value[field] !== identity[field]) {
        return Object.freeze({
          ok: false,
          code: 'misrouted',
          reason: `expected ${field}=${identity[field]}`,
        });
      }
    }
    return Object.freeze({ok: true, identity: Object.freeze({
      service: value.service,
      status: value.status,
      runtime: value.runtime,
    })});
  }

  function buildAssetPath(path, build) {
    const value = String(path || '');
    if (!value.startsWith('/') || value.startsWith('//')) throw new Error('build asset path must be same-origin');
    return `${value}?build=${encodeURIComponent(String(build || BUILD))}`;
  }

  function workerScriptMatches(scriptURL, expectedBuild, locationLike) {
    if (!scriptURL) return false;
    const origin = String(locationLike && locationLike.origin || '');
    if (!origin) return false;
    let parsed;
    try {
      parsed = new URL(String(scriptURL), origin);
    } catch (_error) {
      return false;
    }
    const query = [...parsed.searchParams.entries()];
    return parsed.origin === origin
      && parsed.pathname === '/service-worker.js'
      && query.length === 1
      && query[0][0] === 'build'
      && query[0][1] === String(expectedBuild || BUILD);
  }

  function createConnectionMonitor(options) {
    const opts = options || {};
    const fetchFn = opts.fetchFn;
    const clock = opts.clock || {
      now: () => Date.now(),
      setTimeout: (fn, ms) => setTimeout(fn, ms),
      clearTimeout: id => clearTimeout(id),
    };
    const locationLike = opts.locationLike;
    const endpoint = opts.endpoint || DEFAULT_STATUS_ENDPOINT;
    const expectedIdentity = opts.expectedIdentity || EXPECTED_GATEWAY_IDENTITY;
    const healthyIntervalMs = Number(opts.healthyIntervalMs || 15000);
    const timeoutMs = Number(opts.timeoutMs || 5000);
    const schedule = opts.reconnectMs || DEFAULT_RECONNECT_MS;
    const onState = typeof opts.onState === 'function' ? opts.onState : function () {};
    let timer = null;
    let currentAttempt = null;
    let stopped = true;
    let generation = 0;
    let failures = 0;
    let state = Object.freeze({status: 'idle', latencyMs: null, failures: 0, nextRetryMs: null});

    function publish(next) {
      state = Object.freeze(Object.assign({}, state, next));
      onState(state);
      return state;
    }

    function clearScheduledProbe() {
      if (timer !== null) clock.clearTimeout(timer);
      timer = null;
    }

    function scheduleProbe(delayMs) {
      if (stopped) return;
      if (timer !== null) clock.clearTimeout(timer);
      timer = clock.setTimeout(function () {
        timer = null;
        void probe();
      }, delayMs);
    }

    async function probe() {
      if (stopped || currentAttempt) return state;
      const ownedGeneration = generation;
      const started = clock.now();
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      const attempt = {generation: ownedGeneration, controller, timeout: null};
      currentAttempt = attempt;
      if (controller) attempt.timeout = clock.setTimeout(() => controller.abort(), timeoutMs);
      publish({status: failures ? 'reconnecting' : 'connecting', nextRetryMs: null});
      try {
        const response = await fetchFn(sameOriginUrl(endpoint, locationLike), {
          cache: 'no-store',
          credentials: 'same-origin',
          redirect: 'error',
          signal: controller ? controller.signal : undefined,
        });
        if (!response || !response.ok) throw new Error(`status HTTP ${response && response.status}`);
        if (typeof response.json !== 'function') {
          const malformed = new Error('status response has no JSON body');
          malformed.code = 'malformed_identity';
          throw malformed;
        }
        let body;
        try {
          body = await response.json();
        } catch (_error) {
          const malformed = new Error('status response is not valid JSON');
          malformed.code = 'malformed_identity';
          throw malformed;
        }
        const identity = validateGatewayIdentity(body, expectedIdentity);
        if (!identity.ok) {
          const mismatch = new Error(identity.reason);
          mismatch.code = identity.code;
          throw mismatch;
        }
        if (stopped || ownedGeneration !== generation) return state;
        failures = 0;
        const latencyMs = Math.max(0, Math.round(clock.now() - started));
        publish({
          status: 'online',
          latencyMs,
          failures: 0,
          nextRetryMs: healthyIntervalMs,
          identity: identity.identity,
          error: null,
        });
        scheduleProbe(healthyIntervalMs);
      } catch (error) {
        if (stopped || ownedGeneration !== generation) return state;
        failures += 1;
        const delay = reconnectDelay(failures, schedule);
        const identityFailure = error && (error.code === 'misrouted' || error.code === 'malformed_identity');
        publish({
          status: identityFailure ? 'misrouted' : 'offline',
          latencyMs: null,
          failures,
          nextRetryMs: delay,
          error: String(error && error.message || error || 'connection failed'),
        });
        scheduleProbe(delay);
      } finally {
        if (attempt.timeout !== null) clock.clearTimeout(attempt.timeout);
        if (currentAttempt === attempt) currentAttempt = null;
      }
      return state;
    }

    function start() {
      if (!stopped) return state;
      stopped = false;
      generation += 1;
      void probe();
      return state;
    }

    function stop() {
      stopped = true;
      generation += 1;
      clearScheduledProbe();
      const ownedAttempt = currentAttempt;
      currentAttempt = null;
      if (ownedAttempt && ownedAttempt.timeout !== null) {
        clock.clearTimeout(ownedAttempt.timeout);
      }
      if (ownedAttempt && ownedAttempt.controller) ownedAttempt.controller.abort();
      return publish({status: 'stopped', nextRetryMs: null});
    }

    function retryNow() {
      if (stopped) stopped = false;
      if (timer !== null) clock.clearTimeout(timer);
      timer = null;
      return probe();
    }

    return Object.freeze({start, stop, probe, retryNow, inspect: () => state});
  }

  function createWakeLockController(options) {
    const opts = options || {};
    const navigatorLike = opts.navigatorLike || {};
    const documentLike = opts.documentLike || {};
    const clock = opts.clock || {
      setTimeout: (fn, ms) => setTimeout(fn, ms),
      clearTimeout: id => clearTimeout(id),
    };
    const retrySchedule = Array.isArray(opts.retryMs) && opts.retryMs.length
      ? opts.retryMs.map(value => Math.max(1, Number(value) || 1))
      : [250, 1000, 4000, 15000];
    let desired = false;
    let sentinel = null;
    let pending = null;
    let generation = 0;
    let retryTimer = null;
    let retryFailures = 0;

    function clearRetryTimer() {
      if (retryTimer !== null) clock.clearTimeout(retryTimer);
      retryTimer = null;
    }

    function canRetry(ownedGeneration) {
      return ownedGeneration === generation && desired
        && documentLike.visibilityState !== 'hidden'
        && navigatorLike.wakeLock
        && typeof navigatorLike.wakeLock.request === 'function';
    }

    function scheduleRetry(ownedGeneration) {
      if (!canRetry(ownedGeneration) || retryTimer !== null || sentinel) return false;
      retryFailures += 1;
      const delay = reconnectDelay(retryFailures, retrySchedule);
      retryTimer = clock.setTimeout(function () {
        retryTimer = null;
        if (!canRetry(ownedGeneration) || sentinel || pending) return;
        void acquire();
      }, delay);
      return true;
    }

    async function acquire() {
      if (!desired || documentLike.visibilityState === 'hidden' || sentinel
          || !navigatorLike.wakeLock || typeof navigatorLike.wakeLock.request !== 'function') return false;
      if (pending) return pending.promise;
      clearRetryTimer();
      const attempt = {generation, promise: null, retryRequired: false};
      pending = attempt;
      attempt.promise = (async () => {
        let acquired = null;
        try {
          acquired = await navigatorLike.wakeLock.request('screen');
        } catch (_error) {
          attempt.retryRequired = true;
          return false;
        }
        if (pending !== attempt || attempt.generation !== generation || !desired
            || documentLike.visibilityState === 'hidden') {
          if (acquired && typeof acquired.release === 'function') {
            try { await acquired.release(); } catch (_error) { /* already released */ }
          }
          return false;
        }
        if (!acquired) {
          attempt.retryRequired = true;
          return false;
        }
        sentinel = acquired;
        if (acquired && typeof acquired.addEventListener === 'function') {
          const handleRelease = function () {
            if (sentinel !== acquired) return;
            sentinel = null;
            if (pending === attempt) attempt.retryRequired = true;
            else scheduleRetry(attempt.generation);
          };
          acquired.addEventListener('release', handleRelease);
          if (acquired.released === true) handleRelease();
        }
        if (sentinel === acquired) retryFailures = 0;
        return sentinel === acquired;
      })().finally(() => {
        if (pending === attempt) pending = null;
        if (attempt.retryRequired) scheduleRetry(attempt.generation);
      });
      return attempt.promise;
    }

    async function release() {
      generation += 1;
      clearRetryTimer();
      retryFailures = 0;
      pending = null;
      const owned = sentinel;
      sentinel = null;
      if (owned && typeof owned.release === 'function') {
        try { await owned.release(); } catch (_error) { /* already released */ }
      }
    }

    async function setDesired(value) {
      desired = value === true;
      if (!desired) await release();
      else await acquire();
      return inspect();
    }

    function visibilityChanged() {
      if (documentLike.visibilityState === 'hidden') void release();
      else if (desired) void acquire();
    }
    if (typeof documentLike.addEventListener === 'function') {
      documentLike.addEventListener('visibilitychange', visibilityChanged);
    }

    function inspect() {
      return Object.freeze({desired, held: Boolean(sentinel), supported: Boolean(navigatorLike.wakeLock)});
    }
    return Object.freeze({setDesired, acquire, release, inspect});
  }

  function renderConnection(documentLike, state) {
    const status = documentLike.getElementById('oracleConnectionState');
    const latency = documentLike.getElementById('oracleLatencyState');
    if (status) {
      status.dataset.state = state.status;
      status.textContent = state.status === 'online' ? 'private link online'
        : state.status === 'misrouted' ? 'private link misrouted'
        : state.status === 'offline' ? `private link retry ${Math.ceil(state.nextRetryMs / 1000)}s`
        : state.status;
    }
    if (latency) latency.textContent = state.latencyMs === null ? 'latency --' : `latency ${state.latencyMs} ms`;
  }

  function inspectServiceWorkerReadiness(windowLike, registration, expectedBuild) {
    const serviceWorker = windowLike.navigator.serviceWorker;
    const locationLike = windowLike.location;
    const controller = serviceWorker.controller;
    const active = registration && registration.active;
    const controllerMatches = workerScriptMatches(controller && controller.scriptURL, expectedBuild, locationLike);
    const activeMatches = workerScriptMatches(active && active.scriptURL, expectedBuild, locationLike);
    if (controllerMatches && activeMatches && (!active.state || active.state === 'activated')) {
      return Object.freeze({status: 'controlled', controlled: true, active: true});
    }
    if ((controller && !controllerMatches) || (active && !activeMatches)) {
      return Object.freeze({status: 'generation_mismatch', controlled: controllerMatches, active: activeMatches});
    }
    if (registration && registration.installing) {
      return Object.freeze({status: 'installing', controlled: false, active: false});
    }
    if (registration && registration.waiting) {
      return Object.freeze({status: 'activating', controlled: false, active: false});
    }
    if (activeMatches) {
      return Object.freeze({status: 'registered_uncontrolled', controlled: false, active: true});
    }
    return Object.freeze({status: 'registered', controlled: false, active: false});
  }

  function renderServiceWorkerState(documentLike, readiness) {
    const state = documentLike.getElementById('oraclePwaState');
    if (!state) return;
    const labels = {
      controlled: 'PWA ready',
      installing: 'PWA installing',
      activating: 'PWA activating',
      registered_uncontrolled: 'PWA registered; reload pending',
      generation_mismatch: 'PWA blocked: generation mismatch',
      failed_install: 'PWA installation failed',
      registered: 'PWA registered',
    };
    state.dataset.state = readiness.status;
    state.textContent = labels[readiness.status] || readiness.status;
  }

  function reloadOnceForExpectedController(windowLike, expectedBuild) {
    const serviceWorker = windowLike.navigator.serviceWorker;
    const controller = serviceWorker.controller;
    if (!workerScriptMatches(controller && controller.scriptURL, expectedBuild, windowLike.location)) return false;
    const key = `oracle-pwa-controller-reload:${expectedBuild}`;
    let storage = null;
    let alreadyReloaded = windowLike.__ORACLE_PWA_CONTROLLER_RELOAD_BUILD === expectedBuild;
    try {
      storage = windowLike.sessionStorage;
      if (storage && storage.getItem(key) === '1') alreadyReloaded = true;
    } catch (_error) { /* storage can be disabled without blocking coherence */ }
    if (alreadyReloaded) return false;
    let previousGlobalFence = null;
    try {
      previousGlobalFence = windowLike.__ORACLE_PWA_CONTROLLER_RELOAD_BUILD || null;
      windowLike.__ORACLE_PWA_CONTROLLER_RELOAD_BUILD = expectedBuild;
    } catch (_error) {}
    try { if (storage) storage.setItem(key, '1'); } catch (_error) {}
    if (windowLike.location && typeof windowLike.location.reload === 'function') {
      try {
        windowLike.location.reload();
        return true;
      } catch (_error) {
        try { windowLike.__ORACLE_PWA_CONTROLLER_RELOAD_BUILD = previousGlobalFence; } catch (_inner) {}
        try { if (storage) storage.removeItem(key); } catch (_inner) {}
      }
    }
    return false;
  }

  async function registerServiceWorker(windowLike, documentLike, options) {
    const opts = options || {};
    const expectedBuild = String(opts.expectedBuild || windowLike.__ORACLE_PWA_EXPECTED_BUILD || BUILD);
    const state = documentLike.getElementById('oraclePwaState');
    if (!windowLike.isSecureContext) {
      if (state) state.textContent = 'PWA blocked: private HTTPS required';
      return {status: 'blocked_insecure_context'};
    }
    if (!windowLike.navigator.serviceWorker) {
      if (state) state.textContent = 'PWA unsupported';
      return {status: 'unsupported'};
    }
    try {
      const serviceWorker = windowLike.navigator.serviceWorker;
      const registration = await serviceWorker.register(buildAssetPath('/service-worker.js', expectedBuild), {
        scope: '/',
        updateViaCache: 'none',
      });
      let activationRequestedFor = null;
      const activateExpectedWaitingWorker = function () {
        const worker = registration.waiting;
        if (!worker || activationRequestedFor === worker
            || !workerScriptMatches(worker.scriptURL, expectedBuild, windowLike.location)
            || typeof worker.postMessage !== 'function') return false;
        activationRequestedFor = worker;
        try {
          worker.postMessage({type: 'oracle-activate-build', build: expectedBuild});
          return true;
        } catch (_error) {
          activationRequestedFor = null;
          return false;
        }
      };
      const refresh = function () {
        activateExpectedWaitingWorker();
        const readiness = inspectServiceWorkerReadiness(windowLike, registration, expectedBuild);
        renderServiceWorkerState(documentLike, readiness);
        return readiness;
      };
      const watchWorker = function (worker) {
        if (worker && typeof worker.addEventListener === 'function') {
          worker.addEventListener('statechange', function () {
            if (worker.state === 'redundant') {
              renderServiceWorkerState(documentLike, {status: 'failed_install'});
            } else {
              refresh();
            }
          });
        }
      };
      for (const worker of [registration.installing, registration.waiting, registration.active]) {
        watchWorker(worker);
      }
      if (typeof registration.addEventListener === 'function') {
        registration.addEventListener('updatefound', function () {
          watchWorker(registration.installing);
          refresh();
        });
      }
      if (typeof serviceWorker.addEventListener === 'function') {
        serviceWorker.addEventListener('controllerchange', function () {
          refresh();
          reloadOnceForExpectedController(windowLike, expectedBuild);
        });
      }
      const readiness = refresh();
      return Object.freeze(Object.assign({scope: registration.scope}, readiness));
    } catch (error) {
      if (state) state.textContent = 'PWA registration failed';
      return {status: 'failed', error: String(error && error.message || error)};
    }
  }

  function bootstrap(windowLike, documentLike) {
    const expectedBuild = String(windowLike.__ORACLE_PWA_EXPECTED_BUILD || '');
    const build = documentLike.getElementById('oracleBuildTruth');
    if (build) build.textContent = BUILD;
    const wakeLock = createWakeLockController({
      navigatorLike: windowLike.navigator,
      documentLike,
    });
    const connection = createConnectionMonitor({
      fetchFn: windowLike.fetch.bind(windowLike),
      locationLike: windowLike.location,
      onState: state => renderConnection(documentLike, state),
    });
    if (expectedBuild !== BUILD) {
      const pwaState = documentLike.getElementById('oraclePwaState');
      if (pwaState) {
        pwaState.dataset.state = 'build_mismatch';
        pwaState.textContent = 'PWA blocked: HTML/module build mismatch';
      }
      renderConnection(documentLike, {status: 'blocked', latencyMs: null});
      for (const id of [
        'messageInput',
        'micButton',
        'sendButton',
        'oracleStartHandsfree',
        'settingsFullDuplex',
      ]) {
        const control = documentLike.getElementById(id);
        if (control) control.disabled = true;
      }
      return Object.freeze({
        version: VERSION,
        build: BUILD,
        expectedBuild,
        status: 'blocked_build_mismatch',
        connection,
        wakeLock,
      });
    }
    const retry = documentLike.getElementById('oracleReconnectNow');
    if (retry) retry.addEventListener('click', () => { void connection.retryNow(); });
    windowLike.addEventListener('online', () => { void connection.retryNow(); });
    windowLike.addEventListener('offline', () => renderConnection(documentLike, {
      status: 'offline', latencyMs: null, nextRetryMs: reconnectDelay(1),
    }));
    connection.start();
    void registerServiceWorker(windowLike, documentLike, {expectedBuild});
    return Object.freeze({version: VERSION, build: BUILD, expectedBuild, connection, wakeLock});
  }

  return Object.freeze({
    VERSION,
    BUILD,
    DEFAULT_STATUS_ENDPOINT,
    EXPECTED_GATEWAY_IDENTITY,
    DEFAULT_RECONNECT_MS,
    sameOriginUrl,
    reconnectDelay,
    validateGatewayIdentity,
    buildAssetPath,
    workerScriptMatches,
    createConnectionMonitor,
    createWakeLockController,
    inspectServiceWorkerReadiness,
    renderServiceWorkerState,
    reloadOnceForExpectedController,
    registerServiceWorker,
    bootstrap,
  });
});
