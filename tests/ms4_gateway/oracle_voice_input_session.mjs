import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createVoiceInputSession } = require('../../machine_spirit_4/web/voice_input_session.js');

class VirtualClock {
  constructor() {
    this.time = 0;
    this.nextId = 1;
    this.timers = new Map();
  }

  now = () => this.time;

  setTimeout = (callback, delay) => {
    const id = this.nextId++;
    this.timers.set(id, { at: this.time + Math.max(0, delay), callback });
    return id;
  };

  clearTimeout = (id) => {
    this.timers.delete(id);
  };

  advance(ms) {
    const target = this.time + ms;
    while (true) {
      let selected = null;
      for (const [id, timer] of this.timers) {
        if (timer.at <= target && (!selected || timer.at < selected.timer.at
            || (timer.at === selected.timer.at && id < selected.id))) {
          selected = { id, timer };
        }
      }
      if (!selected) break;
      this.time = selected.timer.at;
      this.timers.delete(selected.id);
      selected.timer.callback();
    }
    this.time = target;
  }
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

async function flush() {
  for (let turn = 0; turn < 24; turn += 1) await Promise.resolve();
}

function fixture(overrides = {}) {
  const clock = new VirtualClock();
  const calls = [];
  const prepares = [];
  const submissions = [];
  const draftAppends = [];
  const draftCancels = [];
  const barges = [];
  const finalizations = [];
  const errors = [];
  const transport = {
    transcribe: async (request) => {
      calls.push(request);
      return { text: '' };
    },
    prepareSubmit: async (request) => {
      prepares.push(request);
      return { prepared: request };
    },
    commitSubmit: (request) => {
      submissions.push(request);
      return true;
    },
    appendDraft: (request) => {
      draftAppends.push(request);
      return true;
    },
    cancelDraft: async (request) => {
      draftCancels.push(request);
    },
    ...overrides.transport,
  };
  const session = createVoiceInputSession({
    clock,
    transport,
    onBarge: (event) => barges.push(event),
    onFinalizing: (event) => finalizations.push({...event, at: clock.now()}),
    onError: (error) => errors.push(error),
    sampleRate: 1000,
    maxPcmMs: 20_000,
    partialIntervalMs: 5_000,
    finalSilenceMs: 3_000,
    maxSessionMs: 6 * 60_000,
    maxTranscriptChars: 16_384,
    mutableTailWords: 12,
    ...overrides.options,
  });
  return {
    clock, calls, prepares, submissions, draftAppends, draftCancels, barges,
    finalizations, errors, session,
  };
}

const tests = [];
function test(name, fn) {
  tests.push({ name, fn });
}

test('three minutes of speech survives 1.0-1.5 second sentence pauses', async () => {
  const f = fixture();
  f.session.start();
  for (let sentence = 0; sentence < 120; sentence += 1) {
    f.session.speechStart();
    f.session.appendPcm(new Float32Array(1_000).fill(sentence / 120));
    f.clock.advance(300);
    f.session.speechEnd();
    f.clock.advance(1_000 + (sentence % 6) * 100);
    await flush();
    assert.equal(f.submissions.length, 0, `sentence ${sentence} finalized early`);
  }
  assert.equal(f.session.inspect().state, 'active');
  assert.equal(f.barges.length, 1);
});

test('the proven 900 ms pause never finalizes but 3 seconds does exactly once', async () => {
  const f = fixture();
  f.session.start();
  f.session.speechStart();
  f.session.appendPcm(new Float32Array(500));
  f.session.speechEnd();
  f.clock.advance(900);
  await flush();
  assert.equal(f.submissions.length, 0);
  assert.equal(f.session.inspect().state, 'active');

  f.session.speechStart();
  f.session.appendPcm(new Float32Array(500));
  f.session.speechEnd();
  f.clock.advance(3_000);
  await flush();
  assert.equal(f.submissions.length, 1);
  assert.equal(f.submissions[0].reason, 'final_silence');
  f.clock.advance(30_000);
  await f.session.stop();
  assert.equal(f.submissions.length, 1);
});

test('finalization notification waits for committed VAD silence and precedes final ASR', async () => {
  const finalAsr = deferred();
  const f = fixture({
    transport: {
      transcribe: (request) => {
        f.calls.push(request);
        return finalAsr.promise;
      },
    },
  });
  f.session.start();
  f.session.speechStart();
  f.session.appendPcm(new Float32Array(500));
  f.session.speechEnd();

  // A normal continuation pause must remain recoverable and must not arm the
  // client's final-ASR feedback lifecycle.
  f.clock.advance(1_500);
  await flush();
  assert.deepEqual(f.finalizations, []);
  assert.equal(f.calls.length, 0);

  f.session.speechStart();
  f.session.appendPcm(new Float32Array(500));
  f.session.speechEnd();
  f.clock.advance(2_999);
  await flush();
  assert.deepEqual(f.finalizations, []);
  assert.equal(f.calls.length, 0);

  // The hook fires synchronously at the committed final-silence edge, before
  // the final transcription promise is even invoked/settled.
  f.clock.advance(1);
  assert.equal(f.finalizations.length, 1);
  assert.equal(f.finalizations[0].reason, 'final_silence');
  assert.equal(f.finalizations[0].at, 4_500);
  assert.equal(f.calls.length, 0);
  await flush();
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].final, true);

  finalAsr.resolve({text: 'continued thought'});
  await flush();
  assert.equal(f.submissions.length, 1);
  assert.equal(f.submissions[0].reason, 'final_silence');
  assert.equal(f.finalizations.length, 1);
});

test('explicit stop and session maximum are independent exactly-once finalizers', async () => {
  const explicit = fixture();
  explicit.session.start();
  explicit.session.appendPcm(new Float32Array(50));
  await Promise.all([explicit.session.stop(), explicit.session.stop()]);
  assert.equal(explicit.submissions.length, 1);
  assert.equal(explicit.submissions[0].reason, 'explicit_stop');

  const maximum = fixture({ options: { maxSessionMs: 10_000 } });
  maximum.session.start();
  maximum.session.speechStart();
  maximum.clock.advance(10_000);
  await flush();
  assert.equal(maximum.submissions.length, 1);
  assert.equal(maximum.submissions[0].reason, 'session_maximum');
});

test('slow partial ASR serializes requests and coalesces stale windows', async () => {
  const first = deferred();
  const f = fixture({
    transport: {
      transcribe: (request) => {
        f.calls.push(request);
        return f.calls.length === 1 ? first.promise : Promise.resolve({ text: 'latest words' });
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(1_000));
  f.clock.advance(5_000);
  await flush();
  assert.equal(f.calls.length, 1);
  const firstEnd = f.calls[0].endSample;

  for (let i = 0; i < 8; i += 1) {
    f.session.appendPcm(new Float32Array(1_000));
    f.clock.advance(5_000);
  }
  await flush();
  assert.equal(f.calls.length, 1, 'a second partial overlapped the slow request');
  assert.equal(f.session.inspect().pendingWindow, true);

  first.resolve({ text: 'older words' });
  await flush();
  assert.equal(f.calls.length, 2, 'stale windows were not coalesced to one follow-up');
  assert.ok(f.calls[1].endSample > firstEnd);
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.session.transcript(), 'older words latest words');
});

test('overlapping punctuation rewrites preserve words without duplication', async () => {
  const hypotheses = [
    'we should test this carefully',
    'test this carefully before release',
    'carefully, before release. Then document it',
    'Then document it clearly.',
  ];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return { text: hypotheses.shift() || '' };
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  for (let i = 0; i < 4; i += 1) {
    f.session.appendPcm(new Float32Array(500));
    f.clock.advance(1_000);
    await flush();
  }
  assert.equal(
    f.session.transcript(),
    'we should test this carefully, before release. Then document it clearly.',
  );
});

test('a changed mutable-tail word replaces the prior wording', async () => {
  const hypotheses = ['we should test this carefully', 'test this thoroughly'];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        const text = hypotheses.shift() || '';
        return { text, revision: text === 'test this thoroughly' };
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(500));
    f.clock.advance(1_000);
    await flush();
  }
  assert.equal(f.session.transcript(), 'we should test this thoroughly');
});

test('a one-word mutable-tail anchor supports short ASR rewrites', async () => {
  const hypotheses = ['turn left', 'turn right'];
  const f = fixture({
    transport: {
      transcribe: async () => {
        const text = hypotheses.shift() || '';
        return { text, revision: text === 'turn right' };
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }
  assert.equal(f.session.transcript(), 'turn right');
});

test('explicit revision metadata permits a one-word moving-window correction', async () => {
  const responses = [
    { text: 'please turn left' },
    { text: 'turn right', revision: true },
  ];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return responses.shift() || { text: '' };
      },
    },
    options: {
      maxPcmMs: 100,
      overlapPcmMs: 50,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }
  assert.ok(f.calls[1].startSample > f.calls[0].startSample);
  assert.equal(f.session.transcript(), 'please turn right');
});

test('a one-word continuation outside the same PCM window is not destructive', async () => {
  const hypotheses = ['we need more context', 'more details follow'];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return { text: hypotheses.shift() || '' };
      },
    },
    options: {
      maxPcmMs: 100,
      overlapPcmMs: 50,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }
  assert.ok(f.calls[1].startSample > f.calls[0].startSample);
  assert.equal(f.session.transcript(), 'we need more context more details follow');
});

test('five virtual minutes keep PCM and transcript storage bounded', async () => {
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        const firstWord = Math.floor(request.startSample / 1_000);
        const endWord = Math.ceil(request.endSample / 1_000);
        return {
          text: Array.from(
            { length: endWord - firstWord },
            (_, index) => `word${firstWord + index}`,
          ).join(' '),
        };
      },
    },
    options: {
      maxPcmMs: 4_000,
      maxTranscriptChars: 2_048,
      partialIntervalMs: 1_000,
      overlapPcmMs: 1_000,
      maxSessionMs: 301_000,
    },
  });
  f.session.start();
  for (let second = 0; second < 300; second += 1) {
    f.session.speechStart();
    f.session.appendPcm(new Float32Array(1_000));
    f.clock.advance(1_000);
    await flush();
  }
  const state = f.session.inspect();
  assert.ok(state.pcmSamples <= 4_000, state);
  assert.ok(state.maxObservedPcmSamples <= 4_000, state);
  assert.ok(state.transcriptChars <= 2_048, state);
  assert.ok(state.maxObservedTranscriptChars <= 2_048, state);
  assert.equal(f.submissions.length, 0);
  assert.ok(f.calls.every(
    (request) => request.samples.length === request.endSample - request.startSample,
  ));
  const retainedWords = (
    f.draftAppends.map((entry) => entry.text).join('') + f.session.transcript()
  ).trim().split(/\s+/);
  assert.deepEqual(
    retainedWords,
    Array.from({ length: 300 }, (_, index) => `word${index}`),
    'bounded draft handoff lost or duplicated transcript words',
  );
});

test('eviction commits only stable text and keeps the mutable tail rewriteable', async () => {
  const hypotheses = [
    'alpha beta gamma delta epsilon zeta',
    'delta epsilon theta',
  ];
  const f = fixture({
    transport: {
      transcribe: async () => {
        const text = hypotheses.shift() || '';
        return { text, revision: text === 'delta epsilon theta' };
      },
    },
    options: {
      maxTranscriptChars: 24,
      mutableTailWords: 3,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }
  assert.equal(
    f.draftAppends.map((entry) => entry.text).join('') + f.session.transcript(),
    'alpha beta gamma delta epsilon theta',
  );
  assert.ok(f.draftAppends.every((entry) => !/delta|epsilon|zeta/.test(entry.text)));
});

test('transcript eviction fails closed unless a synchronous draft sink owns it', async () => {
  const missing = fixture({
    transport: {
      appendDraft: undefined,
      transcribe: async () => ({ text: 'one two three four five six seven eight nine ten' }),
    },
    options: {
      maxTranscriptChars: 20,
      mutableTailWords: 3,
      partialIntervalMs: 1_000,
    },
  });
  missing.session.start();
  missing.session.appendPcm(new Float32Array(100));
  missing.clock.advance(1_000);
  await flush();
  assert.equal(missing.session.inspect().state, 'cancelled');
  assert.ok(missing.errors.some((error) => /appendDraft/.test(error.message)));

  const asynchronous = fixture({
    transport: {
      appendDraft: () => Promise.resolve(true),
      transcribe: async () => ({ text: 'one two three four five six seven eight nine ten' }),
    },
    options: {
      maxTranscriptChars: 20,
      mutableTailWords: 3,
      partialIntervalMs: 1_000,
    },
  });
  asynchronous.session.start();
  asynchronous.session.appendPcm(new Float32Array(100));
  asynchronous.clock.advance(1_000);
  await flush();
  assert.equal(asynchronous.session.inspect().state, 'cancelled');
  assert.ok(asynchronous.errors.some((error) => /synchronous/.test(error.message)));
});

test('PCM eviction during slow ASR fails closed instead of losing an audio gap', async () => {
  const pending = deferred();
  const f = fixture({
    transport: {
      transcribe: (request) => {
        f.calls.push(request);
        if (request.signal) {
          request.signal.addEventListener('abort', () => pending.reject(
            Object.assign(new Error('aborted'), { name: 'AbortError' }),
          ), { once: true });
        }
        return pending.promise;
      },
    },
    options: { maxPcmMs: 2_000, partialIntervalMs: 1_000 },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(1_000));
  f.clock.advance(1_000);
  await flush();
  f.session.appendPcm(new Float32Array(3_000));
  await flush();
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.calls[0].signal.aborted, true);
  assert.ok(f.errors.some((error) => /backpressure overflow/.test(error.message)));
  assert.equal(f.submissions.length, 0);
});

test('a rejected partial ASR fails closed before its PCM can be forgotten', async () => {
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('partial ASR unavailable');
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(1_000));
  f.clock.advance(1_000);
  await flush();
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.ok(f.errors.some((error) => /partial ASR unavailable/.test(error.message)));
  assert.equal(f.submissions.length, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('a rejected final ASR never submits an incomplete transcript', async () => {
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('final ASR unavailable');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  await f.session.stop();
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.ok(f.errors.some((error) => /final ASR unavailable/.test(error.message)));
  assert.equal(f.submissions.length, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('session onset barges once and same-session segments never self-cancel', async () => {
  const f = fixture({ options: { partialIntervalMs: 1_000 } });
  f.session.start();
  for (let segment = 0; segment < 10; segment += 1) {
    f.session.speechStart();
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    f.session.speechEnd();
    f.clock.advance(1_200);
    await flush();
  }
  assert.equal(f.barges.length, 1);
  assert.equal(f.draftCancels.length, 0);
  assert.equal(f.session.inspect().state, 'active');
});

test('cancel aborts request and clears every timer, PCM byte, transcript, and draft', async () => {
  const pending = deferred();
  const f = fixture({
    transport: {
      transcribe: (request) => {
        f.calls.push(request);
        request.signal.addEventListener('abort', () => pending.reject(
          Object.assign(new Error('aborted'), { name: 'AbortError' }),
        ), { once: true });
        return pending.promise;
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  f.session.speechStart();
  f.session.appendPcm(new Float32Array(2_000));
  f.clock.advance(1_000);
  await flush();
  assert.equal(f.calls.length, 1);

  await f.session.cancel();
  const state = f.session.inspect();
  assert.equal(state.state, 'cancelled');
  assert.equal(state.timerCount, 0);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.transcriptChars, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(f.calls[0].signal.aborted, true);
  assert.equal(f.draftCancels.length, 1);

  f.clock.advance(60_000);
  await flush();
  assert.equal(f.session.transcript(), '');
  assert.equal(f.submissions.length, 0);
});

test('cancel aborts and awaits an in-flight final submission', async () => {
  const pendingSubmit = deferred();
  const prepareCalls = [];
  const commitCalls = [];
  const f = fixture({
    transport: {
      prepareSubmit: (request) => {
        prepareCalls.push(request);
        request.signal.addEventListener('abort', () => pendingSubmit.reject(
          Object.assign(new Error('aborted'), { name: 'AbortError' }),
        ), { once: true });
        return pendingSubmit.promise;
      },
      commitSubmit: (request) => {
        commitCalls.push(request);
        return true;
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  const stopping = f.session.stop();
  await flush();
  assert.equal(prepareCalls.length, 1);

  await f.session.cancel();
  await stopping;
  assert.equal(prepareCalls[0].signal.aborted, true);
  assert.equal(commitCalls.length, 0);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.session.inspect().submissions, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('a rejected synchronous submit commit fails closed and cancels its draft', async () => {
  const f = fixture({
    transport: {
      commitSubmit: () => {
        throw new Error('commit rejected');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  await f.session.stop();
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().submissions, 0);
  assert.equal(f.draftCancels.length, 1);
  assert.ok(f.errors.some((error) => /commit rejected/.test(error.message)));
});

test('synchronous commit is an atomic point of no return for reentrant cancel', async () => {
  let session;
  const f = fixture({
    transport: {
      commitSubmit: () => {
        session.cancel();
        return true;
      },
    },
  });
  session = f.session;
  session.start();
  session.appendPcm(new Float32Array(100));
  await session.stop();
  assert.equal(session.inspect().state, 'finalized');
  assert.equal(session.inspect().submissions, 1);
  assert.equal(f.draftCancels.length, 0);
});

test('draft cleanup rejection is surfaced and cancellation cannot report success', async () => {
  const f = fixture({
    transport: {
      commitSubmit: () => {
        throw new Error('commit failed first');
      },
      cancelDraft: async () => {
        throw new Error('draft cleanup failed');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  await assert.rejects(f.session.stop(), /draft cleanup failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.match(f.session.inspect().cancellationError, /draft cleanup failed/);
  assert.ok(f.errors.some((error) => /draft cleanup failed/.test(error.message)));
});

test('final ASR failure cannot swallow a later draft cleanup rejection', async () => {
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('final ASR failed first');
      },
      cancelDraft: async () => {
        throw new Error('final-ASR draft cleanup failed');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  await assert.rejects(f.session.stop(), /final-ASR draft cleanup failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.match(
    f.session.inspect().cancellationError,
    /final-ASR draft cleanup failed/,
  );
});

test('oversized ASR hypotheses fail closed before tokenization or repeated slicing', async () => {
  const oversized = 'token '.repeat(3_000);
  const originalTrim = String.prototype.trim;
  const originalSlice = Array.prototype.slice;
  const originalJoin = Array.prototype.join;
  let oversizedTrimCalls = 0;
  let largeArraySliceCalls = 0;
  let largeArrayJoinCalls = 0;
  const f = fixture({
    transport: {
      transcribe: async () => ({ text: oversized }),
    },
    options: {
      maxTranscriptChars: 64,
      mutableTailWords: 3,
      partialIntervalMs: 1_000,
    },
  });

  String.prototype.trim = function countedTrim(...args) {
    if (this.length > 64) oversizedTrimCalls += 1;
    return originalTrim.apply(this, args);
  };
  Array.prototype.slice = function countedSlice(...args) {
    if (this.length > 1_000) largeArraySliceCalls += 1;
    return originalSlice.apply(this, args);
  };
  Array.prototype.join = function countedJoin(...args) {
    if (this.length > 1_000) largeArrayJoinCalls += 1;
    return originalJoin.apply(this, args);
  };
  try {
    f.session.start();
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  } finally {
    String.prototype.trim = originalTrim;
    Array.prototype.slice = originalSlice;
    Array.prototype.join = originalJoin;
  }

  assert.equal(oversizedTrimCalls, 0, 'oversized text reached trim/tokenization');
  assert.equal(largeArraySliceCalls, 0, 'oversized tokens reached repeated slice work');
  assert.equal(largeArrayJoinCalls, 0, 'oversized tokens reached repeated join work');
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().transcriptChars, 0);
  assert.equal(f.draftAppends.length, 0);
  assert.ok(f.errors.some((error) => /hypothesis.*limit/i.test(error.message)));
});

test('stop rejects when an in-flight partial fails closed and draft cleanup fails', async () => {
  const pending = deferred();
  const f = fixture({
    transport: {
      transcribe: () => pending.promise,
      cancelDraft: async () => {
        throw new Error('partial cleanup failed');
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();

  const stopping = f.session.stop();
  pending.reject(new Error('partial failed first'));
  await assert.rejects(stopping, /partial cleanup failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.match(f.session.inspect().cancellationError, /partial cleanup failed/);
  assert.equal(f.submissions.length, 0);
});

test('stop rejects when hypothesis processing fails closed and draft cleanup fails', async () => {
  const pending = deferred();
  const f = fixture({
    transport: {
      transcribe: () => pending.promise,
      cancelDraft: async () => {
        throw new Error('hypothesis cleanup failed');
      },
    },
    options: {
      maxHypothesisChars: 64,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();

  const stopping = f.session.stop();
  pending.resolve({ text: 'oversized '.repeat(100) });
  await assert.rejects(stopping, /hypothesis cleanup failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.match(f.session.inspect().cancellationError, /hypothesis cleanup failed/);
  assert.equal(f.submissions.length, 0);
});

test('autonomous partial cleanup rejection is observed without an unhandled rejection', async () => {
  const unhandled = [];
  const onUnhandled = (error) => unhandled.push(error);
  process.on('unhandledRejection', onUnhandled);
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('autonomous partial failed first');
      },
      cancelDraft: async () => {
        throw new Error('autonomous cleanup failed');
      },
    },
    options: { partialIntervalMs: 1_000 },
  });
  try {
    f.session.start();
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.removeListener('unhandledRejection', onUnhandled);
  }

  assert.deepEqual(unhandled, []);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.match(f.session.inspect().cancellationError, /autonomous cleanup failed/);
  assert.equal(f.submissions.length, 0);
});

test('a throwing onError callback cannot prevent fail-closed cancellation', async () => {
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('ASR safety failure');
      },
    },
    options: {
      partialIntervalMs: 1_000,
      onError: () => {
        throw new Error('diagnostic callback failed');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();

  const state = f.session.inspect();
  assert.equal(state.state, 'cancelled');
  assert.equal(state.timerCount, 0);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(f.draftCancels.length, 1);
  assert.equal(f.submissions.length, 0);
});

test('timer finalization cleanup rejection is observed without an unhandled rejection', async () => {
  for (const scenario of ['silence', 'maximum']) {
    const unhandled = [];
    const onUnhandled = (error) => unhandled.push(error);
    process.on('unhandledRejection', onUnhandled);
    const f = fixture({
      transport: {
        transcribe: async () => {
          throw new Error(`${scenario} final ASR failed`);
        },
        cancelDraft: async () => {
          throw new Error(`${scenario} final cleanup failed`);
        },
      },
      options: {
        finalSilenceMs: 1_000,
        maxSessionMs: scenario === 'maximum' ? 1_000 : 10_000,
      },
    });
    try {
      f.session.start();
      f.session.appendPcm(new Float32Array(100));
      if (scenario === 'silence') f.session.speechEnd();
      f.clock.advance(1_000);
      await flush();
      await new Promise((resolve) => setImmediate(resolve));
    } finally {
      process.removeListener('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], `${scenario} leaked cleanup rejection`);
    assert.equal(f.session.inspect().state, 'cancelled');
    assert.equal(f.session.inspect().requestInFlight, false);
    assert.match(
      f.session.inspect().cancellationError,
      new RegExp(`${scenario} final cleanup failed`),
    );
    assert.equal(f.submissions.length, 0);
  }
});

test('PCM overflow cleanup rejection is observed without an unhandled rejection', async () => {
  const unhandled = [];
  const onUnhandled = (error) => unhandled.push(error);
  process.on('unhandledRejection', onUnhandled);
  const f = fixture({
    transport: {
      cancelDraft: async () => {
        throw new Error('overflow cleanup failed');
      },
    },
    options: { maxPcmMs: 100 },
  });
  let appended;
  try {
    f.session.start();
    appended = f.session.appendPcm(new Float32Array(200));
    await flush();
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.removeListener('unhandledRejection', onUnhandled);
  }

  assert.equal(appended, false);
  assert.deepEqual(unhandled, []);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().timerCount, 0);
  assert.equal(f.session.inspect().pcmSamples, 0);
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.match(f.session.inspect().cancellationError, /overflow cleanup failed/);
  assert.equal(f.submissions.length, 0);
});

test('stop rejects after fail-closed cancellation has already started and cleanup fails', async () => {
  for (const scenario of ['transport', 'hypothesis']) {
    const f = fixture({
      transport: {
        transcribe: async () => {
          if (scenario === 'transport') throw new Error('partial failed first');
          return { text: 'oversized '.repeat(100) };
        },
        cancelDraft: async () => {
          throw new Error(`${scenario} post-failure cleanup failed`);
        },
      },
      options: {
        maxHypothesisChars: 64,
        partialIntervalMs: 1_000,
      },
    });
    f.session.start();
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
    await new Promise((resolve) => setImmediate(resolve));

    await assert.rejects(
      f.session.stop(),
      new RegExp(`${scenario} post-failure cleanup failed`),
    );
    assert.equal(f.session.inspect().state, 'cancelled');
    assert.equal(f.session.inspect().requestInFlight, false);
    assert.equal(f.submissions.length, 0);
  }
});

test('synchronous abort-controller construction failure fails the partial closed', async () => {
  const f = fixture({
    options: {
      partialIntervalMs: 1_000,
      createAbortController: () => {
        throw new Error('abort controller unavailable');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  assert.doesNotThrow(() => f.clock.advance(1_000));
  await flush();

  const state = f.session.inspect();
  assert.equal(state.state, 'cancelled');
  assert.equal(state.timerCount, 0);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(f.draftCancels.length, 1);
  assert.ok(f.errors.some((error) => /abort controller unavailable/.test(error.message)));
});

test('a throwing onBarge callback fails start closed without an active session', async () => {
  const f = fixture({
    options: {
      onBarge: () => {
        throw new Error('barge callback failed');
      },
    },
  });
  assert.equal(f.session.start(), false);
  await flush();

  const state = f.session.inspect();
  assert.equal(state.state, 'cancelled');
  assert.equal(state.timerCount, 0);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(f.draftCancels.length, 1);
  assert.ok(f.errors.some((error) => /barge callback failed/.test(error.message)));
});

test('discarded reentrant cancel promises cannot leak cleanup rejections', async () => {
  for (const scenario of ['appendDraft', 'onBarge']) {
    const unhandled = [];
    const onUnhandled = (error) => unhandled.push(error);
    process.on('unhandledRejection', onUnhandled);
    let session;
    const f = fixture({
      transport: {
        transcribe: async () => ({
          text: 'one two three four five six seven eight nine ten eleven twelve',
        }),
        appendDraft: () => {
          session.cancel();
          return true;
        },
        cancelDraft: async () => {
          throw new Error(`${scenario} reentrant cleanup failed`);
        },
      },
      options: {
        maxTranscriptChars: 24,
        mutableTailWords: 3,
        partialIntervalMs: 1_000,
        onBarge: scenario === 'onBarge' ? () => { session.cancel(); } : undefined,
      },
    });
    session = f.session;
    try {
      const started = session.start();
      if (scenario === 'appendDraft') {
        assert.equal(started, true);
        session.appendPcm(new Float32Array(100));
        f.clock.advance(1_000);
      } else {
        assert.equal(started, false);
      }
      await flush();
      await new Promise((resolve) => setImmediate(resolve));
    } finally {
      process.removeListener('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], `${scenario} leaked cleanup rejection`);
    assert.equal(session.inspect().state, 'cancelled');
    assert.equal(session.inspect().transcriptChars, 0);
    assert.equal(session.inspect().draftCommittedChars, 0);
    assert.equal(session.inspect().timerCount, 0);
    assert.equal(session.inspect().requestInFlight, false);
    assert.equal(session.inspect().pcmSamples, 0);
    assert.equal(session.inspect().submissions, 0);
  }
});

test('a throwing abort cannot block cancellation cleanup or settlement', async () => {
  const pending = deferred();
  const f = fixture({
    transport: {
      transcribe: () => pending.promise,
    },
    options: {
      partialIntervalMs: 1_000,
      createAbortController: () => {
        const controller = new AbortController();
        controller.abort = () => {
          throw new Error('abort operation failed');
        };
        return controller;
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();

  const outcome = await Promise.race([
    f.session.cancel().then(
      () => ({ status: 'resolved' }),
      (error) => ({ status: 'rejected', error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ status: 'timeout' }), 50)),
  ]);
  assert.equal(outcome.status, 'rejected');
  assert.match(outcome.error.message, /abort operation failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.draftCancels.length, 1);
  assert.equal(f.submissions.length, 0);
});

test('abort-controller construction reentrancy cannot start two ASR requests', async () => {
  let session;
  let stopping;
  let controllerCount = 0;
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return { text: 'one safe request' };
      },
    },
    options: {
      partialIntervalMs: 1_000,
      createAbortController: () => {
        controllerCount += 1;
        if (controllerCount === 1) stopping = session.stop();
        return new AbortController();
      },
    },
  });
  session = f.session;
  session.start();
  session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();
  await stopping;

  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].final, true);
  assert.equal(f.submissions.length, 1);
  assert.equal(session.inspect().state, 'finalized');
  assert.equal(session.inspect().requestInFlight, false);
});

test('cancelDraft returning the active cancel promise fails closed without deadlock', async () => {
  let session;
  const f = fixture({
    transport: {
      cancelDraft: () => session.cancel(),
    },
  });
  session = f.session;
  session.start();

  const outcome = await Promise.race([
    session.cancel().then(
      () => ({ status: 'resolved' }),
      (error) => ({ status: 'rejected', error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ status: 'timeout' }), 50)),
  ]);
  assert.equal(outcome.status, 'rejected');
  assert.match(outcome.error.message, /cancelDraft.*active cancel promise/i);
  assert.equal(session.inspect().state, 'cancelled');
  assert.equal(session.inspect().requestInFlight, false);
  assert.equal(session.inspect().submissions, 0);
});

test('prepareSubmit returning the active stop promise fails closed without deadlock', async () => {
  let stopping;
  const f = fixture({
    transport: {
      prepareSubmit: () => stopping,
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  stopping = f.session.stop();

  const outcome = await Promise.race([
    stopping.then(
      () => ({ status: 'resolved' }),
      (error) => ({ status: 'rejected', error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ status: 'timeout' }), 50)),
  ]);
  assert.notEqual(outcome.status, 'timeout');
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.session.inspect().submissions, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('submission controller cancellation prevents stale prepare work', async () => {
  let session;
  let controllerCount = 0;
  const f = fixture({
    transport: {
      cancelDraft: async () => {
        f.draftCancels.push({ sessionId: session.inspect().sessionId });
        throw new Error('submission reentrant cleanup failed');
      },
    },
    options: {
      createAbortController: () => {
        controllerCount += 1;
        if (controllerCount === 2) session.cancel();
        return new AbortController();
      },
    },
  });
  session = f.session;
  session.start();
  session.appendPcm(new Float32Array(100));

  await assert.rejects(session.stop(), /submission reentrant cleanup failed/);
  assert.equal(f.prepares.length, 0);
  assert.equal(f.submissions.length, 0);
  assert.equal(f.draftCancels.length, 1);
  assert.equal(session.inspect().state, 'cancelled');
  assert.equal(session.inspect().requestInFlight, false);
});

test('onBarge reentrant stop cannot submit while start reports failure', async () => {
  let session;
  let stopping;
  const f = fixture({
    options: {
      onBarge: () => {
        stopping = session.stop();
      },
    },
  });
  session = f.session;

  assert.equal(session.start(), false);
  await stopping;
  assert.equal(f.prepares.length, 0);
  assert.equal(f.submissions.length, 0);
  assert.equal(session.inspect().state, 'cancelled');
  assert.equal(session.inspect().requestInFlight, false);
  assert.equal(f.draftCancels.length, 1);
});

test('onError reentrant stop cannot leak a later cleanup rejection', async () => {
  const unhandled = [];
  const onUnhandled = (error) => unhandled.push(error);
  process.on('unhandledRejection', onUnhandled);
  let session;
  const f = fixture({
    transport: {
      transcribe: async () => {
        throw new Error('partial failed before onError');
      },
      cancelDraft: async () => {
        throw new Error('onError reentrant cleanup failed');
      },
    },
    options: {
      partialIntervalMs: 1_000,
      onError: () => {
        session.stop();
      },
    },
  });
  session = f.session;
  try {
    session.start();
    session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.removeListener('unhandledRejection', onUnhandled);
  }

  assert.deepEqual(unhandled, []);
  assert.equal(session.inspect().state, 'cancelled');
  assert.equal(session.inspect().requestInFlight, false);
  assert.match(session.inspect().cancellationError, /onError reentrant cleanup failed/);
  assert.equal(f.submissions.length, 0);
});

test('transcribe returning the active stop promise fails closed without deadlock', async () => {
  for (const scenario of ['partial', 'final']) {
    let session;
    let stopping;
    const f = fixture({
      transport: {
        transcribe: () => {
          if (scenario === 'partial' && !stopping) stopping = session.stop();
          return stopping;
        },
      },
      options: { partialIntervalMs: 1_000 },
    });
    session = f.session;
    session.start();
    session.appendPcm(new Float32Array(100));
    if (scenario === 'partial') {
      f.clock.advance(1_000);
      await flush();
    } else {
      stopping = session.stop();
    }
    assert.ok(stopping, `${scenario} did not create a stop promise`);

    const outcome = await Promise.race([
      stopping.then(
        () => ({ status: 'resolved' }),
        (error) => ({ status: 'rejected', error }),
      ),
      new Promise((resolve) => setTimeout(() => resolve({ status: 'timeout' }), 50)),
    ]);
    assert.notEqual(outcome.status, 'timeout', `${scenario} transcribe deadlocked`);
    assert.equal(session.inspect().state, 'cancelled');
    assert.equal(session.inspect().requestInFlight, false);
    assert.equal(session.inspect().submissions, 0);
    assert.equal(f.draftCancels.length, 1);
  }
});

test('stop joins rejected cleanup after stale preparation settles', async () => {
  const pendingPrepare = deferred();
  const f = fixture({
    transport: {
      prepareSubmit: (request) => {
        f.prepares.push(request);
        return pendingPrepare.promise;
      },
      cancelDraft: async () => {
        throw new Error('stale preparation cleanup failed');
      },
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  const stopping = f.session.stop();
  await flush();
  assert.equal(f.prepares.length, 1);

  const cancelling = f.session.cancel();
  pendingPrepare.resolve({ stale: true });
  const [stopResult, cancelResult] = await Promise.allSettled([stopping, cancelling]);
  assert.equal(stopResult.status, 'rejected');
  assert.match(stopResult.reason.message, /stale preparation cleanup failed/);
  assert.equal(cancelResult.status, 'rejected');
  assert.match(cancelResult.reason.message, /stale preparation cleanup failed/);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.session.inspect().submissions, 0);
});

test('disjoint PCM windows preserve a legitimate repeated suffix and prefix', async () => {
  const responses = [
    { text: 'intro alpha beta' },
    { text: 'alpha beta new tail' },
  ];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return responses.shift() || { text: '' };
      },
    },
    options: {
      maxPcmMs: 100,
      overlapPcmMs: 50,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }

  assert.deepEqual(
    f.calls.map(({ startSample, endSample }) => [startSample, endSample]),
    [[0, 100], [100, 200]],
  );
  assert.equal(
    f.session.transcript(),
    'intro alpha beta alpha beta new tail',
  );
});

test('cancelDraft returning the active stop promise after commit failure cannot deadlock', async () => {
  let stopping;
  const f = fixture({
    transport: {
      commitSubmit: () => {
        throw new Error('commit failed before cleanup cycle');
      },
      cancelDraft: () => stopping,
    },
  });
  f.session.start();
  f.session.appendPcm(new Float32Array(100));
  stopping = f.session.stop();

  const outcome = await Promise.race([
    stopping.then(
      () => ({ status: 'resolved' }),
      (error) => ({ status: 'rejected', error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ status: 'timeout' }), 100)),
  ]);
  assert.equal(outcome.status, 'rejected');
  assert.match(outcome.error.message, /cancelDraft.*active.*promise/i);
  assert.equal(f.session.inspect().state, 'cancelled');
  assert.equal(f.session.inspect().timerCount, 0);
  assert.equal(f.session.inspect().pcmSamples, 0);
  assert.equal(f.session.inspect().requestInFlight, false);
  assert.equal(f.session.inspect().submissions, 0);
});

test('ambiguous repeated anchors preserve valid text without explicit revision metadata', async () => {
  const responses = [
    { text: 'alpha beta keep this alpha beta ending' },
    { text: 'alpha beta corrected' },
  ];
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        return responses.shift() || { text: '' };
      },
    },
    options: {
      maxPcmMs: 100,
      overlapPcmMs: 50,
      partialIntervalMs: 1_000,
    },
  });
  f.session.start();
  for (let i = 0; i < 2; i += 1) {
    f.session.appendPcm(new Float32Array(100));
    f.clock.advance(1_000);
    await flush();
  }

  assert.ok(f.calls[1].startSample > f.calls[0].startSample);
  assert.equal(
    f.session.transcript(),
    'alpha beta keep this alpha beta ending alpha beta corrected',
  );
});

test('synchronous cancel from appendDraft leaves no session-owned residue or success', async () => {
  let session;
  let cancellation;
  const draftAppends = [];
  const f = fixture({
    transport: {
      transcribe: async () => ({
        text: 'one two three four five six seven eight nine ten eleven twelve',
      }),
      appendDraft: (request) => {
        draftAppends.push(request);
        cancellation = session.cancel();
        return true;
      },
    },
    options: {
      maxTranscriptChars: 24,
      mutableTailWords: 3,
      partialIntervalMs: 1_000,
    },
  });
  session = f.session;
  assert.equal(session.start(), true);
  session.appendPcm(new Float32Array(100));
  f.clock.advance(1_000);
  await flush();
  await cancellation;

  const state = session.inspect();
  assert.equal(draftAppends.length, 1);
  assert.equal(state.state, 'cancelled');
  assert.equal(state.transcriptChars, 0);
  assert.equal(state.draftCommittedChars, 0);
  assert.equal(state.timerCount, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.submissions, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('synchronous cancel from onBarge makes start fail with no session-owned residue', async () => {
  let session;
  let cancellation;
  const f = fixture({
    options: {
      onBarge: () => {
        cancellation = session.cancel();
      },
    },
  });
  session = f.session;

  assert.equal(session.start(), false);
  await cancellation;
  const state = session.inspect();
  assert.equal(state.state, 'cancelled');
  assert.equal(state.transcriptChars, 0);
  assert.equal(state.draftCommittedChars, 0);
  assert.equal(state.timerCount, 0);
  assert.equal(state.requestInFlight, false);
  assert.equal(state.pcmSamples, 0);
  assert.equal(state.submissions, 0);
  assert.equal(f.draftCancels.length, 1);
});

test('multi-minute final submission retains evicted draft and mutable tail exactly once', async () => {
  const f = fixture({
    transport: {
      transcribe: async (request) => {
        f.calls.push(request);
        const firstWord = Math.floor(request.startSample / 1_000);
        const endWord = Math.ceil(request.endSample / 1_000);
        return {
          text: Array.from(
            { length: endWord - firstWord },
            (_, index) => `word${firstWord + index}`,
          ).join(' '),
        };
      },
    },
    options: {
      maxPcmMs: 4_000,
      maxTranscriptChars: 256,
      partialIntervalMs: 1_000,
      overlapPcmMs: 1_000,
      maxSessionMs: 181_000,
    },
  });
  f.session.start();
  for (let second = 0; second < 180; second += 1) {
    f.session.speechStart();
    f.session.appendPcm(new Float32Array(1_000));
    f.clock.advance(1_000);
    await flush();
  }
  await f.session.stop();

  assert.equal(f.submissions.length, 1);
  assert.ok(f.draftAppends.length > 0, 'test did not exercise draft eviction');
  const committedDraft = f.draftAppends.map((entry) => entry.text).join('');
  assert.equal(f.submissions[0].draftCommittedChars, committedDraft.length);
  assert.equal(f.submissions[0].transcript, f.session.transcript());
  assert.deepEqual(
    (committedDraft + f.submissions[0].transcript).trim().split(/\s+/),
    Array.from({ length: 180 }, (_, index) => `word${index}`),
  );
  assert.equal(f.session.inspect().state, 'finalized');
});

const selectedPattern = process.argv[2];
const selectedTests = selectedPattern
  ? tests.filter(({ name }) => name.includes(selectedPattern))
  : tests;
if (selectedPattern && selectedTests.length === 0) {
  throw new Error(`no tests matched ${JSON.stringify(selectedPattern)}`);
}

let passed = 0;
for (const { name, fn } of selectedTests) {
  try {
    await fn();
    passed += 1;
    console.log(`ok ${passed} - ${name}`);
  } catch (error) {
    console.error(`not ok ${passed + 1} - ${name}`);
    console.error(error && error.stack ? error.stack : error);
    process.exitCode = 1;
    break;
  }
}

if (!process.exitCode) {
  console.log(JSON.stringify({ ok: true, tests: passed }));
}
