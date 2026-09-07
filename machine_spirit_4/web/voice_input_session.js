/* Bounded Oracle voice-input session core.
 *
 * This dependency-free module deliberately owns no DOM or fetch behavior.
 * Browser integration injects clock, ASR/draft transport, and one onset-barge
 * hook. The same bytes therefore run under deterministic Node virtual time.
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.MS4VoiceInputSession = api;
  if (typeof globalThis !== 'undefined' && !globalThis.MS4VoiceInputSession) {
    globalThis.MS4VoiceInputSession = api;
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var VERSION = 'ms4_voice_input_session/1.0.0-candidate.1';

  function positiveInteger(value, fallback, name) {
    var selected = value == null ? fallback : Number(value);
    if (!Number.isFinite(selected) || selected <= 0) {
      throw new TypeError(name + ' must be a positive number');
    }
    return Math.max(1, Math.floor(selected));
  }

  function PcmRing(capacity) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
    this.writeIndex = 0;
    this.length = 0;
    this.totalSamples = 0;
  }

  PcmRing.prototype.append = function append(samples) {
    if (!samples || typeof samples.length !== 'number') {
      throw new TypeError('PCM samples must be array-like');
    }
    var sourceLength = Math.max(0, Math.floor(samples.length));
    var sourceStart = Math.max(0, sourceLength - this.capacity);
    for (var i = sourceStart; i < sourceLength; i += 1) {
      var sample = Number(samples[i]);
      this.buffer[this.writeIndex] = Number.isFinite(sample) ? sample : 0;
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      if (this.length < this.capacity) this.length += 1;
    }
    this.totalSamples += sourceLength;
  };

  PcmRing.prototype.snapshot = function snapshot() {
    return this.snapshotFrom(this.totalSamples - this.length);
  };

  PcmRing.prototype.snapshotFrom = function snapshotFrom(absoluteStart) {
    var oldest = this.totalSamples - this.length;
    var selectedStart = Math.max(
      oldest,
      Math.min(this.totalSamples, Math.floor(Number(absoluteStart) || 0)),
    );
    var selectedLength = this.totalSamples - selectedStart;
    var output = new Float32Array(selectedLength);
    var oldestIndex = (this.writeIndex - this.length + this.capacity) % this.capacity;
    var offset = selectedStart - oldest;
    for (var i = 0; i < selectedLength; i += 1) {
      output[i] = this.buffer[(oldestIndex + offset + i) % this.capacity];
    }
    return {
      samples: output,
      startSample: selectedStart,
      endSample: this.totalSamples,
    };
  };

  PcmRing.prototype.clear = function clear() {
    this.buffer.fill(0);
    this.writeIndex = 0;
    this.length = 0;
    this.totalSamples = 0;
  };

  function words(text) {
    var clean = String(text == null ? '' : text).trim();
    return clean ? clean.split(/\s+/) : [];
  }

  function lexical(token) {
    var lowered = String(token).toLocaleLowerCase();
    try {
      return lowered.replace(/[^\p{L}\p{N}]+/gu, '');
    } catch (_error) {
      return lowered.replace(/[^a-z0-9]+/g, '');
    }
  }

  function tokensEqual(left, right) {
    var a = lexical(left);
    var b = lexical(right);
    return a.length > 0 && a === b;
  }

  function mergeHypothesis(
    current, incoming, mutableTailWords, explicitRevision, allowOverlap
  ) {
    var prior = words(current);
    var next = words(incoming);
    if (!next.length) return prior;
    if (!prior.length) return next;

    var mutableStart = Math.max(0, prior.length - mutableTailWords);
    var maxOverlap = allowOverlap
      ? Math.min(prior.length - mutableStart, next.length)
      : 0;
    var overlap = 0;
    for (var size = maxOverlap; size > 0; size -= 1) {
      var matches = true;
      for (var offset = 0; offset < size; offset += 1) {
        if (!tokensEqual(prior[prior.length - size + offset], next[offset])) {
          matches = false;
          break;
        }
      }
      if (matches) {
        overlap = size;
        break;
      }
    }
    if (!overlap && explicitRevision) {
      var bestStart = -1;
      var bestPrefix = 0;
      for (var start = mutableStart; start < prior.length; start += 1) {
        var prefix = 0;
        while (start + prefix < prior.length && prefix < next.length
            && tokensEqual(prior[start + prefix], next[prefix])) {
          prefix += 1;
        }
        if (prefix > bestPrefix) {
          bestPrefix = prefix;
          bestStart = start;
        }
      }
      // Non-overlapping replacement is destructive, so only explicit ASR
      // revision semantics may authorize even a one-word mutable-tail anchor.
      if (bestPrefix >= 1) return prior.slice(0, bestStart).concat(next);
    }
    return prior.slice(0, prior.length - overlap).concat(next);
  }

  function createVoiceInputSession(options) {
    var opts = options || {};
    var clock = opts.clock || {
      now: function now() { return Date.now(); },
      setTimeout: function setTimer(callback, delay) { return setTimeout(callback, delay); },
      clearTimeout: function clearTimer(id) { clearTimeout(id); },
    };
    if (!clock || typeof clock.now !== 'function'
        || typeof clock.setTimeout !== 'function'
        || typeof clock.clearTimeout !== 'function') {
      throw new TypeError('clock must provide now, setTimeout, and clearTimeout');
    }

    var transport = opts.transport || {};
    if (typeof transport.transcribe !== 'function'
        || typeof transport.prepareSubmit !== 'function'
        || typeof transport.commitSubmit !== 'function') {
      throw new TypeError(
        'transport must provide transcribe, prepareSubmit, and commitSubmit',
      );
    }

    var sampleRate = positiveInteger(opts.sampleRate, 48_000, 'sampleRate');
    var maxPcmMs = positiveInteger(opts.maxPcmMs, 30_000, 'maxPcmMs');
    var overlapPcmMs = positiveInteger(
      opts.overlapPcmMs, Math.min(3_000, maxPcmMs), 'overlapPcmMs',
    );
    var partialIntervalMs = positiveInteger(
      opts.partialIntervalMs, 5_000, 'partialIntervalMs',
    );
    var finalSilenceMs = positiveInteger(opts.finalSilenceMs, 3_000, 'finalSilenceMs');
    var maxSessionMs = positiveInteger(opts.maxSessionMs, 5 * 60_000, 'maxSessionMs');
    var maxTranscriptChars = positiveInteger(
      opts.maxTranscriptChars, 64 * 1024, 'maxTranscriptChars',
    );
    var maxHypothesisChars = positiveInteger(
      opts.maxHypothesisChars, maxTranscriptChars * 4, 'maxHypothesisChars',
    );
    var mutableTailWords = positiveInteger(opts.mutableTailWords, 32, 'mutableTailWords');
    var maxPcmSamples = Math.max(1, Math.ceil(sampleRate * maxPcmMs / 1000));
    var overlapSamples = Math.min(
      maxPcmSamples,
      Math.max(1, Math.ceil(sampleRate * overlapPcmMs / 1000)),
    );
    var ring = new PcmRing(maxPcmSamples);
    var sessionId = opts.sessionId || (
      'voice-input-' + Math.floor(clock.now()) + '-' + Math.random().toString(36).slice(2)
    );
    var onBarge = typeof opts.onBarge === 'function' ? opts.onBarge : function () {};
    var onFinalizing = typeof opts.onFinalizing === 'function'
      ? opts.onFinalizing
      : function () {};
    var onError = typeof opts.onError === 'function' ? opts.onError : function () {};
    var makeAbortController = typeof opts.createAbortController === 'function'
      ? opts.createAbortController
      : function makeController() { return new AbortController(); };

    var state = 'idle';
    var startedAt = null;
    var timers = new Map();
    var transcriptText = '';
    var maxObservedPcmSamples = 0;
    var maxObservedTranscriptChars = 0;
    var lastRequestedEnd = 0;
    var lastSuccessfulEnd = 0;
    var lastAcceptedWindow = null;
    var pendingWindow = false;
    var request = null;
    var requestPromise = null;
    var finalPromise = null;
    var cancelPromise = null;
    var generation = 0;
    var finalReason = null;
    var bargeCount = 0;
    var draftCommittedChars = 0;
    var submissions = 0;
    var cancellationError = null;

    function reportError(error) {
      try { onError(error); } catch (_callbackError) {}
    }

    function observeFailure(promise) {
      if (promise && typeof promise.then === 'function') {
        Promise.resolve(promise).catch(function failureAlreadyReported() {});
      }
    }

    function clearTimer(name) {
      if (!timers.has(name)) return;
      clock.clearTimeout(timers.get(name));
      timers.delete(name);
    }

    function setNamedTimer(name, delay, callback) {
      clearTimer(name);
      var id = clock.setTimeout(function timerFired() {
        if (timers.get(name) !== id) return;
        timers.delete(name);
        callback();
      }, delay);
      timers.set(name, id);
    }

    function clearAllTimers() {
      for (var id of timers.values()) clock.clearTimeout(id);
      timers.clear();
    }

    function commitDroppedPrefix(text) {
      if (!text) return;
      if (typeof transport.appendDraft !== 'function') {
        throw new Error('transport.appendDraft is required before transcript eviction');
      }
      var ownerGeneration = generation;
      var accepted = transport.appendDraft({ sessionId: sessionId, text: text });
      if (accepted && typeof accepted.then === 'function') {
        Promise.resolve(accepted).catch(function ignoreRejectedAsyncSink() {});
        throw new Error('transport.appendDraft must transfer ownership synchronously');
      }
      if (accepted !== true) {
        throw new Error('transport.appendDraft must return true after taking ownership');
      }
      if (ownerGeneration !== generation || state === 'cancelled') {
        observeFailure(cancelPromise);
        return;
      }
      draftCommittedChars += text.length;
    }

    function boundTranscript(tokens) {
      var joined = tokens.join(' ');
      if (joined.length <= maxTranscriptChars) return joined;

      var keepFrom = 0;
      var retainedLength = joined.length;
      var stableTokenCount = Math.max(0, tokens.length - mutableTailWords);
      while (keepFrom < stableTokenCount && retainedLength > maxTranscriptChars) {
        retainedLength -= tokens[keepFrom].length;
        if (keepFrom + 1 < tokens.length) retainedLength -= 1;
        keepFrom += 1;
      }
      var dropped = tokens.slice(0, keepFrom).join(' ');
      if (dropped) commitDroppedPrefix(dropped + ' ');
      joined = tokens.slice(keepFrom).join(' ');
      if (joined.length > maxTranscriptChars) {
        throw new Error('maxTranscriptChars is too small for the mutable transcript tail');
      }
      return joined;
    }

    function acceptHypothesis(text, ownerGeneration, payload, explicitRevision) {
      if (ownerGeneration !== generation || state === 'cancelled') return;
      if (typeof text !== 'string') {
        throw new TypeError('ASR hypothesis text must be a string');
      }
      if (text.length > maxHypothesisChars) {
        throw new Error('ASR hypothesis exceeds the configured transient-work limit');
      }
      var bounded = boundTranscript(
        mergeHypothesis(
          transcriptText,
          text,
          mutableTailWords,
          explicitRevision === true,
          explicitRevision === true || !lastAcceptedWindow
            || payload.startSample < lastAcceptedWindow.endSample,
        ),
      );
      if (ownerGeneration !== generation || state === 'cancelled') return;
      transcriptText = bounded;
      lastSuccessfulEnd = Math.max(lastSuccessfulEnd, payload.endSample);
      lastAcceptedWindow = {
        startSample: payload.startSample,
        endSample: payload.endSample,
      };
      maxObservedTranscriptChars = Math.max(
        maxObservedTranscriptChars, transcriptText.length,
      );
    }

    function transcriptionPayload(final) {
      var oldest = ring.totalSamples - ring.length;
      var requestedStart = Math.max(oldest, lastSuccessfulEnd - overlapSamples);
      var window = ring.snapshotFrom(requestedStart);
      return {
        sessionId: sessionId,
        samples: window.samples,
        sampleRate: sampleRate,
        startSample: window.startSample,
        endSample: window.endSample,
        final: final === true,
      };
    }

    function beginTranscription(final) {
      var payload = transcriptionPayload(final);
      if (!final && payload.endSample <= lastRequestedEnd) return null;
      if (!final) lastRequestedEnd = payload.endSample;

      var ownerGeneration = generation;
      var expectedState = final ? 'finalizing' : 'active';
      var controller = makeAbortController();
      if (ownerGeneration !== generation || state !== expectedState) return null;
      payload.signal = controller.signal;
      request = { controller: controller, final: final === true };
      var promise = Promise.resolve()
        .then(function callTransport() {
          var result = transport.transcribe(payload);
          if (result === finalPromise || result === cancelPromise) {
            throw new Error(
              'transport.transcribe must not return an active session-operation promise',
            );
          }
          return result;
        })
        .then(function transcriptionSucceeded(result) {
          var text = result && typeof result === 'object' ? result.text : result;
          try {
            acceptHypothesis(
              text || '',
              ownerGeneration,
              payload,
              !!(result && typeof result === 'object' && result.revision === true),
            );
          } catch (error) {
            return Promise.resolve(failClosed(error)).then(function failedClosed() {
              return result;
            });
          }
          return result;
        })
        .catch(function transcriptionFailed(error) {
          if (cancellationError) throw cancellationError;
          if (ownerGeneration === generation && state !== 'cancelled'
              && !(controller.signal && controller.signal.aborted)) {
            return failClosed(error);
          }
          return null;
        })
        .finally(function transcriptionSettled() {
          if (ownerGeneration !== generation) return;
          if (request && request.controller === controller) request = null;
          if (requestPromise === promise) requestPromise = null;
          if (!final && state === 'active' && pendingWindow) {
            pendingWindow = false;
            startPartial();
          }
        });
      requestPromise = promise;
      return promise;
    }

    function startPartial() {
      if (state !== 'active') return;
      clearTimer('partial');
      if (requestPromise) {
        pendingWindow = true;
        return;
      }
      var partialPromise;
      try {
        partialPromise = beginTranscription(false);
      } catch (error) {
        observeFailure(failClosed(error));
        return;
      }
      observeFailure(partialPromise);
    }

    function schedulePartial() {
      if (state !== 'active' || timers.has('partial')) return;
      if (ring.totalSamples <= lastRequestedEnd) return;
      setNamedTimer('partial', partialIntervalMs, startPartial);
    }

    function beginSubmission(reason, ownerGeneration) {
      var controller = makeAbortController();
      if (ownerGeneration !== generation || state !== 'finalizing') {
        return cancelPromise || Promise.resolve(null);
      }
      request = { controller: controller, final: true, kind: 'submit' };
      var payload = {
        sessionId: sessionId,
        reason: reason,
        transcript: transcriptText,
        draftCommittedChars: draftCommittedChars,
        signal: controller.signal,
      };
      var promise = Promise.resolve()
        .then(function prepareSubmission() {
          var prepared = transport.prepareSubmit(payload);
          if (prepared === finalPromise) {
            throw new Error('transport.prepareSubmit must not return the active stop promise');
          }
          return prepared;
        })
        .then(function commitPreparedSubmission(prepared) {
          if (ownerGeneration !== generation || state !== 'finalizing') return null;
          // JavaScript runs this synchronous section atomically. Once commit
          // begins, cancellation is too late; reentrant cancel observes the
          // point-of-no-return state and cannot steal ownership.
          state = 'committing';
          var accepted;
          try {
            accepted = transport.commitSubmit({
              sessionId: sessionId,
              reason: reason,
              transcript: transcriptText,
              draftCommittedChars: draftCommittedChars,
              prepared: prepared,
            });
          } catch (error) {
            state = 'finalizing';
            throw error;
          }
          if (accepted !== true) {
            state = 'finalizing';
            throw new Error('transport.commitSubmit must synchronously return true');
          }
          submissions += 1;
          state = 'finalized';
          return prepared;
        })
        .finally(function submissionSettled() {
          if (ownerGeneration !== generation) return;
          if (request && request.controller === controller) request = null;
          if (requestPromise === promise) requestPromise = null;
        });
      requestPromise = promise;
      return promise;
    }

    function beginFinalize(reason) {
      if (state === 'finalized') return Promise.resolve();
      if (state === 'cancelled') return cancelPromise || Promise.resolve();
      if (finalPromise) return finalPromise;
      if (state !== 'active') return Promise.resolve();

      state = 'finalizing';
      finalReason = reason;
      pendingWindow = false;
      clearAllTimers();
      var ownerGeneration = generation;
      // This is the committed finalization edge: continuation timers have been
      // cancelled and final ASR has not started yet. Keep the notification
      // synchronous so clients can bridge that remaining ASR/model wait without
      // arming during an ordinary, still-recoverable speech pause.
      try {
        onFinalizing({ sessionId: sessionId, reason: reason });
      } catch (error) {
        reportError(error);
      }
      if (ownerGeneration !== generation || state !== 'finalizing') {
        return cancelPromise || Promise.resolve();
      }
      var priorRequest = requestPromise;
      finalPromise = Promise.resolve(priorRequest)
        .catch(function propagatePartialFailure(error) {
          if (cancellationError) throw cancellationError;
          throw error;
        })
        .then(function sendFinalWindow() {
          if (ownerGeneration !== generation || state !== 'finalizing') return null;
          return beginTranscription(true);
        })
        .then(function submitFinalTranscript() {
          if (ownerGeneration !== generation || state !== 'finalizing') return null;
          return beginSubmission(reason, ownerGeneration);
        })
        .then(function finalizeComplete() {
          if (ownerGeneration !== generation || state === 'cancelled') {
            return cancelPromise || null;
          }
          if (ownerGeneration === generation && state === 'finalizing') state = 'finalized';
          return null;
        })
        .catch(function finalizeFailed(error) {
          if (cancellationError) throw cancellationError;
          if (ownerGeneration === generation
              && state !== 'cancelled' && state !== 'finalized') {
            return failClosed(error);
          }
          return null;
        });
      return finalPromise;
    }

    function start() {
      if (state !== 'idle') return false;
      state = 'active';
      startedAt = clock.now();
      generation += 1;
      var ownerGeneration = generation;
      bargeCount += 1;
      try {
        onBarge({ sessionId: sessionId, reason: 'session_onset' });
      } catch (error) {
        observeFailure(failClosed(error));
        return false;
      }
      if (ownerGeneration !== generation || state !== 'active') {
        if (state === 'finalizing') {
          observeFailure(beginCancel(false));
        } else {
          observeFailure(cancelPromise);
        }
        return false;
      }
      setNamedTimer('maximum', maxSessionMs, function maximumReached() {
        observeFailure(beginFinalize('session_maximum'));
      });
      return true;
    }

    function appendPcm(samples) {
      if (state !== 'active') return false;
      ring.append(samples);
      maxObservedPcmSamples = Math.max(maxObservedPcmSamples, ring.length);
      if (ring.totalSamples - ring.length > lastSuccessfulEnd) {
        observeFailure(failClosed(new Error(
          'ASR backpressure overflow would evict untranscribed PCM',
        )));
        return false;
      }
      schedulePartial();
      return true;
    }

    function speechStart() {
      if (state !== 'active') return false;
      clearTimer('silence');
      return true;
    }

    function speechEnd() {
      if (state !== 'active') return false;
      setNamedTimer('silence', finalSilenceMs, function finalSilenceReached() {
        observeFailure(beginFinalize('final_silence'));
      });
      return true;
    }

    function stop() {
      return beginFinalize('explicit_stop');
    }

    function beginCancel(awaitRequest) {
      if (cancelPromise) return cancelPromise;
      if (state === 'cancelled') return Promise.resolve();
      if (state === 'finalized') return Promise.resolve();
      if (state === 'committing') return Promise.resolve();
      var settlingRequest = awaitRequest ? requestPromise : null;
      var resolveCancel;
      var rejectCancel;
      var abortError = null;
      cancelPromise = new Promise(function cancellationSettles(resolve, reject) {
        resolveCancel = resolve;
        rejectCancel = reject;
      });
      generation += 1;
      state = 'cancelled';
      clearAllTimers();
      pendingWindow = false;
      if (request && request.controller) {
        try {
          request.controller.abort();
        } catch (error) {
          abortError = error;
          settlingRequest = null;
        }
      }
      ring.clear();
      transcriptText = '';
      draftCommittedChars = 0;
      Promise.resolve(settlingRequest)
        .catch(function ignoreCancelledRequest() {})
        .then(function clearCancelledRequest() {
          request = null;
          requestPromise = null;
          if (typeof transport.cancelDraft !== 'function') return null;
          var cleanup = transport.cancelDraft({ sessionId: sessionId });
          if (cleanup === cancelPromise) {
            throw new Error(
              'transport.cancelDraft must not return the active cancel promise',
            );
          }
          if (cleanup === finalPromise) {
            throw new Error(
              'transport.cancelDraft must not return the active stop promise',
            );
          }
          return cleanup;
        })
        .then(function surfaceAbortFailure() {
          if (abortError) throw abortError;
        })
        .catch(function cancelDraftFailed(error) {
          cancellationError = error;
          reportError(error);
          throw error;
        })
        .then(resolveCancel, rejectCancel);
      return cancelPromise;
    }

    function failClosed(error) {
      if (state === 'cancelled' || state === 'finalized') return Promise.resolve();
      reportError(error);
      var reentrantFinal = finalPromise;
      var cancellation = beginCancel(false);
      if (reentrantFinal) observeFailure(reentrantFinal);
      return cancellation;
    }

    function cancel() {
      return beginCancel(true);
    }

    function inspect() {
      return {
        version: VERSION,
        sessionId: sessionId,
        state: state,
        startedAt: startedAt,
        finalReason: finalReason,
        pcmSamples: ring.length,
        maxPcmSamples: maxPcmSamples,
        maxObservedPcmSamples: maxObservedPcmSamples,
        transcriptChars: transcriptText.length,
        maxTranscriptChars: maxTranscriptChars,
        maxHypothesisChars: maxHypothesisChars,
        maxObservedTranscriptChars: maxObservedTranscriptChars,
        requestInFlight: request !== null,
        pendingWindow: pendingWindow,
        timerCount: timers.size,
        bargeCount: bargeCount,
        submissions: submissions,
        draftCommittedChars: draftCommittedChars,
        cancellationError: cancellationError ? String(cancellationError.message || cancellationError) : null,
      };
    }

    return Object.freeze({
      start: start,
      appendPcm: appendPcm,
      speechStart: speechStart,
      speechEnd: speechEnd,
      stop: stop,
      cancel: cancel,
      transcript: function transcript() { return transcriptText; },
      inspect: inspect,
    });
  }

  return Object.freeze({
    VERSION: VERSION,
    createVoiceInputSession: createVoiceInputSession,
  });
}));
