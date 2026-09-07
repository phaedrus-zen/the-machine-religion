/* ms4_voice_dsp.js -- Oracle voice DSP decision helpers (R3 candidate).
 *
 * CANDIDATE / PROPOSED source for lane `oracle-audio-aec-vad-cadence-fix-r3`.
 * R3 retains the reviewed R2 mechanisms and makes adaptive echo confidence
 * episode-, cadence-, observability-, clipping-, and inactivity-aware.
 * This file, by itself, is NOT physical proof of anything.
 *
 * WHY A SEPARATE FILE
 *   The VAD/AEC/anti-fragment/cadence DECISIONS were previously fixed constants
 *   inlined in index.html (vadSpeechThreshold()==0.025, no playback-aware echo
 *   gate, no anti-fragment submit gate). Those decisions are pure functions of
 *   scalar inputs, so they belong in one dependency-free module that:
 *     - runs in the browser as a classic <script> (installs window.MS4DSP), and
 *     - runs in Node (module.exports) so the EXACT shipped decision bytes are
 *       unit-tested against recorded physical-room RMS traces + synthetic
 *       fixtures, deterministically, with no DOM.
 *   index.html's vadTick STATE MACHINE (silence/maybe_speech/speech/
 *   maybe_silence transitions) stays inline (it is contract-checked by
 *   scripts/validation/ms4_oracle_contract_check.py and behavior-proven by the
 *   owner's real-browser harness); it only DELEGATES its threshold / echo /
 *   submit decisions here.
 *
 * EVIDENCE-DERIVED DEFAULTS (recorded R2 run
 *   oracle_physical_front_back_exec_r2_20260708T1551Z):
 *     - static threshold 0.025 sat ABOVE much room speech (far-field clip).
 *     - t10_barge_3 real barge peak RMS 0.0238 < 0.025  -> MISSED (2/3 barge).
 *     - t07_silence max RMS 4e-5                          -> must stay rejected.
 *     - t11_aec_selftrigger echo RMS 0.08155 fired onset WHILE OPERATOR SILENT
 *       during prior-turn playback                        -> must be suppressed.
 *     - t08/t09 real double-talk barge RMS 0.124/0.166    -> must still fire.
 *   The onset threshold is therefore made noise-floor adaptive, BOUNDED BELOW by
 *   an absolute floor (0.010) that is safely above silence but below 0.0238, and
 *   BOUNDED ABOVE by the operator's configured threshold (so a quiet far-field
 *   voice below 0.025 is no longer clipped). Self-echo is rejected by a dynamic
 *   playback-referenced gate; real double-talk above the projected echo still
 *   fires. Final coupling/double-talk tuning REQUIRES the physical rerun -- unit
 *   tests here prove the MECHANISM and its direction, not a product pass.
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;            // Node / test harness
  }
  if (typeof window !== 'undefined') {
    window.MS4DSP = api;             // browser classic <script>
  }
  // Undeclared-safe global fallback (workers, etc.)
  if (typeof globalThis !== 'undefined' && !globalThis.MS4DSP) {
    globalThis.MS4DSP = api;
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var VERSION = 'ms4_voice_dsp/3.0.0-candidate-r3';

  // ---- numeric guards -----------------------------------------------------
  function isFiniteNum(x) { return typeof x === 'number' && isFinite(x); }
  function clamp(x, lo, hi) {
    if (!isFiniteNum(x)) return lo;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  // ---- defaults (all overridable; browser reads localStorage in index.html) --
  var VAD_DEFAULTS = Object.freeze({
    // Onset threshold shaping.
    floorMin: 0.010,          // absolute min onset threshold (> silence, < 0.0238)
    noiseMult: 2.5,           // onset >= noiseFloor * noiseMult ...
    noiseMargin: 0.004,       // ... + additive headroom
    hysteresisRatio: 0.6,     // silence threshold = onset * ratio (matches legacy)
    // Noise-floor tracker.
    floorAlpha: 0.05,         // slow EMA
    noiseFloorMin: 0.0002,
    noiseFloorMax: 0.020,     // capped so a noisy room can't clip speech entirely
    // Playback-referenced self-echo gate.
    echoCoupling: 0.12,       // fraction of Oracle output level leaking to mic
    echoDoubleTalkMargin: 1.4,// real speech must exceed projected echo by this
    echoCouplingMin: 0.02,    // bounded adaptive playback->mic estimate
    echoCouplingMax: 0.45,
    echoCouplingAlpha: 0.18,
    echoCalibrationFrames: 6,
    echoCalibrationMs: 180,
    echoCalibrationMaxGapMs: 90,
    echoPlaybackMinLevel: 0.01,
    echoObservableFloor: 0.004,
    echoObservableNoiseMult: 3.0,
    echoStabilityFloor: 0.003,
    echoStabilityRatio: 0.12,
    echoClipRms: 0.95,
    echoConfidenceInactivityMs: 900,
    echoPathChangeResidual: 0.030,
    echoReferenceAttack: 0.72,
    echoReferenceRelease: 0.08,
    echoTailMs: 600,          // hold delayed acoustic energy after playback ends
    echoTailDecayPer30Ms: 0.94,
    echoRecalibrationHoldMs: 360,
    echoResidualFloor: 0.006,
    echoResidualOnsetMult: 0.75,
    echoUnknownClearFloor: 0.10,
    echoUnknownOnsetMult: 3.0,
    echoUnknownResidualFloor: 0.025,
    // Anti-fragment submit gate.
    minUtteranceMs: 320,      // reject sub-320ms "utterances" (echo fragments)
    submitPeakMult: 1.2,      // peak must exceed onsetThreshold * this to submit
    // State-machine timing (mirror index.html legacy defaults for replay parity).
    emaAlpha: 0.35,
    minSpeechMs: 160,
    hangoverMs: 1400,
  });

  /**
   * Slowly track the ambient noise floor from quiet frames. The caller MUST
   * freeze updates (pass update:false) whenever Oracle playback is active or the
   * VAD is inside a speech/maybe_silence phase, otherwise self-echo or the
   * user's own voice would inflate the floor and defeat far-field detection.
   */
  function updateNoiseFloor(prevFloor, rms, opts) {
    var o = opts || {};
    var alpha = isFiniteNum(o.floorAlpha) ? o.floorAlpha : VAD_DEFAULTS.floorAlpha;
    var lo = isFiniteNum(o.noiseFloorMin) ? o.noiseFloorMin : VAD_DEFAULTS.noiseFloorMin;
    var hi = isFiniteNum(o.noiseFloorMax) ? o.noiseFloorMax : VAD_DEFAULTS.noiseFloorMax;
    var prev = isFiniteNum(prevFloor) ? prevFloor : lo;
    if (o.update === false) return clamp(prev, lo, hi);
    if (!isFiniteNum(rms)) return clamp(prev, lo, hi);
    var next = (1 - alpha) * prev + alpha * rms;
    return clamp(next, lo, hi);
  }

  /**
   * Bounded, noise-floor-aware onset + silence thresholds.
   *   onset   = clamp(noiseFloor*mult + margin, floorMin, userThreshold)
   *   silence = onset * hysteresisRatio
   * The user's configured threshold is a CEILING (never clip a quiet far-field
   * voice below it); floorMin is the hard floor (never trigger on room hiss).
   */
  function effectiveOnsetThreshold(userThreshold, noiseFloor, opts) {
    var o = opts || {};
    var floorMin = isFiniteNum(o.floorMin) ? o.floorMin : VAD_DEFAULTS.floorMin;
    var mult = isFiniteNum(o.noiseMult) ? o.noiseMult : VAD_DEFAULTS.noiseMult;
    var margin = isFiniteNum(o.noiseMargin) ? o.noiseMargin : VAD_DEFAULTS.noiseMargin;
    var ratio = isFiniteNum(o.hysteresisRatio) ? o.hysteresisRatio : VAD_DEFAULTS.hysteresisRatio;
    var ceiling = isFiniteNum(userThreshold) && userThreshold > 0 ? userThreshold : 0.025;
    // If the room is so noisy that floorMin would exceed the user ceiling, the
    // ceiling still wins (never clip): the effective onset can never exceed it.
    var adaptive = (isFiniteNum(noiseFloor) ? noiseFloor : 0) * mult + margin;
    var lowerBound = Math.min(floorMin, ceiling);
    var onset = clamp(adaptive, lowerBound, ceiling);
    return { onset: onset, silence: onset * ratio };
  }

  /**
   * Adaptive playback-to-mic reference. Coupling is learned only from frames
   * that are still classified as silence while playback is active. The estimate
   * is bounded so a bad observation cannot permanently blind VAD, and the
   * observed envelope is held briefly after playback to cover acoustic/device
   * delay. Calibration is deliberately explicit: callers must treat an unknown
   * estimate conservatively at the speech-promotion boundary.
   */
  function createEchoReferenceState(opts) {
    var o = opts || {};
    var seed = isFiniteNum(o.echoCoupling) ? o.echoCoupling : VAD_DEFAULTS.echoCoupling;
    return {
      couplingEstimate: clamp(seed, VAD_DEFAULTS.echoCouplingMin, VAD_DEFAULTS.echoCouplingMax),
      calibrationFrames: 0,
      calibrationElapsedMs: 0,
      calibrationStartedAt: null,
      calibrated: false,
      calibratedAt: null,
      referenceRms: 0,
      lastPlaybackAt: 0,
      playbackWasActive: false,
      playbackEpisode: 0,
      playbackEpisodeStartedAt: null,
      episodeMinCoupling: null,
      lastObservedEcho: null,
      weakResidualActive: false,
      lastQualifiedAt: null,
      tailUntil: 0,
      tailActive: false,
      confidenceHoldUntil: 0,
      lastUpdateAt: null,
    };
  }

  function updateEchoReference(previous, args, opts) {
    var a = args || {};
    var o = Object.assign({}, VAD_DEFAULTS, opts || {});
    var p = previous || createEchoReferenceState(o);
    var now = isFiniteNum(a.now) ? a.now : 0;
    var lastUpdate = isFiniteNum(p.lastUpdateAt) ? p.lastUpdateAt : now;
    var elapsed = Math.max(0, now - lastUpdate);
    var rms = isFiniteNum(a.smoothedRms) ? Math.max(0, a.smoothedRms) : 0;
    var noise = isFiniteNum(a.noiseFloor) ? Math.max(0, a.noiseFloor) : 0;
    var level = isFiniteNum(a.playbackLevel) ? Math.max(0, a.playbackLevel) : 0;
    var active = !!a.playbackActive;
    var phase = typeof a.phase === 'string' ? a.phase : 'silence';
    var coupling = isFiniteNum(p.couplingEstimate) ? p.couplingEstimate : o.echoCoupling;
    var frames = isFiniteNum(p.calibrationFrames) ? Math.max(0, p.calibrationFrames | 0) : 0;
    var calibrationStartedAt = isFiniteNum(p.calibrationStartedAt) ? p.calibrationStartedAt : null;
    var calibrationElapsedMs = isFiniteNum(p.calibrationElapsedMs) ? Math.max(0, p.calibrationElapsedMs) : 0;
    var calibrated = !!p.calibrated;
    var calibratedAt = isFiniteNum(p.calibratedAt) ? p.calibratedAt : null;
    var reference = isFiniteNum(p.referenceRms) ? Math.max(0, p.referenceRms) : 0;
    var lastPlaybackAt = isFiniteNum(p.lastPlaybackAt) ? p.lastPlaybackAt : 0;
    var playbackWasActive = !!p.playbackWasActive;
    var playbackEpisode = isFiniteNum(p.playbackEpisode) ? Math.max(0, p.playbackEpisode | 0) : 0;
    var playbackEpisodeStartedAt = isFiniteNum(p.playbackEpisodeStartedAt) ? p.playbackEpisodeStartedAt : null;
    var episodeMinCoupling = isFiniteNum(p.episodeMinCoupling) ? p.episodeMinCoupling : null;
    var lastObservedEcho = isFiniteNum(p.lastObservedEcho) ? p.lastObservedEcho : null;
    var weakResidualActive = !!p.weakResidualActive;
    var lastQualifiedAt = isFiniteNum(p.lastQualifiedAt) ? p.lastQualifiedAt : null;
    var tailUntil = isFiniteNum(p.tailUntil) ? p.tailUntil : 0;
    var confidenceHoldUntil = isFiniteNum(p.confidenceHoldUntil) ? p.confidenceHoldUntil : 0;

    if (active) {
      if (!playbackWasActive) {
        // Confidence is episode-local. A later playback start must earn new
        // evidence rather than inheriting a stale room/device observation.
        playbackEpisode += 1;
        playbackEpisodeStartedAt = now;
        frames = 0;
        calibrationStartedAt = null;
        calibrationElapsedMs = 0;
        calibrated = false;
        calibratedAt = null;
        episodeMinCoupling = null;
        lastObservedEcho = null;
        weakResidualActive = false;
        lastQualifiedAt = null;
        reference = 0;
      }
      lastPlaybackAt = now;
      tailUntil = now + o.echoTailMs;
      var observedEcho = Math.max(0, rms - noise);
      var observableFloor = Math.max(o.echoObservableFloor, noise * o.echoObservableNoiseMult);
      var clipped = !!a.clipped || rms >= o.echoClipRms;
      var qualified = (phase === 'silence' || phase === 'maybe_speech')
        && level >= o.echoPlaybackMinLevel
        && observedEcho >= observableFloor
        && !clipped;
      var ambiguousNearEnd = false;
      var pathChange = false;
      // Sudden near-end energy the FROZEN (pre-update) echo estimate cannot
      // explain. Classified from the current frame against the pre-update
      // coupling/reference so a burst can never first teach the estimate that
      // would then suppress it. See the nearEndIntrusion gate below.
      var nearEndIntrusion = false;

      if (clipped) {
        // Overload is not acoustic evidence. It invalidates confidence and
        // cannot advance either frame or elapsed-time requirements.
        frames = 0;
        calibrationStartedAt = null;
        calibrationElapsedMs = 0;
        calibrated = false;
        calibratedAt = null;
        episodeMinCoupling = null;
        lastObservedEcho = null;
        weakResidualActive = false;
        lastQualifiedAt = null;
        confidenceHoldUntil = now + o.echoRecalibrationHoldMs;
      } else if (qualified) {
        var observedCoupling = clamp(observedEcho / level, o.echoCouplingMin, o.echoCouplingMax);
        var projectedBeforeUpdate = level * coupling;
        var positiveResidual = observedEcho - projectedBeforeUpdate;
        var stabilityLimit = Math.max(
          o.echoStabilityFloor,
          (lastObservedEcho === null ? 0 : lastObservedEcho * o.echoStabilityRatio)
        );
        var observationStable = lastObservedEcho === null
          || Math.abs(observedEcho - lastObservedEcho) <= stabilityLimit;
        var evidenceMature = calibrated || (
          frames >= o.echoCalibrationFrames
          && calibrationElapsedMs >= Math.max(0, o.echoCalibrationMs - o.echoCalibrationMaxGapMs)
        );
        var couplingRise = episodeMinCoupling !== null
          && observedCoupling > episodeMinCoupling
            + Math.max(0.02, episodeMinCoupling * 0.25);
        var residualClassifiable = evidenceMature || weakResidualActive;
        ambiguousNearEnd = residualClassifiable && couplingRise
          && positiveResidual >= o.echoResidualFloor
          && positiveResidual <= o.echoPathChangeResidual;
        pathChange = residualClassifiable
          && rms < o.echoUnknownClearFloor
          && positiveResidual > o.echoPathChangeResidual;
        // A large positive residual against the frozen coupling whose TOTAL
        // energy is already in the unmistakable near-end band (rms at or above
        // the unknown clear floor) is a sudden near-end burst, not echo. Unlike
        // a quiet path change (rms below the clear floor) it is classifiable as
        // near-end WITHOUT waiting for mature evidence, so it must be recognized
        // BEFORE any adaptive update. It never teaches coupling/reference/
        // confidence/calibration; the promotion guard then judges it against the
        // frozen (genuine-echo) estimate instead of one the burst inflated.
        nearEndIntrusion = positiveResidual > o.echoPathChangeResidual
          && rms >= o.echoUnknownClearFloor;

        if (!nearEndIntrusion && !observationStable && !residualClassifiable) {
          // Browser RMS is an EMA. Its attack ramp is not a sequence of stable
          // acoustic observations, so restart the evidence window at the latest
          // ratio instead of teaching the ramp's low early values as coupling.
          frames = 1;
          calibrationStartedAt = now;
          calibrationElapsedMs = 0;
          calibrated = false;
          calibratedAt = null;
          episodeMinCoupling = observedCoupling;
          lastQualifiedAt = now;
          coupling = observedCoupling;
        }

        if (pathChange) {
          // A large path change is allowed to recalibrate, but only after a new
          // continuous evidence window. A smaller positive residual is treated
          // as weak double-talk and is never learned into the echo estimate.
          frames = 0;
          calibrationStartedAt = null;
          calibrationElapsedMs = 0;
          calibrated = false;
          calibratedAt = null;
          episodeMinCoupling = null;
          lastQualifiedAt = null;
          coupling = observedCoupling;
          weakResidualActive = false;
          confidenceHoldUntil = now + o.echoRecalibrationHoldMs;
        }

        if (ambiguousNearEnd && rms < o.echoUnknownClearFloor) {
          // Weak positive residual is possible near-end speech. Drop confidence
          // but retain the lower coupling baseline, and keep extending the hold
          // while that residual persists so it can never become the new echo.
          frames = 0;
          calibrationStartedAt = null;
          calibrationElapsedMs = 0;
          calibrated = false;
          calibratedAt = null;
          lastQualifiedAt = null;
          weakResidualActive = true;
          confidenceHoldUntil = now + o.echoRecalibrationHoldMs;
        }

        if (!nearEndIntrusion && !ambiguousNearEnd && (observationStable || pathChange)) {
          var gap = lastQualifiedAt === null ? null : now - lastQualifiedAt;
          if (gap !== null && (gap <= 0 || gap > o.echoCalibrationMaxGapMs)) {
            if (gap > o.echoCalibrationMaxGapMs) {
              frames = 0;
              calibrationStartedAt = null;
              calibrationElapsedMs = 0;
              episodeMinCoupling = null;
            }
          }
          if (lastQualifiedAt === null || gap > 0) {
            if (calibrationStartedAt === null) calibrationStartedAt = now;
            frames += 1;
            lastQualifiedAt = now;
            calibrationElapsedMs = Math.max(0, now - calibrationStartedAt);
            episodeMinCoupling = episodeMinCoupling === null
              ? observedCoupling
              : Math.min(episodeMinCoupling, observedCoupling);
            var learnCoupling = Math.min(observedCoupling, episodeMinCoupling + 0.01);
            coupling = clamp(
              (1 - o.echoCouplingAlpha) * coupling + o.echoCouplingAlpha * learnCoupling,
              o.echoCouplingMin,
              o.echoCouplingMax
            );
          }
          if (frames >= o.echoCalibrationFrames
              && calibrationElapsedMs >= o.echoCalibrationMs
              && playbackEpisodeStartedAt !== null
              && now - playbackEpisodeStartedAt >= o.echoCalibrationMs
              && now >= confidenceHoldUntil) {
            calibrated = true;
            if (calibratedAt === null) calibratedAt = now;
          }
        }
        // A near-end burst is not an echo observation, so it must not become the
        // stability baseline for the next genuine echo frame either.
        if (!nearEndIntrusion) lastObservedEcho = observedEcho;
      }

      var projected = level * coupling;
      var target = (qualified && !ambiguousNearEnd && !nearEndIntrusion)
        ? Math.min(rms, projected + noise)
        : (calibrated || frames > 0 ? projected + noise : 0);
      var alpha = target >= reference ? o.echoReferenceAttack : o.echoReferenceRelease;
      reference = Math.max(0, (1 - alpha) * reference + alpha * target);
    } else if (now <= tailUntil && reference > 0) {
      var steps = elapsed > 0 ? elapsed / 30 : 0;
      var decayed = Math.max(noise, reference * Math.pow(o.echoTailDecayPer30Ms, steps));
      var tailObserved = Math.max(0, rms - noise);
      var tailLooksLikeEcho = calibrated
        && tailObserved <= Math.max(reference * 1.25, reference + 0.012);
      reference = tailLooksLikeEcho ? Math.max(decayed, tailObserved) : decayed;
    } else {
      reference = 0;
      tailUntil = 0;
    }

    if (!active && playbackEpisode > 0
        && now - lastPlaybackAt > o.echoConfidenceInactivityMs) {
      frames = 0;
      calibrationStartedAt = null;
      calibrationElapsedMs = 0;
      calibrated = false;
      calibratedAt = null;
      episodeMinCoupling = null;
      lastObservedEcho = null;
      weakResidualActive = false;
      lastQualifiedAt = null;
      reference = 0;
      tailUntil = 0;
      confidenceHoldUntil = 0;
    }

    return {
      couplingEstimate: coupling,
      calibrationFrames: frames,
      calibrationElapsedMs: calibrationElapsedMs,
      calibrationStartedAt: calibrationStartedAt,
      calibrated: calibrated,
      calibratedAt: calibratedAt,
      referenceRms: reference,
      lastPlaybackAt: lastPlaybackAt,
      playbackWasActive: active,
      playbackEpisode: playbackEpisode,
      playbackEpisodeStartedAt: playbackEpisodeStartedAt,
      episodeMinCoupling: episodeMinCoupling,
      lastObservedEcho: lastObservedEcho,
      weakResidualActive: weakResidualActive,
      lastQualifiedAt: lastQualifiedAt,
      tailUntil: tailUntil,
      tailActive: !active && tailUntil > 0 && now <= tailUntil && reference > 0,
      confidenceHoldUntil: confidenceHoldUntil,
      lastUpdateAt: now,
    };
  }

  /**
   * Dynamic self-echo gate. When Oracle is playing, the mic picks up its own
   * output (acoustic echo). Onset is allowed only if the captured level exceeds
   * BOTH the ordinary onset threshold AND the projected echo level times a
   * double-talk margin -- so pure echo (the t11 phantom-while-silent failure) is
   * rejected while a genuine louder barge-in (t08/t09) still fires. When
   * playback is inactive it is a plain threshold compare (no behavior change).
   */
  function echoGuardOnset(args) {
    var a = args || {};
    var rms = isFiniteNum(a.smoothedRms) ? a.smoothedRms : 0;
    var onset = isFiniteNum(a.onsetThreshold) ? a.onsetThreshold : VAD_DEFAULTS.floorMin;
    var tailActive = !!a.tailActive;
    if (!a.playbackActive && !tailActive) {
      return { allow: rms >= onset, reason: rms >= onset ? 'onset' : 'below_onset',
        effectiveThreshold: onset, projectedEcho: 0 };
    }
    var coupling = isFiniteNum(a.coupling) ? a.coupling : VAD_DEFAULTS.echoCoupling;
    var dtMargin = isFiniteNum(a.dtMargin) ? a.dtMargin : VAD_DEFAULTS.echoDoubleTalkMargin;
    var level = isFiniteNum(a.playbackLevel) ? a.playbackLevel : 0;
    var adaptiveReference = isFiniteNum(a.referenceRms) ? Math.max(0, a.referenceRms) : 0;
    var projectedEcho = Math.max(level * coupling, adaptiveReference);
    var effective = Math.max(onset, projectedEcho * dtMargin);
    var allow = rms >= effective;
    return {
      allow: allow,
      reason: allow ? 'double_talk_over_echo' : 'echo_suppressed',
      effectiveThreshold: effective,
      projectedEcho: projectedEcho,
    };
  }

  /**
   * Recheck the echo decision immediately before maybe_speech is promoted to
   * speech and barge-in side effects fire. A calibrated decision needs both
   * total energy above the adaptive echo bar and positive residual energy above
   * the learned reference. While calibration is unknown, only unmistakable
   * near-end energy may promote; ambiguous energy fails closed. This is an
   * intentionally conservative candidate policy pending physical calibration.
   */
  function echoGuardTransition(args) {
    var a = args || {};
    var rms = isFiniteNum(a.smoothedRms) ? Math.max(0, a.smoothedRms) : 0;
    var onset = isFiniteNum(a.onsetThreshold) ? a.onsetThreshold : VAD_DEFAULTS.floorMin;
    var echoWindow = !!a.playbackActive || !!a.tailActive;
    var now = isFiniteNum(a.now) ? a.now : 0;
    var confidenceHoldUntil = isFiniteNum(a.confidenceHoldUntil)
      ? Math.max(0, a.confidenceHoldUntil) : 0;
    var reference = isFiniteNum(a.referenceRms) ? Math.max(0, a.referenceRms) : 0;
    var noise = isFiniteNum(a.noiseFloor) ? Math.max(0, a.noiseFloor) : 0;
    var holdActive = confidenceHoldUntil > now;
    var calibrated = !!a.calibrated && !holdActive;
    if (!echoWindow) {
      return { allow: rms >= onset, reason: rms >= onset ? 'transition_onset' : 'below_onset',
        calibrated: calibrated, referenceRms: 0, residualRms: rms,
        effectiveThreshold: onset };
    }

    var residual = rms - reference;
    if (holdActive) {
      var holdClear = Math.max(
        VAD_DEFAULTS.echoUnknownClearFloor * 1.5,
        onset * VAD_DEFAULTS.echoUnknownOnsetMult
      );
      var holdAllow = rms >= holdClear;
      return { allow: holdAllow,
        reason: holdAllow ? 'confidence_hold_clear_near_end' : 'confidence_hold_fail_closed',
        calibrated: false, referenceRms: reference, residualRms: residual,
        effectiveThreshold: holdClear, confidenceHoldUntil: confidenceHoldUntil };
    }
    if (!!a.tailActive && !a.playbackActive && !calibrated) {
      var tailClear = Math.max(
        VAD_DEFAULTS.echoUnknownClearFloor,
        onset * VAD_DEFAULTS.echoUnknownOnsetMult,
        reference + VAD_DEFAULTS.echoUnknownResidualFloor
      );
      var tailAllow = rms >= tailClear;
      return { allow: tailAllow, reason: tailAllow ? 'tail_clear_near_end' : 'tail_fail_closed',
        calibrated: false, referenceRms: reference, residualRms: residual,
        effectiveThreshold: tailClear };
    }
    if (!calibrated) {
      var unknownClear = Math.max(
        VAD_DEFAULTS.echoUnknownClearFloor,
        onset * VAD_DEFAULTS.echoUnknownOnsetMult,
        reference + VAD_DEFAULTS.echoUnknownResidualFloor
      );
      var clear = rms >= unknownClear;
      var unknownReason = holdActive ? 'confidence_hold' : 'uncalibrated';
      return { allow: clear, reason: clear ? unknownReason + '_clear_near_end' : unknownReason + '_fail_closed',
        calibrated: false, referenceRms: reference, residualRms: residual,
        effectiveThreshold: unknownClear, confidenceHoldUntil: confidenceHoldUntil };
    }

    var dtMargin = isFiniteNum(a.dtMargin) ? a.dtMargin : VAD_DEFAULTS.echoDoubleTalkMargin;
    var residualFloor = Math.max(
      VAD_DEFAULTS.echoResidualFloor,
      onset * VAD_DEFAULTS.echoResidualOnsetMult,
      noise * 2,
      reference * 0.25
    );
    var effective = Math.max(onset, reference * dtMargin);
    var allow = rms >= effective && residual >= residualFloor;
    var allowedReason = !!a.tailActive && !a.playbackActive
      ? 'adaptive_tail_double_talk'
      : 'adaptive_double_talk';
    return { allow: allow, reason: allow ? allowedReason : 'adaptive_echo_suppressed',
      calibrated: true, referenceRms: reference, residualRms: residual,
      residualFloor: residualFloor, effectiveThreshold: effective };
  }

  /**
   * Anti-fragment submit gate. A confirmed offset still must not submit a short
   * echo fragment to ASR (the source of the "Thank you." hallucinations). Reject
   * when the utterance is too short, its peak is too weak, or it happened almost
   * entirely under active playback without ever exceeding the projected echo.
   */
  function shouldSubmitUtterance(args) {
    var a = args || {};
    var minMs = isFiniteNum(a.minUtteranceMs) ? a.minUtteranceMs : VAD_DEFAULTS.minUtteranceMs;
    var peakMult = isFiniteNum(a.submitPeakMult) ? a.submitPeakMult : VAD_DEFAULTS.submitPeakMult;
    var onset = isFiniteNum(a.onsetThreshold) ? a.onsetThreshold : VAD_DEFAULTS.floorMin;
    var speechMs = isFiniteNum(a.speechMs) ? a.speechMs : 0;
    var peakRms = isFiniteNum(a.peakRms) ? a.peakRms : 0;
    if (speechMs < minMs) {
      return { submit: false, reason: 'too_short', detail: speechMs + 'ms<' + minMs + 'ms' };
    }
    if (peakRms < onset * peakMult) {
      return { submit: false, reason: 'weak_peak', detail: peakRms + '<' + (onset * peakMult) };
    }
    // Echo-dominated fragment: mostly under playback and never clearly above echo.
    var pbFrac = isFiniteNum(a.playbackActiveFraction) ? a.playbackActiveFraction : 0;
    if (pbFrac >= 0.6) {
      var coupling = isFiniteNum(a.coupling) ? a.coupling : VAD_DEFAULTS.echoCoupling;
      var dtMargin = isFiniteNum(a.dtMargin) ? a.dtMargin : VAD_DEFAULTS.echoDoubleTalkMargin;
      var pbLevel = isFiniteNum(a.playbackMeanLevel) ? a.playbackMeanLevel : 0;
      var echoBar = pbLevel * coupling * dtMargin;
      if (peakRms < echoBar) {
        return { submit: false, reason: 'echo_fragment', detail: peakRms + '<' + echoBar };
      }
    }
    return { submit: true, reason: 'ok', detail: '' };
  }

  /**
   * Reference VAD state machine mirroring index.html's vadTick transitions, so a
   * recorded RMS trace can be replayed deterministically in Node. Uses the same
   * pure decision helpers the served page delegates to. Emits 'onset'/'offset'
   * events with the same min-speech / hysteresis / hangover semantics.
   *
   * push(frame) where frame = {rms, now, playbackActive?, playbackLevel?}
   *   returns {phase, event, onsetThreshold, silenceThreshold, noiseFloor, ...}
   */
  function createVadMachine(cfg) {
    var c = Object.assign({}, VAD_DEFAULTS, cfg || {});
    var st = {
      phase: 'silence',
      smoothedRms: 0,
      noiseFloor: c.noiseFloorMin,
      lastTransitionAt: null,
      speechStartedAt: 0,
      lastSpeechSampleAt: 0,
      peakRms: 0,
      playbackFramesInSpeech: 0,
      framesInSpeech: 0,
      playbackLevelSumInSpeech: 0,
      echoReference: createEchoReferenceState(c),
      transitionRejectReason: null,
    };

    function push(frame) {
      var f = frame || {};
      var rms = isFiniteNum(f.rms) ? f.rms : 0;
      var now = isFiniteNum(f.now) ? f.now : 0;
      var pbActive = !!f.playbackActive;
      var pbLevel = isFiniteNum(f.playbackLevel) ? f.playbackLevel : 0;
      if (st.lastTransitionAt === null) st.lastTransitionAt = now;

      st.smoothedRms = (1 - c.emaAlpha) * st.smoothedRms + c.emaAlpha * rms;

      // Freeze noise-floor learning during playback and OUTSIDE the pure
      // 'silence' phase. Learning through 'maybe_speech' would let the onset ramp
      // inflate the floor and re-clip the very quiet-far-field speech we are
      // trying to catch (observed: floor 0.0014 -> 0.0053, onset 0.010 -> 0.017,
      // which then failed the anti-fragment peak gate on a real 0.018 utterance).
      var learn = !pbActive && st.phase === 'silence';
      st.noiseFloor = updateNoiseFloor(st.noiseFloor, st.smoothedRms, {
        floorAlpha: c.floorAlpha, noiseFloorMin: c.noiseFloorMin,
        noiseFloorMax: c.noiseFloorMax, update: learn,
      });

      var thr = effectiveOnsetThreshold(c.userThreshold != null ? c.userThreshold : 0.025,
        st.noiseFloor, c);
      var speechT = thr.onset;
      var silenceT = thr.silence;
      var event = null;
      st.echoReference = updateEchoReference(st.echoReference, {
        now: now,
        smoothedRms: st.smoothedRms,
        noiseFloor: st.noiseFloor,
        playbackActive: pbActive,
        playbackLevel: pbLevel,
        phase: st.phase,
      }, c);

      switch (st.phase) {
        case 'silence': {
          var guard = echoGuardOnset({
            smoothedRms: st.smoothedRms, onsetThreshold: speechT,
            playbackActive: pbActive, playbackLevel: pbLevel,
            coupling: st.echoReference.couplingEstimate,
            dtMargin: c.echoDoubleTalkMargin,
            referenceRms: st.echoReference.referenceRms,
            tailActive: st.echoReference.tailActive,
          });
          if (guard.allow) {
            st.phase = 'maybe_speech';
            st.lastTransitionAt = now;
          }
          break;
        }
        case 'maybe_speech': {
          if (st.smoothedRms < silenceT) {
            st.phase = 'silence';
            st.lastTransitionAt = now;
            break;
          }
          if (now - st.lastTransitionAt >= c.minSpeechMs) {
            var transition = echoGuardTransition({
              now: now,
              smoothedRms: st.smoothedRms,
              onsetThreshold: speechT,
              noiseFloor: st.noiseFloor,
              playbackActive: pbActive,
              tailActive: st.echoReference.tailActive,
              referenceRms: st.echoReference.referenceRms,
              calibrated: st.echoReference.calibrated,
              confidenceHoldUntil: st.echoReference.confidenceHoldUntil,
              dtMargin: c.echoDoubleTalkMargin,
            });
            if (!transition.allow) {
              st.phase = 'silence';
              st.lastTransitionAt = now;
              st.transitionRejectReason = transition.reason;
              break;
            }
            st.phase = 'speech';
            st.lastTransitionAt = now;
            st.speechStartedAt = now;
            st.lastSpeechSampleAt = now;
            st.peakRms = st.smoothedRms;
            st.framesInSpeech = 1;
            st.playbackFramesInSpeech = pbActive ? 1 : 0;
            st.playbackLevelSumInSpeech = pbActive ? pbLevel : 0;
            st.transitionRejectReason = null;
            event = 'onset';
          }
          break;
        }
        case 'speech': {
          st.framesInSpeech += 1;
          if (pbActive) { st.playbackFramesInSpeech += 1; st.playbackLevelSumInSpeech += pbLevel; }
          if (st.smoothedRms > st.peakRms) st.peakRms = st.smoothedRms;
          if (st.smoothedRms >= silenceT) {
            st.lastSpeechSampleAt = now;
          } else {
            st.phase = 'maybe_silence';
            st.lastTransitionAt = now;
          }
          break;
        }
        case 'maybe_silence': {
          st.framesInSpeech += 1;
          if (pbActive) { st.playbackFramesInSpeech += 1; st.playbackLevelSumInSpeech += pbLevel; }
          if (st.smoothedRms > st.peakRms) st.peakRms = st.smoothedRms;
          if (st.smoothedRms >= speechT) {
            st.phase = 'speech';
            st.lastSpeechSampleAt = now;
            break;
          }
          var speechMs = st.lastSpeechSampleAt - st.speechStartedAt;
          var hang = c.hangoverMs;
          if (speechMs >= (c.adaptiveLongSpeechMs || 2600)) {
            hang = Math.max((c.adaptiveFloorMs || 900), Math.round(c.hangoverMs * 0.65));
          }
          if (now - st.lastTransitionAt >= hang) {
            st.phase = 'silence';
            st.lastTransitionAt = now;
            event = 'offset';
          }
          break;
        }
        default:
          break;
      }

      return {
        phase: st.phase,
        event: event,
        onsetThreshold: speechT,
        silenceThreshold: silenceT,
        noiseFloor: st.noiseFloor,
        smoothedRms: st.smoothedRms,
        peakRms: st.peakRms,
        speechMs: st.lastSpeechSampleAt - st.speechStartedAt,
        playbackActiveFraction: st.framesInSpeech ? st.playbackFramesInSpeech / st.framesInSpeech : 0,
        playbackMeanLevel: st.playbackFramesInSpeech ? st.playbackLevelSumInSpeech / st.playbackFramesInSpeech : 0,
        echoReferenceRms: st.echoReference.referenceRms,
        echoCouplingEstimate: st.echoReference.couplingEstimate,
        echoCalibrated: st.echoReference.calibrated,
        echoConfidenceHoldUntil: st.echoReference.confidenceHoldUntil,
        echoTailActive: st.echoReference.tailActive,
        transitionRejectReason: st.transitionRejectReason,
      };
    }

    return { push: push, state: function () { return Object.assign({}, st); } };
  }

  /**
   * Deterministic ordered chunk buffer (low/high watermark jitter buffer MODEL).
   * Guarantees, for a single turn:
   *   - IN-ORDER emission (dequeue order == monotonic enqueue index order),
   *   - NO drop and NO reorder of accepted chunks,
   *   - deterministic playback start once depth >= highWatermark OR flush(),
   *   - bounded depth (>= maxDepth raises backpressure; it does NOT drop),
   *   - honest underflow signalling when a pull finds the buffer empty while
   *     playback is running (a producer slower than real time is REPORTED, never
   *     hidden by growing the buffer -- consistent with the served page's
   *     deliberate "a jitter buffer cannot close a multi-second deficit" design).
   * This is a decision/accounting model for tests + reviewer reference; it does
   * not itself schedule Web Audio.
   */
  function createOrderedChunkBuffer(opts) {
    var o = opts || {};
    var low = isFiniteNum(o.lowWatermark) ? o.lowWatermark : 1;
    var high = isFiniteNum(o.highWatermark) ? o.highWatermark : 2;
    var maxDepth = isFiniteNum(o.maxDepth) ? o.maxDepth : 64;
    if (high < low) high = low;
    var q = [];
    var nextExpected = 0;      // monotonic index we will accept next
    var emittedOrder = [];
    var started = false;
    var underflowCount = 0;
    var droppedStale = 0;

    function enqueue(index, item) {
      // Reject a stale/duplicate/out-of-order index: never reorder, never
      // silently overwrite. A caller that barges a turn uses a NEW buffer.
      if (index !== nextExpected) {
        droppedStale += 1;
        return { accepted: false, reason: index < nextExpected ? 'stale_or_duplicate' : 'gap', depth: q.length };
      }
      if (q.length >= maxDepth) {
        return { accepted: false, reason: 'backpressure', depth: q.length };
      }
      q.push({ index: index, item: item });
      nextExpected += 1;
      if (!started && q.length >= high) started = true;
      return { accepted: true, reason: started ? 'started' : 'buffering', depth: q.length, started: started };
    }

    function flush() { started = true; return started; }

    function pull(playbackRunning) {
      if (!started) return { chunk: null, reason: 'not_started', depth: q.length };
      if (q.length === 0) {
        if (playbackRunning) underflowCount += 1;
        return { chunk: null, reason: playbackRunning ? 'underflow' : 'drained', depth: 0 };
      }
      var head = q.shift();
      emittedOrder.push(head.index);
      return { chunk: head.item, index: head.index, reason: 'ok', depth: q.length };
    }

    function state() {
      return {
        depth: q.length, started: started, nextExpected: nextExpected,
        emittedOrder: emittedOrder.slice(), underflowCount: underflowCount,
        droppedStale: droppedStale, lowWatermark: low, highWatermark: high, maxDepth: maxDepth,
      };
    }

    return { enqueue: enqueue, pull: pull, flush: flush, state: state };
  }

  // ---- WAV finite-sample helpers (browser encodeWavBlob hardening) --------
  function isFiniteSampleArray(samples) {
    if (!samples || typeof samples.length !== 'number' || samples.length <= 0) return false;
    for (var i = 0; i < samples.length; i++) {
      if (!isFiniteNum(samples[i])) return false;
    }
    return true;
  }
  function pcm16(sample) {
    // Non-finite -> 0 (silence), clamped, matching a valid finite WAV writer.
    var s = isFiniteNum(sample) ? sample : 0;
    if (s > 1) s = 1; else if (s < -1) s = -1;
    return s < 0 ? (s * 0x8000) | 0 : (s * 0x7FFF) | 0;
  }
  function wavByteLength(numSamples, opts) {
    var o = opts || {};
    var ch = isFiniteNum(o.channels) ? o.channels : 1;
    var bytesPer = (isFiniteNum(o.bitsPerSample) ? o.bitsPerSample : 16) / 8;
    return 44 + Math.max(0, numSamples | 0) * ch * bytesPer;
  }

  return {
    VERSION: VERSION,
    VAD_DEFAULTS: VAD_DEFAULTS,
    isFiniteNum: isFiniteNum,
    clamp: clamp,
    updateNoiseFloor: updateNoiseFloor,
    effectiveOnsetThreshold: effectiveOnsetThreshold,
    createEchoReferenceState: createEchoReferenceState,
    updateEchoReference: updateEchoReference,
    echoGuardOnset: echoGuardOnset,
    echoGuardTransition: echoGuardTransition,
    shouldSubmitUtterance: shouldSubmitUtterance,
    createVadMachine: createVadMachine,
    createOrderedChunkBuffer: createOrderedChunkBuffer,
    isFiniteSampleArray: isFiniteSampleArray,
    pcm16: pcm16,
    wavByteLength: wavByteLength,
  };
}));
