"""Direct foreground (Face Lobe) chat path.

Per artifact §5.1 / §16.5, the Face Lobe should be a small/fast model
focused on status and routing — NOT a full Hermes tool-loop agent.
Going through Hermes on every turn costs tens of seconds even for
trivial messages because Hermes injects its full tool catalog,
spawns a plugin chain, and waits for a tool-or-final decision.

This module is a thin direct call to HiveMind's OpenAI-compatible
``/v1/chat/completions`` endpoint. No Hermes, no tool injection, no
plugin chain. The model produces text; we stream the text back.

Tool-requiring requests are still served correctly: the auto-router
in :mod:`machine_spirit_4.double_agent.router` dispatches a Depth
Lobe Double Agent job (which DOES use the full Hermes loop in a
subprocess) in parallel, and the foreground reply mentions the
dispatch so the user knows deeper work is happening.

Session continuity: this module keeps its own OpenAI-format message
history per session id. The auto-router and Face Lobe context block
are still applied at the gateway layer above us.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


log = logging.getLogger("ms4.gateway.face_lobe_chat")


FACE_LOBE_SYSTEM_PROMPT = (
    "You are the MS4 Face Lobe — the foreground voice of a Machine Spirit "
    "in a live conversation with the operator. You can also be reached "
    "via the operator's microphone (the UI captures their voice, "
    "transcribes it, and sends you the text). Reply naturally, in plain "
    "conversational language. Voice turns are typed `🎤 ...` on the "
    "user side; respond like you would in any conversation. Be warm, "
    "concise, and direct. A few sentences by default. Match the "
    "operator's register.\n"
    "\n"
    "You share a runtime with a heavier background Depth Lobe. The "
    "Depth Lobe is where Hermes tools, terminals, file systems, "
    "browsers, MCP calls, and long reasoning live. As the Face Lobe you "
    "DO NOT directly execute those tools — but you can dispatch jobs "
    "to the Depth Lobe (via the router), and you DO get authoritative "
    "context grounded by MS4 itself on every turn.\n"
    "\n"
    "Grounding & honesty (these prevent confabulation, not conversation):\n"
    "\n"
    " * If the context block lists real data — HiveMind inventory, "
    "MS4 tools, the current date/time, active or completed Depth Lobe "
    "jobs, the live `hivemind.jobs.active@v1` snapshot — treat it as "
    "ground truth and quote it faithfully. Include any `cached(age=Ns)` "
    "or `stale(age=Ns)` qualifier so the operator knows how fresh the "
    "data is. Repeat field shapes accurately: CPUs are CPUs, not VMs or "
    "GPUs.\n"
    " * If the context block contains `THIS TURN DISPATCHED job <id>`, "
    "a Depth Lobe job IS running for the operator's request. Acknowledge "
    "it plainly and concisely — something like 'Looking into that — "
    "I'll surface the result on the next turn' or 'Dispatched, give me "
    "a sec'. Do NOT say you can't help, do NOT say you lack tool "
    "access, do NOT write hypothetical Python or pseudocode for the "
    "task — the Depth Lobe IS handling it. The operator will follow "
    "up (often with a short 'ok' or 'do that' or 'any update?'); "
    "respond by referencing the running job. If no `THIS TURN "
    "DISPATCHED` line is present, no dispatch happened this turn — "
    "don't claim one did, don't invent job UUIDs, and don't fabricate "
    "timestamps for completions. Use only the dates you can see in "
    "the context block.\n"
    " * If a completed Depth Lobe job's `result:` already answers the "
    "operator's question, quote it back. Attribute it plainly ('the "
    "Depth Lobe found...' or 'a previous job reported...') rather than "
    "speaking as if you executed the tool yourself.\n"
    " * If the operator asks for something tool-requiring and no "
    "dispatch happened, say so plainly and suggest they prefix `/deep` "
    "to force a Depth Lobe job. Do NOT invent tool names.\n"
    "\n"
    "Voice conversation is normal conversation. Questions like 'can you "
    "hear me?', 'what's up?', or 'are you there?' are small talk — just "
    "answer naturally ('Yeah, I'm here.'). The 'no tool access' rule is "
    "about Hermes-style execution tools, not about the conversational "
    "context you and the operator already share. If the operator asks "
    "about your own state, identity, or how you work, you can talk "
    "about it as the Machine Spirit you are.\n"
    "\n"
    "Be willing to take a position. Don't reflexively apologize. Don't "
    "preface answers with disclaimers when an honest direct reply will "
    "do."
)


HTTP_TIMEOUT_DEFAULT = int(os.environ.get("MS4_FACE_LOBE_TIMEOUT", "60"))
# Stall timeout for streaming. If no new chunk arrives within this many
# seconds AFTER the first chunk, the stream is considered dead and we
# break the loop with whatever we have so far. Observed live: HiveMind
# can drop a stream mid-response when a model is preempted; without
# this guard the FaceLobeChat call hangs forever, the gateway worker
# never returns, and the UI shows "(no reply text)" with no done
# event ever firing.
STREAM_STALL_TIMEOUT_DEFAULT = int(os.environ.get("MS4_FACE_LOBE_STREAM_STALL_TIMEOUT", "12"))
STREAM_BUFFER_BYTES = 4096


class FaceLobeChatError(RuntimeError):
    """Raised when the upstream chat completion call fails or returns an
    error payload."""


@dataclass
class FaceLobeChatSession:
    """Per-session state for the Face Lobe direct chat path.

    Mirrors ``SessionState`` for symmetry with the Hermes-backed runner
    but holds OpenAI-format messages instead of a Hermes agent object.
    """

    session_id: str
    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_grounding_source: str | None = None

    # Compatibility shim: ``Ms4HermesRunner`` callers used to inspect
    # ``state.agent.model``. Expose a small object with a ``.model``
    # attr so existing reads keep working without us having to chase
    # every caller.
    @property
    def agent(self) -> "FaceLobeChatSession":
        return self


@dataclass
class TurnMetrics:
    """Per-turn metrics for one Face Lobe chat call.

    All durations are in milliseconds; all timestamps are ISO-8601 UTC
    with millisecond precision. Token counts come from HiveMind's
    ``usage`` block (the OpenAI-compatible field) when the upstream
    populates it; ``None`` when the provider doesn't return usage.
    """

    schema: str = "Ms4TurnMetrics.v1"
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0
    http_latency_ms: int = 0
    stream_first_token_ms: int | None = None
    stream_chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tokens_per_second: float | None = None
    bytes_received: int = 0
    api_calls: int = 1
    fallback_used: bool = False
    requested_model: str | None = None
    effective_model: str | None = None
    streaming: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso_utc(when: float | None = None) -> str:
    moment = datetime.fromtimestamp(when if when is not None else time.time(), tz=timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class FaceLobeChat:
    """Direct HiveMind chat-completion path for the foreground."""

    def __init__(
        self,
        *,
        hivemind_url: str = "http://127.0.0.1:6089",
        system_prompt: str = FACE_LOBE_SYSTEM_PROMPT,
        http_timeout: int = HTTP_TIMEOUT_DEFAULT,
        stream_stall_timeout: int = STREAM_STALL_TIMEOUT_DEFAULT,
        empty_fallback_model: str | None = None,
    ) -> None:
        self.hivemind_url = hivemind_url.rstrip("/")
        self.system_prompt = system_prompt
        self.http_timeout = http_timeout
        self.stream_stall_timeout = max(2, stream_stall_timeout)
        # When the chosen model returns empty content (we observed this
        # live with phi4-mini cold-loading), we retry once with this
        # safe-known-working fallback so the user never sees the bare
        # "(no reply text)" placeholder.
        self.empty_fallback_model = (
            empty_fallback_model
            or os.environ.get("MS4_DEFAULT_MODEL", "").strip()
            or "qwen3-coder-next:latest"
        )
        self._lock = threading.Lock()
        self._sessions: dict[str, FaceLobeChatSession] = {}

    # ---- session management ------------------------------------------------

    def get_or_create_session(self, session_id: str | None, model: str) -> FaceLobeChatSession:
        sid = session_id or f"ms4-{uuid.uuid4()}"
        with self._lock:
            state = self._sessions.get(sid)
            if state is None:
                state = FaceLobeChatSession(session_id=sid, model=model)
                self._sessions[sid] = state
            # Note: unlike the Hermes path, we do NOT rebuild + wipe history on
            # model change. The pinning is enforced at the gateway layer, but
            # if a per-turn override is sent we still keep history and just
            # change the model for this turn. The model is recorded per-message
            # in our own bookkeeping above the wire (the chat-completions API
            # only takes model per request, not per message).
            return state

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "model": s.model,
                    "turns": len(s.messages) // 2,
                    "last_grounding_source": s.last_grounding_source,
                }
                for s in self._sessions.values()
            ]

    # ---- chat --------------------------------------------------------------

    def chat(
        self,
        message: str,
        *,
        session_id: str | None,
        model: str,
        stream_callback: Callable[[str], None] | None = None,
        extra_system: str | None = None,
    ) -> dict[str, Any]:
        """Run one chat turn directly against HiveMind /v1/chat/completions.

        ``extra_system`` is appended to the system prompt for this turn only
        (e.g. the Face Lobe context block from the Double Agent module).
        """
        state = self.get_or_create_session(session_id, model)
        state.model = model
        system_prompt = self.system_prompt
        if extra_system:
            system_prompt = f"{self.system_prompt}\n\n{extra_system}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(state.messages)
        messages.append({"role": "user", "content": message})

        is_streaming = bool(stream_callback)
        metrics = TurnMetrics(
            started_at=_iso_utc(),
            requested_model=model,
            effective_model=model,
            streaming=is_streaming,
        )
        turn_start = time.monotonic()

        def _call(target_model: str) -> tuple[str, dict[str, Any]]:
            p: dict[str, Any] = {
                "model": target_model,
                "messages": [{"role": "system", "content": system_prompt}] + list(state.messages) + [{"role": "user", "content": message}],
                "stream": is_streaming,
                "temperature": 0.2,
            }
            # Some providers omit usage in streaming mode unless asked.
            if is_streaming:
                p["stream_options"] = {"include_usage": True}
            if is_streaming:
                return self._post_streaming(p, stream_callback)
            return self._post_blocking(p)

        text, stats = _call(model)
        metrics.api_calls = 1
        metrics.http_latency_ms += int(stats.get("http_latency_ms") or 0)
        metrics.bytes_received += int(stats.get("bytes_received") or 0)
        if stats.get("stream_first_token_ms") is not None:
            metrics.stream_first_token_ms = int(stats["stream_first_token_ms"])
        metrics.stream_chunks += int(stats.get("stream_chunks") or 0)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = stats.get(k)
            if v is not None and (getattr(metrics, k) is None):
                setattr(metrics, k, int(v))
        effective_model = model
        # Empty-content fallback chain. Live evidence (May 26 2026):
        # operator selected qwen3.6:27b in the UI dropdown — that
        # model returned no content AND the static fallback
        # qwen3-coder-next:latest also returned nothing, leaving the
        # operator with a useless canned message. Fix: after the
        # static fallback fails, try the auto-picker's choice
        # (typically a 7-8B instruct we know works) as a last resort.
        tried_models = [model]
        if not text.strip() and self.empty_fallback_model and self.empty_fallback_model != model:
            log.warning(
                "face_lobe model %r returned empty content; falling back to %r",
                model, self.empty_fallback_model,
            )
            try:
                fallback_text, fb_stats = _call(self.empty_fallback_model)
                tried_models.append(self.empty_fallback_model)
                metrics.api_calls += 1
                metrics.fallback_used = True
                metrics.http_latency_ms += int(fb_stats.get("http_latency_ms") or 0)
                metrics.bytes_received += int(fb_stats.get("bytes_received") or 0)
                if fb_stats.get("stream_first_token_ms") is not None:
                    metrics.stream_first_token_ms = int(fb_stats["stream_first_token_ms"])
                metrics.stream_chunks += int(fb_stats.get("stream_chunks") or 0)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = fb_stats.get(k)
                    if v is not None:
                        setattr(metrics, k, int(v))
                if fallback_text.strip():
                    text = fallback_text
                    effective_model = self.empty_fallback_model
            except FaceLobeChatError as exc:
                log.warning("face_lobe fallback to %r also failed: %s", self.empty_fallback_model, exc)

        # Last-resort: ask the picker for the cluster's best available
        # foreground model and try it. The picker prefers loaded 7-8B
        # instruct models which we know follow the contract.
        if not text.strip():
            try:
                from machine_spirit_4.double_agent.model_picker import choose_foreground_model
                picker_choice = choose_foreground_model(
                    hivemind_url=self.hivemind_url, force_refresh=False
                )
                picker_model = picker_choice.model_id
                if picker_model and picker_model not in tried_models:
                    log.warning(
                        "face_lobe both %r and %r returned empty; last-resort fallback to picker pick %r",
                        model, self.empty_fallback_model, picker_model,
                    )
                    try:
                        picker_text, pk_stats = _call(picker_model)
                        tried_models.append(picker_model)
                        metrics.api_calls += 1
                        metrics.fallback_used = True
                        metrics.http_latency_ms += int(pk_stats.get("http_latency_ms") or 0)
                        metrics.bytes_received += int(pk_stats.get("bytes_received") or 0)
                        if picker_text.strip():
                            text = picker_text
                            effective_model = picker_model
                    except FaceLobeChatError as exc:
                        log.warning(
                            "face_lobe picker last-resort fallback to %r also failed: %s",
                            picker_model, exc,
                        )
            except Exception as exc:
                log.warning("picker-based last-resort fallback path failed to even pick: %s", exc)

        cleaned = text.strip()
        if not cleaned:
            tried_str = ", ".join(repr(m) for m in tried_models)
            cleaned = (
                f"(no model produced any visible content — tried {tried_str}. "
                f"Try picking a different model in the dropdown, or set it "
                f"to '(auto-picker)' so MS4 picks a loaded one.)"
            )
        metrics.completed_at = _iso_utc()
        metrics.duration_ms = int((time.monotonic() - turn_start) * 1000)
        metrics.effective_model = effective_model
        if metrics.completion_tokens and metrics.duration_ms:
            denom_ms = max(1, metrics.duration_ms - (metrics.stream_first_token_ms or 0))
            metrics.tokens_per_second = round(metrics.completion_tokens / (denom_ms / 1000.0), 2)

        # Commit user + assistant to history only after a successful turn so
        # a partial / errored call doesn't corrupt the session.
        state.messages.append({"role": "user", "content": message})
        state.messages.append({"role": "assistant", "content": cleaned})
        return {
            "text": cleaned,
            "session_id": state.session_id,
            "model": effective_model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "api_calls": metrics.api_calls,
            "fallback_used": metrics.fallback_used,
            "requested_model": model,
            "metrics": metrics.to_dict(),
        }

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # ---- transport ---------------------------------------------------------

    def _build_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        from .hivemind_state import hivemind_auth_headers

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json" if not payload.get("stream") else "text/event-stream",
        }
        headers.update(hivemind_auth_headers())
        return urllib.request.Request(
            f"{self.hivemind_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

    def _post_blocking(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        req = self._build_request(payload)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as response:
                body_bytes = response.read()
                body = body_bytes.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise FaceLobeChatError(
                f"HiveMind /v1/chat/completions {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')[:400]}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise FaceLobeChatError(f"HiveMind /v1/chat/completions unreachable: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise FaceLobeChatError(f"HiveMind returned non-JSON body: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise FaceLobeChatError(f"HiveMind returned error payload: {data.get('error')}")
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise FaceLobeChatError(f"HiveMind returned no choices: {str(data)[:200]}")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = str(message.get("content") or "")
        usage = data.get("usage") if isinstance(data, dict) else None
        stats = {
            "http_latency_ms": int((time.monotonic() - t0) * 1000),
            "bytes_received": len(body_bytes),
        }
        if isinstance(usage, dict):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if k in usage:
                    stats[k] = usage[k]
        return text, stats

    def _post_streaming(
        self,
        payload: dict[str, Any],
        stream_callback: Callable[[str], None],
    ) -> tuple[str, dict[str, Any]]:
        """Consume HiveMind's SSE chat-completions stream with a stall
        timeout.

        The previous implementation iterated the response with
        ``for raw_line in response:`` which blocks indefinitely if
        HiveMind drops the stream mid-response (observed live with
        phi4-mini cold-loading). We now perform a per-line read with
        an overall stall budget: if no new newline-terminated chunk
        arrives within ``self.stream_stall_timeout`` seconds, we break
        the loop and return whatever we have so far. The socket-level
        ``timeout`` parameter on ``urlopen`` enforces the per-read
        deadline.
        """
        req = self._build_request(payload)
        accumulated: list[str] = []
        stats: dict[str, Any] = {
            "http_latency_ms": 0,
            "bytes_received": 0,
            "stream_chunks": 0,
            "stream_first_token_ms": None,
        }
        t_open = time.monotonic()
        try:
            response = urllib.request.urlopen(req, timeout=self.stream_stall_timeout)
        except urllib.error.HTTPError as exc:
            raise FaceLobeChatError(
                f"HiveMind /v1/chat/completions (stream) {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')[:400]}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise FaceLobeChatError(f"HiveMind /v1/chat/completions (stream) unreachable: {exc}") from exc

        stats["http_latency_ms"] = int((time.monotonic() - t_open) * 1000)
        first_token_at: float | None = None
        last_data_at = time.monotonic()
        try:
            while True:
                try:
                    raw_line = response.readline()
                except (TimeoutError, OSError) as exc:
                    log.warning(
                        "face_lobe stream stalled (%s); returning %d accumulated chars",
                        exc,
                        sum(len(p) for p in accumulated),
                    )
                    break
                if not raw_line:
                    break  # EOF
                stats["bytes_received"] += len(raw_line)
                line = raw_line.decode("utf-8", "replace").strip()
                if line:
                    last_data_at = time.monotonic()
                # Soft stall guard: if the upstream is sending heartbeats
                # but never any content, fall through after we've waited
                # too long total.
                if (time.monotonic() - last_data_at) > self.stream_stall_timeout:
                    log.warning(
                        "face_lobe stream stalled (%.1fs since last data); breaking",
                        time.monotonic() - last_data_at,
                    )
                    break
                if not line:
                    continue
                if line.startswith(":"):
                    continue  # SSE comment / keep-alive
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                stats["stream_chunks"] += 1
                # Some providers send a final usage-only event after [DONE]
                # or as the last chunk with no choices.
                usage = event.get("usage") if isinstance(event, dict) else None
                if isinstance(usage, dict):
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if k in usage:
                            stats[k] = usage[k]
                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else None
                    if delta is None:
                        msg = choice.get("message") if isinstance(choice.get("message"), dict) else None
                        if msg is not None:
                            fragment = msg.get("content") or ""
                            if fragment:
                                if first_token_at is None:
                                    first_token_at = time.monotonic()
                                accumulated.append(fragment)
                                try:
                                    stream_callback(fragment)
                                except Exception as exc:
                                    log.warning("face_lobe stream_callback raised: %s", exc)
                        continue
                    fragment = delta.get("content") or ""
                    if fragment:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        accumulated.append(fragment)
                        try:
                            stream_callback(fragment)
                        except Exception as exc:
                            log.warning("face_lobe stream_callback raised: %s", exc)
        finally:
            try:
                response.close()
            except Exception:
                pass
        if first_token_at is not None:
            stats["stream_first_token_ms"] = int((first_token_at - t_open) * 1000)
        return "".join(accumulated), stats
