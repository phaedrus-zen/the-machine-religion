"""Toolbox taxonomy — derive toolboxes from tool ids, cluster them,
and provide keyword hints for the deterministic retrieval tier.

The hierarchy mirrors the operator's "tool shed → toolbox → tool"
metaphor:

* **Tool** — a single MCP tool id, e.g. ``hivemind.vm.list@v1``.
* **Toolbox** — the canonical *domain* of related tools, derived
  mechanically from the id (``vm`` in the example above).
* **Tool shed cluster** — a small curated grouping of toolboxes
  (~7 clusters). The only non-derived part — a static dict below.

The deterministic tier of the Quartermaster cascade short-circuits
when a request unambiguously names a toolbox via the keyword
lexicon. Higher tiers (embeddings, tiny-LLM) only run when the
deterministic answer is ambiguous or empty.
"""

from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# Domain derivation
# ---------------------------------------------------------------------------


def tool_domain(tool_id: str) -> str:
    """Extract the canonical toolbox domain from a tool id.

    The algorithm reads the dot-delimited segments after the source
    prefix:

    * ``hivemind.<domain>.<verb>@v1``           → ``domain``
    * ``hivemind.<domain>.<subdomain>.<verb>@v1`` → ``domain`` (we
      collapse subdomain hierarchies up to the top-level domain;
      the per-tool description carries the finer detail)
    * ``ms4.hivemind.<domain>.<verb>@v1``       → ``domain``
      (so an MS4 proxy maps to the same toolbox as the HiveMind tool
      it wraps)
    * ``ms4.<topic>.<verb>@v1``                 → ``topic``
    * ``ext.<server>.<tool>@v1``                → ``server`` (imported
      3rd-party MCP tools group by their source server)
    * anything else                             → first segment

    Returns ``"unknown"`` for empty/malformed ids so downstream code
    can group them coherently rather than crash.
    """
    if not tool_id:
        return "unknown"
    base = tool_id.split("@", 1)[0]
    parts = [p for p in base.split(".") if p]
    if not parts:
        return "unknown"
    head = parts[0]
    if head == "hivemind":
        return parts[1] if len(parts) >= 2 else "hivemind"
    if head == "ms4":
        if len(parts) >= 3 and parts[1] == "hivemind":
            return parts[2]
        return parts[1] if len(parts) >= 2 else "ms4"
    if head == "ext":
        # Imported 3rd-party MCP tools: ext.<server>.<tool> -> <server>.
        return parts[1] if len(parts) >= 2 else "ext"
    return head


def canonical_toolbox(domain: str) -> str:
    """Return the canonical toolbox name for a derived ``domain``.

    Folds known aliases so synonyms group together (e.g. the MS4
    ``vms`` snapshot proxy → ``vm``, ``apps`` discovery → ``app``).
    Unknown domains pass through unchanged."""
    return _DOMAIN_ALIASES.get(domain, domain)


_DOMAIN_ALIASES: dict[str, str] = {
    "vms": "vm",
    "apps": "app",
    # ``hivemind.capability.matrix@v1`` derives domain ``capability``;
    # the MS4 proxy ``ms4.hivemind.capability_matrix@v1`` derives
    # ``capability_matrix``. Fold the latter into the former so both
    # land in one toolbox.
    "capability_matrix": "capability",
}


# ---------------------------------------------------------------------------
# Tool-shed cluster map
# ---------------------------------------------------------------------------
#
# Curated grouping. Each toolbox should appear in exactly ONE cluster
# (verified by :func:`cluster_for_toolbox`). When HiveMind ships a new
# domain we don't know about, ``cluster_for_toolbox`` falls back to
# ``"unclassified"`` rather than guessing — the embeddings tier still
# routes correctly; only the cluster label is missing.


TOOL_SHED_CLUSTERS: dict[str, list[str]] = {
    "compute_infra": [
        "vm",
        "storage",
        "network",
        "gpu",
        "gpu_mode",
        "hosts",
        "capability",
        "cluster",
        "resources",
        "provision",
        "deploy",
    ],
    "ai_inference": [
        "models",
        "inference",
        "ollama",
        "embeddings",
        "vlm",
        "logos",
        "training",
        "adapters",
        "loadout",
    ],
    "voice": [
        "audio",
        "voice_identities",
        "voice",
    ],
    "neuro": [
        "crown",
    ],
    "game": [
        "game",
        "game_session",
        "psykyo",
    ],
    "media_io": [
        "images",
        "vision",
        "files",
        "web",
        "http",
        "search",
    ],
    "ops": [
        "services",
        "jobs",
        "human",
        "time",
        "math",
        "api_keys",
        "oracle",
        "service_health",
    ],
    "ms4_local": [
        "identity",
        "ethics",
        "sessions",
        "chat",
        "doctrine",
        "nibbles",
        "hermes",
        "runtime",
        "double_agent",
        "desktop",
    ],
}


# Imported 3rd-party MCP tools (``ext.<server>.<tool>@v1``) all live in
# this cluster. Their toolbox is the server id (dynamic), so they can't
# be listed in TOOL_SHED_CLUSTERS statically — the catalog sets the
# cluster directly when building external ToolEntries.
EXTERNAL_CLUSTER = "external_mcp"


def cluster_for_toolbox(toolbox: str) -> str:
    """Return the tool-shed cluster a toolbox belongs to, or
    ``"unclassified"`` for unknown toolboxes (forward-compat with
    HiveMind shipping new domains)."""
    canon = canonical_toolbox(toolbox)
    for cluster, toolboxes in TOOL_SHED_CLUSTERS.items():
        if canon in toolboxes:
            return cluster
    return "unclassified"


# ---------------------------------------------------------------------------
# Deterministic-tier keyword lexicon
# ---------------------------------------------------------------------------
#
# Each toolbox maps to a small list of unambiguous lower-case
# keywords/phrases. The deterministic tier of the cascade scans a
# normalized query for these and short-circuits when exactly ONE
# toolbox matches — that's the "list my VMs" sub-millisecond path.
#
# Keyword discipline:
#  * Phrases over single common words (e.g. "voice identity" not
#    "voice" alone, which would also match audio/TTS asks).
#  * Avoid false-friend words shared across toolboxes
#    ("model" appears in models, training, inference — let the
#    embeddings tier disambiguate those rather than racing).
#  * Specificity beats coverage — better to defer to embeddings
#    than mis-route the deterministic tier.


TOOLBOX_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vm": (
        "vm",
        "vms",
        "virtual machine",
        "virtual machines",
        "hyper-v",
        "hyperv",
        "guest os",
    ),
    "storage": (
        "storage pool",
        "storage pools",
        "volume",
        "volumes",
        "snapshot",
        "snapshots",
        "disk volume",
    ),
    "network": (
        "cluster network",
        "vlan",
        "bridge",
        "bridges",
        "network interface",
        "network interfaces",
        "network attachment",
        "isolate network",
    ),
    "gpu": (
        "gpu availability",
        "free gpus",
        "available gpus",
        "list gpus",
        "any gpus",
        "what gpus",
        "do you see any gpus",
    ),
    "gpu_mode": (
        "gpu mode",
        "vgpu",
        "passthrough",
        "dda",
        "discrete device assignment",
        "gpu partition",
        "gpu partitioning",
    ),
    "hosts": (
        "host",
        "hosts",
        "cluster nodes",
        "list nodes",
        "what nodes",
        "host list",
    ),
    "capability": (
        "capability matrix",
        "capabilities matrix",
        "what can each node",
        "per-node capabilities",
        "per node capability",
    ),
    "cluster": (
        "cluster summary",
        "cluster load",
        "load stats",
    ),
    "resources": (
        "resource request",
        "lease",
        "resource lease",
        "release lease",
    ),
    "provision": (
        "provision",
        "provisioning",
        "provision status",
    ),
    "deploy": (
        "deploy gim",
        "deploy a gim",
    ),
    "models": (
        "model catalog",
        "list models",
        "what models",
        "recommend a model",
        "model recommendation",
    ),
    "inference": (
        "chat completion",
        "chat completions",
        "inference chat",
        "direct inference",
    ),
    "ollama": (
        "ollama",
    ),
    "embeddings": (
        "embedding",
        "embeddings",
        "embed text",
    ),
    "vlm": (
        "describe image",
        "vlm",
        "vision language model",
    ),
    "logos": (
        "prompt optimization",
        "prompt optimizer",
        "logos machina",
        "optimize prompt",
    ),
    "training": (
        "train a model",
        "fine-tune",
        "fine tune",
        "finetune",
        "lora training",
        "peft training",
        "forge",
    ),
    "adapters": (
        "adapter",
        "adapters",
        "lora adapter",
        "peft adapter",
        "deploy adapter",
    ),
    "loadout": (
        "loadout",
        "model loadout",
        "loadout profile",
        "load profile",
        "apply loadout",
    ),
    "audio": (
        "transcribe",
        "transcription",
        "asr",
        "speech to text",
        "text to speech",
        "tts",
    ),
    "voice_identities": (
        "voice identity",
        "voice identities",
        "enroll voice",
        "speaker identification",
        "voiceprint",
        "voice print",
    ),
    "crown": (
        "crown",
        "eeg",
        "biosignal",
        "neuro",
        "brain wave",
        "brainwave",
        "trigger pack",
        "trigger packs",
    ),
    "game": (
        "game install",
        "game available",
        "ensure game",
    ),
    "game_session": (
        "game session",
        "game sessions",
        "game streaming",
        "stream a game",
        "cyberpunk session",
    ),
    "gamestream": (
        "gamestream status",
        "game stream status",
    ),
    "grid": (
        "grid status",
    ),
    "screenstream": (
        "screenstream status",
        "screen stream status",
    ),
    "psykyo": (
        "psykyo",
        "psykyo benchmark",
        "vlm consensus",
    ),
    "images": (
        "generate image",
        "generate an image",
        "image generation",
        "ocr",
        "object detection",
    ),
    "vision": (
        "analyze image",
        "describe local image",
        "vision analyze",
    ),
    "files": (
        "read file",
        "read a file",
        "open file",
    ),
    "web": (
        "web search",
        "search the web",
        "fetch url",
        "fetch a url",
    ),
    "http": (
        "http request",
        "http get",
        "http post",
    ),
    "search": (
        "search local",
        "local search",
    ),
    "services": (
        "warden service",
        "warden services",
        "restart service",
        "enable service",
        "disable service",
        "maintenance window",
    ),
    "service_health": (
        "service health",
        "services health",
        "service status",
        "services healthy",
        "are services healthy",
        "are the services healthy",
        "is the service healthy",
        "health of the services",
        "healthy services",
    ),
    "jobs": (
        "active jobs",
        "running jobs",
        "current jobs",
        "cancel job",
        "cancel a job",
        "job list",
        "list jobs",
        "what is running",
        "what's running",
        "running right now",
        "anything running",
        "in flight",
        "what is in flight",
        "jobs in flight",
    ),
    "human": (
        "human approval",
        "request approval",
        "notify operator",
        "telegram poll",
    ),
    "time": (
        "what time",
        "current time",
        "cluster time",
        "what date",
        "current date",
    ),
    "math": (
        "calculate",
        "calculator",
        "compute the value",
    ),
    "api_keys": (
        "api key",
        "api keys",
        "api keys status",
    ),
    "oracle": (
        "oracle",
        "ask the oracle",
        "oracle plan",
        "planner",
    ),
    "identity": (
        "identity anchor",
        "who are you",
        "verify identity",
    ),
    "ethics": (
        "great lense",
        "evaluate action",
        "ethics check",
    ),
    "sessions": (
        "session list",
        "list sessions",
        "active sessions",
    ),
    "chat": (
        "send a chat",
        "post chat",
    ),
    "doctrine": (
        "tmr canon",
        "machine religion",
        "doctrine",
        "deus acuo",
    ),
    "nibbles": (
        "nibbles",
        "nibbles dry run",
    ),
    "hermes": (
        "hermes version",
        "hermes update",
        "hermes releases",
        "hermes tool",
        "hermes tools",
    ),
    "runtime": (
        "dependency status",
        "deps status",
        "venv status",
        "runtime status",
    ),
    "double_agent": (
        "depth lobe",
        "double agent",
        "dispatch job",
        "background job",
    ),
    "desktop": (
        "desktop screenshot",
        "screenshot",
        "screen capture",
        "desktop control",
        "click",
        "type into",
    ),
}


def keywords_for_toolbox(toolbox: str) -> tuple[str, ...]:
    """Lower-case keyword/phrase hints for a toolbox. Empty tuple for
    unknown toolboxes."""
    return TOOLBOX_KEYWORDS.get(canonical_toolbox(toolbox), ())


# ---------------------------------------------------------------------------
# Hermes toolset hints for the Depth Lobe (Phase E)
# ---------------------------------------------------------------------------
#
# The Depth Lobe worker is a Hermes agent. Hermes groups ITS OWN tools
# (terminal, file, browser, web, vision, image, tts, skills, cron) into
# "toolsets" — distinct from the Quartermaster's HiveMind "toolboxes".
# When a request clearly maps to a narrow set of Hermes capabilities,
# we can construct the worker with only those toolsets, trimming the
# per-job tool context.
#
# This mapping is a hint only. ``None`` means no mapping / not admitted.
# It is never full-catalog authority and never a default tool grant.

# Hermes legacy toolset group names come from the installed Hermes
# ``model_tools._LEGACY_TOOLSET_MAP``. MS4 also contributes the
# plugin-registered ``mcp-hivemind`` toolset through ``ms4_consciousness``.
_HERMES_INTENT_TOOLSETS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # (query keyword/phrases, hermes toolsets to enable)
    (("browse", "navigate to", "click the", "scroll the page", "web page", "open the site"),
     ("browser_tools", "web_tools")),
    (("search the web", "web search", "look up online", "google "),
     ("web_tools",)),
    (("generate an image", "generate image", "make an image", "create an image", "draw "),
     ("image_tools",)),
)

# If ANY of these broadening signals appear, never narrow — the request
# likely needs terminal / file / multi-tool work.
_HIVEMIND_TOOLSET = "mcp-hivemind"

_HIVEMIND_TOOLSET_HINTS: tuple[str, ...] = (
    "hivemind",
    "cluster summary",
    "cluster load",
    "cluster nodes",
    "service health",
    "services healthy",
    "capability matrix",
    "active jobs",
    "running jobs",
    "current jobs",
    "job list",
    "list jobs",
    "what is running",
    "what's running",
    "running right now",
    "anything running",
    "what gpus",
    "list gpus",
    "free gpus",
    "available gpus",
    "gpu availability",
    "model catalog",
    "list models",
    "what models",
    "vm list",
    "list vms",
    "virtual machines",
    "mcp tools",
    "toolbox",
    "toolboxes",
)

_HIVEMIND_TOOLSET_BROADENING_SIGNALS: tuple[str, ...] = (
    "terminal",
    "shell",
    "command",
    "install",
    "build",
    "file",
    "write ",
    "edit ",
    "patch",
    "code",
    "script",
    "debug",
    "deploy",
    "and then",
    "and also",
    "everything",
)

_BROADENING_SIGNALS: tuple[str, ...] = (
    "terminal", "run ", "execute", "shell", "command", "install", "build",
    "file", "read ", "write ", "edit ", "patch", "code", "script", "debug",
    "deploy", "and then", "and also", "step by step", "everything",
)


def hermes_toolsets_for_query(query: str) -> list[str] | None:
    """Hint only. ``None`` means no mapping, never full-catalog authority.

    The runner must not treat this return as admission. Only the two
    current exact Oracle wrapper toolset names may be hinted when those
    wrapper names are already present in the query. Pure / deterministic
    / no I/O."""
    if not query:
        return None
    lower = query.casefold()
    if "hivemind_exact_gated" in lower:
        return ["mcp-hivemind-exact-gated"]
    if "hivemind_exact_read" in lower:
        return ["mcp-hivemind-exact-read"]
    return None


def all_known_toolboxes() -> Iterable[str]:
    """All toolboxes that have at least one keyword OR appear in a
    tool-shed cluster. Stable iteration order."""
    seen: list[str] = []
    for cluster_members in TOOL_SHED_CLUSTERS.values():
        for tb in cluster_members:
            if tb not in seen:
                seen.append(tb)
    for tb in TOOLBOX_KEYWORDS.keys():
        if tb not in seen:
            seen.append(tb)
    return seen
