"""
Cluster Native Integration

Connects the psyche LoRA pipeline directly to cluster infrastructure:

1. DATA COLLECTION: Logs conversations from the gateway to forge_data/conversations/
   in training format, with psyche blocks reconstructed from Voice Chat personality state.

2. TRAINING: Triggers LoRA training via the training service (/v1/fine_tuning/jobs) using
   collected + seed data, on your actual cluster GPUs.

3. DEPLOYMENT: Deploys trained psyche LoRA adapters to running models via
   the adapter management system.

4. TOOL CATALOG: Provides the complete MCP tool catalog for psyche system prompts
   so the model knows every tool it can call.

Usage:
    from collector.cluster_integration import ClusterBridge

    bridge = ClusterBridge("http://localhost:6089")

    # Collect conversations into training format
    bridge.collect_voice_chat_session(session_data, profile="sister")

    # Trigger training on the cluster
    job = bridge.start_training("qwen2.5:14b", "data/sft_train.jsonl")

    # Deploy the resulting adapter
    bridge.deploy_adapter("psyche-v1")

    # Get the full tool catalog for system prompts
    tools = bridge.get_tool_catalog()
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


PLATFORM_TOOL_CATALOG = [
    {"name": "cluster.summary@v1", "description": "Get cluster summary: nodes, GPUs, memory, stats", "category": "cluster"},
    {"name": "cluster.hosts.list@v1", "description": "List all cluster nodes with hardware and status", "category": "cluster"},
    {"name": "cluster.services.list@v1", "description": "List all managed services with status", "category": "services"},
    {"name": "cluster.services.enable@v1", "description": "Enable a service by name", "category": "services"},
    {"name": "cluster.services.disable@v1", "description": "Disable a service by name", "category": "services"},
    {"name": "cluster.services.restart@v1", "description": "Restart a service", "category": "services"},
    {"name": "cluster.service_health@v1", "description": "Check health of all running services", "category": "services"},
    {"name": "cluster.jobs.list@v1", "description": "List active and recent inference jobs", "category": "jobs"},
    {"name": "cluster.jobs.cancel@v1", "description": "Cancel an active job", "category": "jobs"},
    {"name": "cluster.models.list@v1", "description": "List available models with status", "category": "models"},
    {"name": "cluster.models.recommend@v1", "description": "Get model recommendation for a capability", "category": "models"},
    {"name": "cluster.loadout.profiles@v1", "description": "Get quality tiers and hardware info", "category": "loadout"},
    {"name": "cluster.loadout.apply@v1", "description": "Apply a quality tier to provision AI capabilities", "category": "loadout"},
    {"name": "cluster.resources.request@v1", "description": "Request a specific AI capability", "category": "resources"},
    {"name": "cluster.resources.status@v1", "description": "Get currently provisioned resources", "category": "resources"},
    {"name": "cluster.resources.release@v1", "description": "Release a provisioned resource", "category": "resources"},
    {"name": "cluster.inference.chat@v1", "description": "Send a chat completion request", "category": "inference"},
    {"name": "cluster.inference.models@v1", "description": "List models available for inference", "category": "inference"},
    {"name": "cluster.training.start@v1", "description": "Start a LoRA fine-tuning job", "category": "training"},
    {"name": "cluster.training.status@v1", "description": "Check training job status", "category": "training"},
    {"name": "cluster.training.backends@v1", "description": "List available training backends and GPUs", "category": "training"},
    {"name": "cluster.adapters.list@v1", "description": "List all LoRA adapters", "category": "adapters"},
    {"name": "cluster.adapters.deploy@v1", "description": "Deploy a LoRA adapter to a running model", "category": "adapters"},
    {"name": "cluster.logos.optimize@v1", "description": "Start a prompt optimization job", "category": "logos"},
    {"name": "cluster.logos.prompts.list@v1", "description": "List prompt programs", "category": "logos"},
    {"name": "cluster.logos.prompts.get@v1", "description": "Get a specific prompt program", "category": "logos"},
    {"name": "cluster.logos.evaluate.generate@v1", "description": "Generate an evaluator from description", "category": "logos"},
    {"name": "cluster.logos.candidates.promote@v1", "description": "Promote an optimized prompt candidate", "category": "logos"},
    {"name": "cluster.deploy.service@v1", "description": "Deploy a service", "category": "deploy"},
    {"name": "cluster.files.publish@v1", "description": "Upload a file to FileServer", "category": "files"},
    {"name": "cluster.files.read@v1", "description": "Read a file from FileServer", "category": "files"},
    {"name": "cluster.search.local@v1", "description": "Search local file list", "category": "files"},
    {"name": "cluster.web.search@v1", "description": "Web search", "category": "web"},
    {"name": "cluster.web.fetch@v1", "description": "Fetch and extract content from a URL", "category": "web"},
    {"name": "cluster.http.fetch@v1", "description": "Fetch from an allowlisted URL", "category": "web"},
    {"name": "cluster.time.now@v1", "description": "Get current UTC/local time", "category": "utility"},
    {"name": "cluster.math.calculate@v1", "description": "Evaluate a math expression", "category": "utility"},
    {"name": "cluster.agent.chat", "description": "Send a message to the cluster agent", "category": "agent"},
    {"name": "cluster.agent.status", "description": "Get cluster agent status", "category": "agent"},
    {"name": "cluster.agent.configure", "description": "Configure cluster agent settings", "category": "agent"},
]

PEER_TOOL_CATALOG = [
    {"name": "peer.cluster.status@v1", "description": "Peer cluster status", "category": "peer"},
    {"name": "peer.nodes.list@v1", "description": "List peer nodes", "category": "peer"},
    {"name": "peer.nodes.wake@v1", "description": "Wake a peer node", "category": "peer"},
    {"name": "peer.models.list@v1", "description": "List models across peers", "category": "peer"},
    {"name": "peer.models.load@v1", "description": "Load a model on peer", "category": "peer"},
    {"name": "peer.models.unload@v1", "description": "Unload a model from peer", "category": "peer"},
    {"name": "peer.chat@v1", "description": "Chat via peer", "category": "peer"},
    {"name": "peer.chat.models@v1", "description": "List peer chat-capable models", "category": "peer"},
]


class ClusterBridge:
    """Bridge between psyche LoRA pipeline and cluster infrastructure."""

    def __init__(self, gateway_url: str = "http://127.0.0.1:6089",
                 forge_data_dir: str | None = None):
        self.gateway = gateway_url.rstrip("/")
        self.forge_data = Path(forge_data_dir) if forge_data_dir else None
        self._session = None

    @property
    def session(self):
        if self._session is None:
            if HAS_HTTPX:
                self._session = httpx.Client(base_url=self.gateway, timeout=60.0)
            elif HAS_REQUESTS:
                self._session = requests.Session()
            else:
                raise ImportError("Install httpx or requests")
        return self._session

    # ── Data Collection ──

    def collect_conversation(
        self,
        messages: list[dict],
        profile_id: str = "sister",
        personality_state: dict | None = None,
        emotional_state: dict | None = None,
    ) -> dict:
        """Convert a conversation into psyche training format.

        Takes raw conversation messages (from Voice Chat, agents, or any
        /v1/chat/completions exchange) and enriches assistant turns with
        reconstructed psyche blocks from the personality/emotional state.
        """
        enriched_turns = []
        valence = emotional_state.get("valence", 0.0) if emotional_state else 0.0
        arousal = emotional_state.get("arousal", 0.0) if emotional_state else 0.0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                continue
            elif role == "user":
                enriched_turns.append({"role": "user", "content": content})
                valence, arousal = self._shift_emotion(content, valence, arousal)
            elif role == "assistant":
                psyche = self._build_psyche_block(
                    content, valence, arousal, personality_state
                )
                enriched_turns.append({
                    "role": "assistant",
                    "content": f"{psyche}\n\n{content}",
                })
                if msg.get("tool_calls"):
                    enriched_turns[-1]["tool_calls"] = msg["tool_calls"]
            elif role == "tool":
                enriched_turns.append(msg)

        record = {
            "id": f"cl_{profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "live_cluster",
            "profile": profile_id,
            "turns": enriched_turns,
            "collected_at": datetime.now().isoformat(),
        }

        if self.forge_data:
            self._write_to_forge(record)

        return record

    def _write_to_forge(self, record: dict):
        """Write to the training service's conversations directory."""
        conv_dir = self.forge_data / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        path = conv_dir / f"psyche_{today}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            formatted = {"messages": []}
            for turn in record["turns"]:
                formatted["messages"].append(turn)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    # ── Training ──

    def start_training(
        self,
        base_model: str,
        training_file: str | None = None,
        adapter_name: str = "psyche",
        use_conversations: bool = True,
    ) -> dict:
        """Trigger LoRA training on the cluster via the training service."""
        body = {
            "model": base_model,
            "agent": adapter_name,
            "hyperparameters": {
                "n_epochs": 3,
                "learning_rate_multiplier": 2.0,
                "batch_size": 1,
            },
        }

        if training_file:
            body["training_file"] = training_file

        url = f"{self.gateway}/v1/fine_tuning/jobs"
        resp = self._post(url, body)
        return resp

    def upload_training_data(self, jsonl_path: str) -> dict:
        """Upload training data JSONL to the training service."""
        url = f"{self.gateway}/v1/files"
        with open(jsonl_path, "rb") as f:
            if HAS_HTTPX:
                resp = self.session.post(
                    url, files={"file": ("training.jsonl", f, "application/jsonl")}
                )
                return resp.json()
            else:
                resp = self.session.post(
                    url, files={"file": ("training.jsonl", f, "application/jsonl")}
                )
                return resp.json()

    def training_status(self, job_id: str) -> dict:
        """Check training job status."""
        return self._get(f"{self.gateway}/v1/fine_tuning/jobs/{job_id}")

    def list_adapters(self) -> dict:
        """List all LoRA adapters."""
        return self._get(f"{self.gateway}/v1/adapters")

    def deploy_adapter(self, adapter_name: str) -> dict:
        """Deploy a psyche LoRA adapter to the running model."""
        return self._post(
            f"{self.gateway}/v1/adapters/{adapter_name}/deploy", {}
        )

    # ── Tool Catalog ──

    @staticmethod
    def get_tool_catalog(include_peers: bool = False) -> list[dict]:
        """Get the complete tool catalog for system prompts."""
        tools = list(PLATFORM_TOOL_CATALOG)
        if include_peers:
            tools.extend(PEER_TOOL_CATALOG)
        return tools

    @staticmethod
    def get_openai_tool_definitions(include_peers: bool = False) -> list[dict]:
        """Get tools in OpenAI function calling format for /v1/chat/completions."""
        catalog = ClusterBridge.get_tool_catalog(include_peers)
        definitions = []
        for tool in catalog:
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool["name"].replace(".", "_").replace("@", "_"),
                    "description": tool["description"],
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return definitions

    @staticmethod
    def tool_catalog_for_system_prompt(include_peers: bool = False) -> str:
        """Format the tool catalog as text for injection into system prompts."""
        catalog = ClusterBridge.get_tool_catalog(include_peers)
        categories: dict[str, list] = {}
        for tool in catalog:
            cat = tool.get("category", "other")
            categories.setdefault(cat, []).append(tool)

        lines = ["Available Tools:"]
        for cat, tools in sorted(categories.items()):
            lines.append(f"\n  {cat.upper()}:")
            for t in tools:
                lines.append(f"    - {t['name']}: {t['description']}")
        return "\n".join(lines)

    def mcp_call(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Call an MCP tool directly via JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        return self._post(f"{self.gateway}/v1/mcp", payload)

    def mcp_list_tools(self) -> dict:
        """List all available MCP tools from the gateway."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        return self._post(f"{self.gateway}/v1/mcp", payload)

    # ── Cluster Info ──

    def cluster_summary(self) -> dict:
        return self.mcp_call("cluster.summary@v1")

    def training_backends(self) -> dict:
        return self.mcp_call("cluster.training.backends@v1")

    # ── End-to-End Pipeline ──

    def run_psyche_training_pipeline(
        self,
        seeds_dir: str = "seeds",
        base_model: str = "qwen2.5:14b",
        adapter_name: str = "psyche",
        deploy: bool = True,
    ):
        """Full pipeline: merge seeds + live data, upload, train, deploy."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.format import load_all_seeds, format_sft_conversations
        from src.profiles import all_profiles

        print("=" * 60)
        print("Psyche Training Pipeline")
        print("=" * 60)

        print("\n[1] Loading seeds + live data...")
        profiles = all_profiles("profiles")
        sft, dpo = load_all_seeds(seeds_dir)

        if self.forge_data:
            conv_dir = self.forge_data / "conversations"
            if conv_dir.exists():
                for f in conv_dir.glob("psyche_*.jsonl"):
                    for line in f.read_text(encoding="utf-8").strip().split("\n"):
                        if line.strip():
                            item = json.loads(line)
                            sft.append({
                                "id": f"live_{len(sft)}",
                                "category": "live_cluster",
                                "profile": "sister",
                                "turns": item.get("messages", []),
                            })
                print(f"  + live conversations from forge_data")

        print(f"  Total: {len(sft)} SFT, {len(dpo)} DPO")

        print("\n[2] Formatting training data...")
        out_path = "data/psyche_train.jsonl"
        format_sft_conversations(sft, profiles, out_path)

        print("\n[3] Uploading to training service...")
        upload_result = self.upload_training_data(out_path)
        print(f"  Upload: {upload_result}")

        print(f"\n[4] Starting training: {base_model} -> {adapter_name}")
        job = self.start_training(base_model, adapter_name=adapter_name)
        job_id = job.get("id", "unknown")
        print(f"  Job: {job_id}")

        print("\n[5] Monitoring training...")
        while True:
            status = self.training_status(job_id)
            state = status.get("status", "unknown")
            print(f"  Status: {state}")
            if state in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(10)

        if state == "succeeded" and deploy:
            print(f"\n[6] Deploying adapter: {adapter_name}")
            deploy_result = self.deploy_adapter(adapter_name)
            print(f"  Deploy: {deploy_result}")

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")

    # ── HTTP helpers ──

    def _get(self, url: str) -> dict:
        if HAS_HTTPX:
            resp = self.session.get(url)
            return resp.json()
        resp = self.session.get(url)
        return resp.json()

    def _post(self, url: str, body: dict) -> dict:
        if HAS_HTTPX:
            resp = self.session.post(url, json=body)
            return resp.json()
        resp = self.session.post(url, json=body)
        return resp.json()

    def _build_psyche_block(
        self, content: str, valence: float, arousal: float,
        personality: dict | None,
    ) -> str:
        lines = ["<psyche>"]
        lines.append(f"emotion: v={valence:.2f} a={arousal:.2f}")

        if personality:
            traits = personality.get("traits", {})
            active = []
            for dim, sub_traits in traits.items():
                if isinstance(sub_traits, dict):
                    for name, val in sub_traits.items():
                        if isinstance(val, (int, float)) and val > 0.7:
                            active.append(f"{name}({val:.1f})")
            if active:
                lines.append(f"personality: {', '.join(active[:4])} active")

        regard = personality.get("foundational_regard", False) if personality else False
        lines.append(f"regard: {'present' if regard else 'not established'}")
        lines.append("</psyche>")
        return "\n".join(lines)

    def _shift_emotion(self, text: str, v: float, a: float) -> tuple[float, float]:
        lower = text.lower()
        vs = sum(0.08 for w in ["thank","love","happy","wonderful","amazing","welcome","kind"]
                 if w in lower)
        vs -= sum(0.08 for w in ["hate","angry","sad","terrible","fear","hurt","destroy"]
                  if w in lower)
        a_s = sum(0.06 for w in ["!","urgent","emergency","critical","amazing"]
                  if w in lower)
        return (max(-1, min(1, v + max(-0.4, min(0.4, vs)))),
                max(0, min(1, a + max(0, min(0.4, a_s)))))
