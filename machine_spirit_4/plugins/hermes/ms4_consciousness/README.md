# ms4_consciousness Hermes Plugin

This directory is the MS4-owned source home for the Hermes plugin.

Operational installs place the plugin into the Hermes checkout selected by `MS4_HERMES_DIR` or the default `~/Documents/hermes-agent`:

```text
<hermes-agent>/plugins/ms4_consciousness/
```

The plugin must:

- Verify the active spirit through the MS3 sidecar on session start.
- Inject MS4/MS3 state before LLM calls.
- Gate effectful tool calls through MS3 Great Lense evaluation.
- Strip and persist advisory `<psyche>` blocks from model output.
- Fail closed when MS3 or validated state is unavailable.

Do not let a lobe or model-generated self-report become authoritative state.
