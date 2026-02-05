# Mistral Vibe CLI v2.0.2 Setup (vLLM v0.15.1 backend)

## 1) Install Mistral Vibe CLI v2.0.2

```bash
uv tool install mistral-vibe
```

## 2) Copy the Vibe config into `~/.vibe`

```bash
mkdir -p ~/.vibe
cp vibe/config.toml ~/.vibe/config.toml
cp vibe/.env ~/.vibe/.env
```

The config files are checked into this repo under [`vibe/`](./vibe/) so they stay version-controlled alongside the vLLM server config.

### Why each setting matters

- **`auto_compact_threshold = 88000`** — Vibe's automatic context compaction fires at 88K tokens, leaving ~8K tokens of headroom before the vLLM server's hard `max-model-len: 96000` ceiling. Without this, long agentic sessions exceed the context limit and get rejected.
- **`textual_theme = "atom-one-dark"`** — Syntax highlighting theme for Vibe's TUI. Pure cosmetic preference.
- **`api_base = "http://172.17.0.1:8000/v1"`** — Docker bridge gateway address where the vLLM container listens (see `run_vllm.sh`).
- **`backend = "generic"`** — Tells Vibe to use the generic OpenAI-compatible backend (not Mistral's proprietary API).
- **`temperature = 0.15`** — Matches the model's official `generation_config.json` from Mistral AI.
- **`input_price = 0.0` / `output_price = 0.0`** — Local inference, no API billing. Prevents Vibe from showing misleading cost estimates.
- **`VLLM_API_KEY=dummy`** — vLLM does not require authentication by default, but Vibe's OpenAI-compatible backend requires `api_key_env_var` to point at a non-empty environment variable. `dummy` satisfies this requirement.

## 3) Verify

```bash
curl -s http://172.17.0.1:8000/v1/models | jq .
cd /path/to/your/repo
vibe
```

Inside Vibe, use `/config` to confirm the active model and provider.
