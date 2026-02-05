# Mistral Vibe CLI v2.0.2 Setup (vLLM v0.15.1 backend)

## 1) Install Mistral Vibe CLI v2.0.2

```bash
uv tool install mistral-vibe
```

## 2) Create global config

```bash
mkdir -p ~/.vibe
```

### `~/.vibe/config.toml`

```toml
auto_compact_threshold = 82000

active_model = "devstral-local"

[[providers]]
name = "vllm-devstral"
api_base = "http://172.17.0.1:8000/v1"
api_key_env_var = "VLLM_API_KEY"
api_style = "openai"
backend = "generic"

[[models]]
name = "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit"
provider = "vllm-devstral"
alias = "devstral-local"
temperature = 0.15
input_price = 0.0
output_price = 0.0
```

### `~/.vibe/.env`

```env
VLLM_API_KEY=dummy
```

## 3) Why `auto_compact_threshold = 82000`

The vLLM server is configured with `max-model-len: 90000` (90K tokens). We set Vibe's compaction threshold to 82,000 (leaving ~8K tokens of headroom) so Vibe triggers automatic context summarization **before** hitting the hard ceiling. Without this, long agentic sessions would exceed the context limit and get rejected by the server.

## 4) Verify

```bash
curl -s http://172.17.0.1:8000/v1/models | jq .
cd /path/to/your/repo
vibe
```

Inside Vibe, use `/config` to confirm the active model and provider.
