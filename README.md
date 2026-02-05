# devstral_vllm_config

> Production-ready vLLM v0.15.1 server configuration for **Mistral Devstral Small 2 24B Instruct 2512** (AWQ 4-bit quantization) on **Ubuntu 24.04 LTS + NVIDIA GeForce RTX 5090** (32 GiB GDDR7 VRAM) — tuned for single-user agentic coding with **Mistral Vibe CLI v2.0.2**.
> Repo link: https://github.com/BigBIueWhale/personal_server

---

## What this is

This project pins **server-side defaults** for vLLM v0.15.1 so that **Mistral Vibe CLI v2.0.2** (or any OpenAI-compatible client) doesn't have to pass a dozen sampling knobs. It's built for **my specific machine**:

- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0, 32 GiB GDDR7 VRAM)
- **Driver:** NVIDIA Driver 580.xx (open kernel module)
- **CUDA:** CUDA 13.0
- **OS:** Ubuntu 24.04 LTS
- **Docker:** Docker Engine with NVIDIA Container Toolkit (installed via https://github.com/BigBIueWhale/personal_server)

With this exact combo, **Mistral Devstral Small 2 24B Instruct 2512** runs at **AWQ 4-bit quantization** (~14.0 GiB text-only weights) with **full-precision BF16 KV cache**, achieving **90K token context** on a single NVIDIA GeForce RTX 5090.

### Why vLLM v0.15.1 instead of Ollama

This project deliberately uses **vLLM v0.15.1** (released February 4, 2026) instead of Ollama because:

1. **Tool calling support:** vLLM's `--tool-call-parser mistral` flag converts Devstral's `[TOOL_CALLS]` tokens into OpenAI-compatible `tool_calls` objects. Without this, Mistral Vibe CLI v2.0.2 **cannot perform any agentic actions** (file reads, writes, terminal commands).

2. **Server-side generation defaults:** vLLM's `override-generation-config` lets us pin temperature and `max_new_tokens` on the server. Clients inherit these without needing Modelfile hacks. This is critical because vLLM's default `max_tokens` is only 16 — without the override, every response would be silently truncated.

3. **AWQ hardware-accelerated inference:** vLLM auto-detects the compressed-tensors quantization format from the model's `config.json` and runs hardware-accelerated 4-bit inference on Blackwell (compute capability 12.0).

> **Why custom config is needed (philosophy):** Pretty much all models **require very specific runtime flags** to behave correctly. For Devstral Small 2, using the wrong defaults can make the model **quietly underperform ("silently stupid")** without obvious errors. This repo bakes in the correct settings — notably `tool-call-parser: "mistral"`, `limit-mm-per-prompt` to disable the unused vision encoder, and `max_new_tokens` to prevent silent truncation — plus Mistral AI's recommended temperature. Clients can still override per-request, but the server remains the single source of truth for everything else.

### Why Mistral Devstral Small 2 24B (not the original Devstral)

This guide is for **Mistral Devstral Small 2 24B Instruct 2512** (`mistralai/Devstral-Small-2-24B-Instruct-2512`, released December 2025), **not** the original Devstral (May 2025). Key differences:

- **Architecture:** `ministral3` (40 layers, 32 attention heads, 8 KV heads via Grouped Query Attention, 128 head dimension, 5120 hidden size, 131072 vocabulary)
- **Max context:** 256K tokens (262,144) via YaRN rope scaling (the original was 128K)
- **Native precision:** FP8 (`e4m3`) — the model ships in FP8, not FP16/BF16
- **Tool calling:** First-class `[TOOL_CALLS]` token support for agentic workflows
- **Hugging Face ID:** `mistralai/Devstral-Small-2-24B-Instruct-2512` (the `2512` suffix indicates the December 2025 release date)

---

## Contents

- **[`config.yaml`](./config.yaml)** — vLLM v0.15.1 server config. Sets the model, networking, tool calling flags, text-only mode, and server-side generation defaults (temperature 0.15, max_new_tokens). Clients inherit these unless they explicitly override a field.
- **[`run_vllm.sh`](./run_vllm.sh)** — one-shot "install" script that pulls the `vllm/vllm-openai:v0.15.1` Docker image and creates a persistent Docker container named `devstral_vllm` with `--restart unless-stopped`. Run it once; use `docker start devstral_vllm` on reboots.
- **[`VIBE_SETUP.md`](./VIBE_SETUP.md)** — Mistral Vibe CLI v2.0.2 configuration, pointing at the vLLM server.
- **[`UNINSTALL.md`](./UNINSTALL.md)** — how to stop/remove the container and (optionally) delete the Docker image and Hugging Face cache.

---

## Quick start

1. **Optional:** export your Hugging Face token (for first-time download of model weights from Hugging Face Hub):
   ```bash
   export HF_TOKEN=hf_************************
   ```

2. **Install / create the server container** (one time):
   ```bash
   chmod +x ./run_vllm.sh
   ./run_vllm.sh
   ```

3. **Test the endpoint:**
   ```bash
   # From the host:
   curl -s http://172.17.0.1:8000/v1/models | jq .
   ```

4. **Hook up Mistral Vibe CLI v2.0.2:** see **[VIBE_SETUP.md](./VIBE_SETUP.md)**.

---

## VRAM memory layout on NVIDIA GeForce RTX 5090 (32 GiB GDDR7)

```
AWQ 4-bit text weights (no vision) ... ~14.0 GiB
CUDA context + activations ........... ~ 1.7 GiB
BF16 KV cache (90K tokens) ........... ~14.1 GiB
Headroom ............................. ~ 0.6 GiB
─────────────────────────────────────────────────
Total at 0.95 gpu-memory-utilization . ~30.4 GiB  (of 32 GiB)
```

### Why full-precision KV cache (not FP8)

The KV cache stays in **full precision (BF16)** to preserve attention quality over long contexts. It is enough that we already sacrifice quality with quantized weights — quantizing the KV cache (FP8) on top of that compounds precision loss in ways that hurt the model's ability to attend accurately to context thousands of tokens back. Full-precision KV is non-negotiable when context fidelity matters.

FP8 KV cache (80 KiB/token instead of 160 KiB/token) would roughly double the maximum context to ~180K tokens, but at the cost of attention precision. This project prioritizes context quality over context length.

### Why the vision encoder is disabled

Devstral Small 2 inherits a Pixtral vision encoder from its Mistral Small 3.1 24B base model, but Devstral is a text/code-only model — the vision encoder is unused dead weight. The `limit-mm-per-prompt: '{"image": 0}'` flag activates vLLM's text-only mode, which [skips loading the vision tower and multi-modal projector](https://github.com/vllm-project/vllm/pull/22299) (~1 GiB BF16), freeing that VRAM for KV cache.

### KV cache math (per token)

Mistral Devstral Small 2 24B has 40 layers with 8 KV heads and 128 head dimension:

**BF16 KV:** `2 (K+V) x 40 layers x 8 heads x 128 dim x 2 bytes = 163,840 bytes/token (~160 KiB)`

At BF16, 90,000 tokens of KV cache = `90,000 x 160 KiB = ~14.1 GiB`. This fits alongside the ~14.0 GiB AWQ text-only weights and ~1.7 GiB overhead within the 30.4 GiB budget (0.95 x 32 GiB).

### Why 90K and not 256K

The model supports up to 256K tokens via YaRN rope scaling, but with full-precision BF16 KV cache:
- 256K tokens of BF16 KV cache alone = ~39.1 GiB (exceeds 32 GiB VRAM)
- Even 128K tokens of BF16 KV cache = ~19.5 GiB, plus ~14.0 GiB weights = ~33.5 GiB (exceeds 32 GiB)

**90,000 tokens is the ceiling** for full-precision BF16 KV cache + AWQ 4-bit text-only weights at 0.95 gpu-memory-utilization on 32 GiB.

---

## Why these settings?

### Model: `cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`

Community AWQ 4-bit quantization of `mistralai/Devstral-Small-2-24B-Instruct-2512`, using compressed-tensors format with 4-bit symmetric integer quantization (group size 32). vLLM auto-detects the quantization from the model's `config.json`. The ~16 GiB on-disk model includes the vision encoder (BF16) and lm_head (BF16) alongside the 4-bit text layers; with text-only mode, only ~14 GiB loads into VRAM.

### Tool calling: `enable-auto-tool-choice` + `tool-call-parser: "mistral"`

**Without these two flags, Mistral Vibe CLI v2.0.2 cannot perform any agentic actions.** The model generates `[TOOL_CALLS]` tokens that vLLM's `MistralToolParser` converts into OpenAI-compatible `tool_calls` objects in the streaming response. Mistral Vibe CLI v2.0.2 uses these `tool_calls` to execute file reads, file writes, and terminal commands.

**Known issue (fixed):** vLLM versions before v0.13.0 had a tool call parsing bug that caused "Extra inputs are not permitted" errors with Mistral Vibe CLI (see [mistral-vibe issue #70](https://github.com/mistralai/mistral-vibe/issues/70)). vLLM v0.15.1 includes this fix.

### Sampling: temperature 0.15

Mistral AI's official `generation_config.json` for Devstral Small 2 24B Instruct 2512 specifies **temperature 0.15** with `do_sample: true`. This is the only sampling parameter Mistral explicitly sets — no `top_p`, `top_k`, `min_p`, or penalty values are specified, meaning vLLM's defaults apply (top_p: 1.0, top_k: -1 disabled, min_p: 0.0 disabled, all penalties: neutral).

The AWQ repo's `generation_config.json` is incomplete (missing temperature), which is why `config.yaml` uses `generation-config: "vllm"` with an explicit override instead of `generation-config: "auto"`.

**Note on discrepancy:** Mistral's blog post announcing Devstral 2 mentions temperature 0.2, while the generation_config.json says 0.15. A [community discussion](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512/discussions/9) raised this; Mistral resolved it by updating the generation_config.json to 0.15 as the authoritative default.

### `max_new_tokens: 90000`

**CRITICAL.** vLLM's default `max_tokens` is only **16 tokens** if not overridden. Without this setting, the model would silently truncate every response at 16 tokens. Setting it to 90,000 matches the `max-model-len` context budget so the model is never artificially cut off.

### Single concurrent request: `max-num-seqs: 1`

Dedicates the full KV cache budget to one Mistral Vibe CLI v2.0.2 session. This is the intended single-user agentic coding workflow — one human, one GPU, maximum context.

---

## Operating the server (start / stop / pause)

### Check status

```bash
docker ps
curl -s http://172.17.0.1:8000/v1/models | jq .
docker logs -f devstral_vllm
```

### Stop (frees VRAM)

> Use **stop** to fully release GPU memory.

```bash
docker stop devstral_vllm
```

Verify with:

```bash
nvidia-smi
```

### Start (loads model and uses VRAM again)

```bash
docker start -a devstral_vllm   # attach logs
# or
docker start devstral_vllm
```

### Restart

```bash
docker restart devstral_vllm
```

### Pause vs Stop

* `docker pause` **does not free VRAM** (the process is frozen but GPU memory stays allocated).
* `docker stop` **does free VRAM** (the process exits and releases the GPU).

```bash
docker pause devstral_vllm
docker unpause devstral_vllm
```

### Auto-restart policy

The container is created with `--restart unless-stopped`.

* Disable auto-restart:

```bash
docker update --restart=no devstral_vllm
```

* Re-enable:

```bash
docker update --restart=unless-stopped devstral_vllm
```

### Remove and recreate (if you want a clean slate)

```bash
docker stop devstral_vllm || true
docker rm devstral_vllm || true
./run_vllm.sh
```

---

## Compatibility notes

- Uses **`vllm/vllm-openai:v0.15.1`** Docker image (released February 4, 2026). This image includes native `ministral3` architecture support (added in vLLM v0.12.0), the Mistral tool call parsing fix (added in vLLM v0.13.0), and NVIDIA Blackwell (compute capability 12.0) compatibility.
- The **NVIDIA vLLM container** (`nvcr.io/nvidia/vllm:25.09-py3`) from NVIDIA NGC does **not** support the `ministral3` architecture used by Devstral Small 2. That is why this project uses the upstream `vllm/vllm-openai` image instead.
- Assumes you already installed the **NVIDIA Container Toolkit** and your NVIDIA Driver 580.xx (open kernel module) exposes the GPU inside containers via `--gpus all`. This is set up by https://github.com/BigBIueWhale/personal_server.
- **FlashInfer attention backend** is used automatically on Blackwell (compute capability 12.0). Flash Attention (Dao-AILab) only supports Ampere, Ada, and Hopper — vLLM detects Blackwell and selects FlashInfer instead. No extra flags required.
- **First container start is slow on Blackwell.** The upstream PyTorch 2.9.x wheels do not include precompiled sm_120 (Blackwell) kernels. PyTorch JIT-compiles them from PTX on the first run, adding several extra minutes to the initial `docker start`. Subsequent starts reuse the cached compilation and are fast.
- **Chunked prefill** is always enabled in vLLM V1 (v0.8.0+) and cannot be disabled. It is mathematically equivalent to processing the full prefill at once — no quality impact.
- **Prefix caching** is enabled by default in vLLM V1. It reuses cached KV blocks for repeated prefixes (e.g., the system prompt across Vibe requests), dramatically reducing time-to-first-token. No quality impact.

---

## Networking

The containerized vLLM v0.15.1 server listens on `0.0.0.0:8000` **inside the container**, and the Docker publish in `run_vllm.sh` maps it to **`172.17.0.1:8000` on the host** (the Docker bridge gateway). This keeps the API reachable from the host and from other containers (e.g., via `host.docker.internal`) while not exposing it on your host's primary network interfaces.

If you need LAN exposure (e.g., to query from another machine), change `BIND="172.17.0.1"` to `BIND="0.0.0.0"` in `run_vllm.sh`.

---

## Version matrix

| Component | Version | Notes |
|---|---|---|
| Mistral Devstral Small 2 24B Instruct 2512 | `mistralai/Devstral-Small-2-24B-Instruct-2512` | December 2025 release |
| AWQ 4-bit quantization | `cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit` | Community quantization, compressed-tensors format |
| vLLM | v0.15.1 (February 4, 2026) | Docker `vllm/vllm-openai:v0.15.1` |
| Mistral Vibe CLI | v2.0.2 (January 30, 2026) | Installed via `uv tool install mistral-vibe` |
| mistral_common | >= 1.8.6 (latest: 1.9.0) | Required by vLLM for Mistral tool parsing |
| NVIDIA GeForce RTX 5090 | Blackwell, compute capability 12.0 | 32 GiB GDDR7 VRAM |
| NVIDIA Driver | 580.xx (open kernel module) | Installed via personal_server |
| CUDA (host) | 13.0 | Installed via personal_server |
| CUDA (container) | 12.9 | Bundled in vLLM v0.15.1 Docker image |
| Ubuntu | 24.04 LTS | Host operating system |
| Docker Engine | With NVIDIA Container Toolkit | Installed via personal_server |

---

## Instructions (verbatim)

> Use /tmp to clone repos. Please gain inspiration from https://github.com/BigBIueWhale/nvidia_nemotron_vllm_config and also from https://github.com/BigBIueWhale/mistral_vibe_setup and from https://github.com/BigBIueWhale/personal_server and from https://github.com/BigBIueWhale/ollama_load_balancer. Create the entire project to my standard (I created all those projects). But first- create a document with my instructions verbatim, within this folder. Go!

> Oh, did I mention to research extensively online **a lot** to get everything right, and not miss anything that you need to do? The task: to create information-generous, and yet full, opinionated well-researched and information-dense, nothing is obvious setup for my PC for Mistral Devstral 2 Small (24b) 4-bit at its maximum context length on my 32 GiB VRAM GPU, for serving Mistral Vibe CLI. Add that to the instructions, then reevaluate your plan.

> Please base everything on what I actually have installed on my PC according to https://github.com/BigBIueWhale/personal_server/. Also, always use full names of things. For example: Samsung Galaxy S25 smartphone instead of Galaxy S25.

> And also, research every aspect online to be sure about what needs to be done, instead of giving the reader options.

> Add these instructions to the readme as well

> Note: I'm talking about **specifically** devstral 2. Not anything else. Not devstral original. Also, make sure everything throughout the entire guide is versioned in addition to using full names. Add that to the instructions as well. Don't make up versions- make sure to research about that as well.

---

## License

This repo contains only configuration and scripts I wrote for my own server layout. Model weights are **not** distributed here; they are fetched from Hugging Face Hub under their respective license terms.
