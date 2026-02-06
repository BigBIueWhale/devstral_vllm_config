# Bug Report: Mistral Devstral Small 2 24B AWQ 4-bit produces repetitive gibberish on vLLM v0.15.1

## Summary

[`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit) served via the [`vllm/vllm-openai:v0.15.1`](https://hub.docker.com/r/vllm/vllm-openai/tags) Docker image produces coherent short responses but degenerates into repetitive gibberish on longer outputs. The model echoes fragments of the prompt, repeats import statements in loops, and eventually emits streams of punctuation and disconnected words. The issue is reproducible 100% of the time on prompts that require more than ~50-100 tokens of output.

The root cause is that vLLM v0.15.1 does not have the `Ministral3ForCausalLM` text backbone class in its model registry. The workarounds required to load the AWQ model at all (patching `text_config.model_type` from `"ministral3"` to `"mistral"` to satisfy transformers v4.57.6) trigger a Pixtral-12B compatibility special case in `Mistral3ForConditionalGeneration.__init__` that forces the wrong text backbone class (`MistralForCausalLM`, which inherits from `LlamaForCausalLM`).

**No alternative 4-bit quantization of Mistral Devstral Small 2 24B works on vLLM v0.15.1.** All quantized variants on HuggingFace share the same fundamental architecture routing problem. The only confirmed working path on vLLM v0.15.1 is the official FP8 model ([`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)) loaded via the Mistral-native config path (`config-format: "mistral"`, `load-format: "mistral"`, `tokenizer-mode: "mistral"`), which costs ~11 GiB more VRAM for weights and halves the available KV cache context window.

## User's original report

> There's only one issue. The model seems to output gibberish.

> I think it might have to do with the tool calling syntax or configuration

## Environment

- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell architecture, compute capability 12.0, 32 GiB GDDR7 VRAM)
- **Driver:** NVIDIA Driver 580.105.08 (open kernel module)
- **CUDA:** 13.0
- **OS:** Ubuntu 24.04 LTS
- **Docker image:** [`vllm/vllm-openai:v0.15.1`](https://hub.docker.com/r/vllm/vllm-openai/tags) (released February 4, 2026)
- **transformers version inside container:** 4.57.6 (vLLM v0.15.1 is pinned to `transformers >= 4.56.0, < 5`)
- **Model:** [`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit) at revision `da6366ed3bb6d5207c6544ede10df31e4082e027`

## Config files involved

- `config.yaml` -- vLLM server config (uses `config-format: "hf"`, `load-format: "safetensors"`, `hf-config-path: "/workspace/config_override"`)
- `config_override.json` -- patched copy of the AWQ model's `config.json` with `text_config.model_type` changed from `"ministral3"` to `"mistral"` and `text_config.llama_4_scaling` added (workaround for transformers v4.57.6 not recognizing the `ministral3` model type)

## How to reproduce

### 1. Start the server

```bash
# From the repo root (where config.yaml, config_override.json, and run_vllm.sh live):
chmod +x ./run_vllm.sh
./run_vllm.sh
# Wait ~2 minutes for model loading and CUDA graph capture
```

### 2. Verify the server is healthy (this works fine)

```bash
curl -s http://172.17.0.1:8000/v1/models | jq .
```

### 3. Short response -- works correctly

```bash
curl -s http://172.17.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit",
    "messages": [
      {"role": "user", "content": "What is 2+2? Answer in one sentence."}
    ],
    "max_tokens": 100
  }' | jq -r '.choices[0].message.content'
```

**Expected output:** `The answer to 2+2 is 4.`
**Actual output:** `The answer to 2+2 is 4.` (correct)

### 4. Tool calling -- works correctly

```bash
curl -s http://172.17.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant. You can use tools to help the user."},
      {"role": "user", "content": "Read the file /etc/hostname"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read the contents of a file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "The file path to read"}
            },
            "required": ["path"]
          }
        }
      }
    ],
    "max_tokens": 500
  }' | jq '.choices[0].message.tool_calls'
```

**Expected output:** A valid `tool_calls` array with `read_file` and `{"path": "/etc/hostname"}`
**Actual output:** Correct tool call (works fine)

### 5. Longer text response -- FAILS with gibberish

```bash
curl -s http://172.17.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit",
    "messages": [
      {"role": "system", "content": "You are a senior software engineer."},
      {"role": "user", "content": "Write a Python function that implements binary search on a sorted list. Include type hints and a docstring."}
    ],
    "max_tokens": 1000
  }' | jq -r '.choices[0].message.content'
```

**Expected output:** A coherent Python function with binary search implementation.

**Actual output (observed, verbatim):**

```
Here's a Python function that implements binary search on a sorted list. The function includes type hints and a docstring.

\`\`\`python
from typing import TypeVar, List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
from typing import List, Tuple, Generic, Union
\`\`\`
```

An earlier test (before the `llama_4_scaling` fix) produced even worse degeneration:

```
Here's a Python function that implements binary search on a sorted list, with type hints and a docstring:

\`\`\`python
from typing import TypeVar, List, Optional

T = TypeVar('T')

def binary_search(
    sorted_list: List[T],
    target: T
) -> Optional[int]:
    """
    Perform binary search on a sorted list.
    [... repeats the function signature and docstring several times ...]

\`\`\`python
def binary_search(
    a software engineer. I am a senior software engineer who works at a company. I
\`\`\`
\`\`\`  a senior engineer.\`\`\`  how to search algorithm
 a
 **
 engineer
  not
 **
 2
 **
 a list.

\`\`\`
 **.\`
  __also, **, **, \`S, "1. a a a
    doc
 **
    a
.
. ,. a a.,,
[... continues with streams of punctuation and disconnected words ...]
```

## Root cause investigation

### Three bugs requiring workarounds to load the model at all

Two bugs in vLLM v0.15.1 prevent loading this AWQ model with default config detection:

1. **Mistral-native config path (`config-format: "auto"` / `"mistral"`):** The model's `params.json` contains a `vision_encoder` section (inherited from the Mistral Small 3.1 24B Instruct base model). vLLM v0.15.1's Mistral config adapter in [`vllm/transformers_utils/configs/mistral.py`](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/transformers_utils/configs/mistral.py) unconditionally routes any model with `vision_encoder` to `PixtralForConditionalGeneration` via the `_remap_mistral_vision_args` function, crashing with `KeyError: 'merging_layer.weight'` because Pixtral has no patch merger.

2. **HuggingFace config path (`config-format: "hf"`):** The model's `config.json` declares `text_config.model_type: "ministral3"`, which transformers v4.57.6 (bundled in the `vllm/vllm-openai:v0.15.1` Docker image) does not recognize -- `KeyError: 'ministral3'`. The `ministral3` model type was added in [transformers v5.0.0](https://github.com/huggingface/transformers/releases/tag/v5.0.0) (January 26, 2026), but vLLM v0.15.1 is pinned to `transformers >= 4.56.0, < 5`.

**Workaround applied:** `config_override.json` patches `text_config.model_type` from `"ministral3"` to `"mistral"`, and `config.yaml` uses `config-format: "hf"` with `hf-config-path` pointing at the patched config. This successfully loads the model, but introduces a third bug:

3. **Text backbone misrouted to `MistralForCausalLM`:** In [`vllm/model_executor/models/mistral3.py`](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/mistral3.py) (lines ~340-342), `Mistral3ForConditionalGeneration.__init__` has a special case written for the [HuggingFace-format Pixtral-12B model](https://huggingface.co/mistral-community/pixtral-12b):

   ```python
   # NOTE: These are special cases for Pixtral-12B in the HF-format
   # https://huggingface.co/mistral-community/pixtral-12b/blob/main/config.json
   if (
       config.text_config.architectures is None
       and config.text_config.model_type == "mistral"
   ):
       config.text_config.architectures = ["MistralForCausalLM"]
   ```

   Because our patched config has `model_type: "mistral"` AND `text_config.architectures` is null (the AWQ repo's `config.json` does not set this field), **both conditions are TRUE**. vLLM then calls `init_vllm_registered_model(vllm_config=vllm_config, hf_config=config.text_config)`, which resolves `"MistralForCausalLM"` from the model registry and instantiates the old Mistral 7B architecture (which inherits from `LlamaForCausalLM`) as the text backbone.

   The correct architecture for Mistral Devstral Small 2 24B should be `Ministral3ForCausalLM`, but **this class does not exist in vLLM v0.15.1's model registry at all**. It was added in [transformers v5.0.0](https://github.com/huggingface/transformers/releases/tag/v5.0.0) (January 26, 2026) and has not been backported into vLLM.

   Verified by inspecting the model registry inside the `vllm/vllm-openai:v0.15.1` container:

   ```python
   from vllm.model_executor.models.registry import _VLLM_MODELS
   # Only these Mistral-related entries exist:
   # MistralForCausalLM          -> ('mistral', 'MistralForCausalLM')
   # Mistral3ForConditionalGeneration -> ('mistral3', 'Mistral3ForConditionalGeneration')
   # MistralLarge3ForCausalLM    -> ('mistral_large_3', 'MistralLarge3ForCausalLM')
   # EagleMistralLarge3ForCausalLM -> ('mistral_large_3_eagle', 'EagleMistralLarge3ForCausalLM')
   # PixtralForConditionalGeneration -> ('pixtral', 'PixtralForConditionalGeneration')
   # NO Ministral3ForCausalLM entry
   ```

   `MistralLarge3ForCausalLM` is also not a viable substitute -- it inherits from `DeepseekV3ForCausalLM` (a Mixture-of-Experts architecture), which is architecturally incompatible with Mistral Devstral Small 2 24B (a dense transformer).

### Missing query scaling (`llama_4_scaling`)

Mistral Devstral Small 2 24B uses position-dependent query scaling (sometimes called "LLaMA 4 scaling"), defined by `llama_4_scaling_beta: 0.1` and `original_max_position_embeddings: 8192` in the model's `rope_parameters`.

vLLM v0.15.1's `MistralAttention` class (in [`vllm/model_executor/models/mistral.py`](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/mistral.py), which extends `LlamaAttention`) reads this scaling from `config.llama_4_scaling` (a top-level attribute on the config object). However:

- The **Mistral-native config path** (`params.json`) constructs `llama_4_scaling` as a top-level dict in the adapted config.
- The **HuggingFace-format config path** stores these values inside `rope_parameters.llama_4_scaling_beta`, and `transformers.MistralConfig` (v4.57.6) does **not** construct a separate `llama_4_scaling` attribute from `rope_parameters`.

**Result:** `MistralAttention.__init__` sets `self.do_llama_4_scaling = False` because `getattr(config, "llama_4_scaling", None)` returns `None`. The query scaling that the model was trained with is silently disabled.

**Partial fix applied:** Added `"llama_4_scaling": {"original_max_position_embeddings": 8192, "beta": 0.1}` to `text_config` in `config_override.json`. Verified it is now picked up:

```python
from transformers import MistralConfig
mc = MistralConfig(**text_config)
print(getattr(mc, 'llama_4_scaling'))
# Output: {'original_max_position_embeddings': 8192, 'beta': 0.1}
```

**This fix alone did not resolve the gibberish.** The repetition pattern changed slightly (the earlier test showed degeneration into random words and punctuation; after the fix, it loops on repeated import statements) but the model still cannot produce coherent long-form output.

### Ruled out: chat template and tokenizer mismatch

The `chat_template.jinja` files in the official FP8 model repository ([`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)) and the AWQ model repository ([`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit)) are **byte-for-byte identical**. The `tokenizer_config.json` files are also identical. Both use the Tekken tokenizer with `tokenizer_class: "TokenizersBackend"`. The chat template correctly handles `[SYSTEM_PROMPT]`, `[INST]`, `[TOOL_CALLS]`, `[ARGS]`, and `[TOOL_RESULTS]` special tokens.

The `params.json` files in both repositories are also identical -- same `llama_4_scaling`, same `yarn` RoPE configuration, same `vision_encoder` section.

The chat template and tokenizer are **not** contributing to the gibberish.

### Remaining cause: wrong text backbone architecture

The `MistralForCausalLM` text backbone class (inheriting from `LlamaForCausalLM`) was written for the original Mistral 7B / Mistral Small architecture family. While it does support `llama_4_scaling` query scaling (which the `config_override.json` patch enables), the `Ministral3ForCausalLM` class introduced in [transformers v5.0.0](https://github.com/huggingface/transformers/releases/tag/v5.0.0) likely has additional architectural differences in its attention forward pass beyond just the query scaling. Since `Ministral3ForCausalLM` does not exist anywhere in vLLM v0.15.1's codebase, these differences cannot be resolved by config patching alone.

## Comparison: official FP8 model vs. AWQ 4-bit model loading paths

### Weight file format

The official [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) repository ships **both** weight formats side by side:

| Format | Files | Total size |
|--------|-------|------------|
| Mistral-native consolidated | `consolidated-00001-of-00002.safetensors` (19.8 GiB) + `consolidated-00002-of-00002.safetensors` (5.9 GiB) | ~25.8 GiB |
| HuggingFace sharded | `model-00001-of-00006.safetensors` through `model-00006-of-00006.safetensors` | ~25.8 GiB |

The AWQ model ([`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit)) **only** ships HuggingFace sharded safetensors (`model-*.safetensors`, ~16 GiB total). It has no `consolidated-*.safetensors` files.

This is why the AWQ model **cannot** use `load-format: "mistral"` -- that flag requires Mistral-native consolidated safetensors, which the AWQ quantization tool (llm-compressor) does not produce.

### Config routing: the two loading paths

#### FP8 path (working): `config-format: "mistral"` + `load-format: "mistral"` + `tokenizer-mode: "mistral"`

1. vLLM reads `params.json` (Mistral-native format) via the Mistral config adapter in [`vllm/transformers_utils/configs/mistral.py`](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/transformers_utils/configs/mistral.py).
2. `params.json` has `llama_4_scaling` as a **top-level field** (directly consumed by `MistralAttention.__init__` via `getattr(config, "llama_4_scaling", None)`).
3. `params.json` has a `yarn` section with correct YaRN RoPE scaling parameters.
4. `tokenizer-mode: "mistral"` selects the [`mistral-common`](https://github.com/mistralai/mistral-common) Tekken tokenizer, which applies the correct Mistral chat template natively.
5. The Mistral config adapter routes the model through `Mistral3ForConditionalGeneration` (the correct multimodal wrapper, with vision disabled).
6. Weight files are loaded from `consolidated-*.safetensors` in Mistral's native format.
7. **Result:** Model loads correctly and produces coherent output.

#### AWQ path (broken): `config-format: "hf"` + `load-format: "safetensors"` + `hf-config-path` override

1. vLLM reads `config.json` (or the patched `config_override.json`) via HuggingFace `AutoConfig`.
2. `text_config.model_type` must be patched from `"ministral3"` to `"mistral"` because transformers v4.57.6 does not recognize `"ministral3"` -- raises `KeyError: 'ministral3'`.
3. `text_config.architectures` is **null** (the AWQ repo's `config.json` does not set this field).
4. Top-level `model_type: "mistral3"` and `architectures: ["Mistral3ForConditionalGeneration"]` resolve correctly from vLLM's model registry.
5. `Mistral3ForConditionalGeneration.__init__` runs and hits the [Pixtral-12B special case](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/mistral3.py):
   - `config.text_config.architectures is None` -- **TRUE** (AWQ repo does not set this field)
   - `config.text_config.model_type == "mistral"` -- **TRUE** (we patched it from `"ministral3"`)
   - vLLM forces `text_config.architectures = ["MistralForCausalLM"]`
6. `init_vllm_registered_model` resolves `"MistralForCausalLM"` from the registry, instantiating the old Mistral 7B architecture (inheriting from `LlamaForCausalLM`) as the text backbone.
7. Weight files are loaded from `model-*.safetensors` in HuggingFace sharded format.
8. **Result:** Model loads without errors but produces gibberish on outputs longer than ~50 tokens.

### `generation_config.json` comparison

| Field | Official FP8 ([`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)) | AWQ ([`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit)) |
|-------|------|------|
| `temperature` | **0.15** | **missing** |
| `do_sample` | **true** | **missing** |
| `max_length` | 262144 | 262144 |
| `bos_token_id` | 1 | 1 |
| `eos_token_id` | 2 | 2 |
| `pad_token_id` | 11 | 11 |

The AWQ repo's `generation_config.json` is incomplete -- it is missing both `temperature` and `do_sample`. This is why `config.yaml` uses `generation-config: "vllm"` with explicit `override-generation-config` instead of `generation-config: "auto"`.

### VRAM and context trade-off

| | FP8 Official | AWQ 4-bit |
|--|--|--|
| **Weights in VRAM** | ~25 GiB | ~14 GiB |
| **Remaining for BF16 KV cache** (at `gpu-memory-utilization: 0.98` on 32 GiB) | ~5-6 GiB | ~15 GiB |
| **BF16 KV cache capacity** (at 160 KiB/token) | ~32,000-38,000 tokens | ~98,768 tokens |
| **Output quality** | Correct (coherent) | **Broken** (gibberish on long outputs) |

## Survey of all available 4-bit quantized models (February 2026)

Every quantized variant of Mistral Devstral Small 2 24B on HuggingFace shares the same fundamental blocker: quantized models use HuggingFace-format sharded safetensors, which forces `config-format: "hf"`, which hits the unrecognized `text_config.model_type: "ministral3"` in transformers v4.57.6. The `config_override.json` workaround gets the model to load, but routes the text backbone through `MistralForCausalLM` (the wrong class), causing gibberish.

### AWQ quantizations

| Model | Group size | Disk size | Status on vLLM v0.15.1 |
|-------|-----------|-----------|------------------------|
| [`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit) | 32 | ~16 GiB | **Gibberish** (wrong text backbone) |
| [`androiddrew/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/androiddrew/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit) | 128 | ~16 GiB | **Same architecture routing issue expected** |

Both AWQ models use the compressed-tensors quantization format (INT4, symmetric) via [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor). The `androiddrew` variant uses `group_size: 128` (the standard Marlin-kernel-optimized group size) instead of `cyankiwi`'s `group_size: 32`, but both share the same `config.json` structure with `text_config.model_type: "ministral3"` and missing `text_config.architectures`. Switching to the `androiddrew` variant would not resolve the gibberish because the same Pixtral-12B special case would fire.

### GPTQ quantizations

| Model | Method | Disk size | Status on vLLM v0.15.1 |
|-------|--------|-----------|------------------------|
| [`btbtyler09/Devstral-Small-2-24B-Instruct-INT4-INT8-Mixed-GPTQ`](https://huggingface.co/btbtyler09/Devstral-Small-2-24B-Instruct-INT4-INT8-Mixed-GPTQ) | Mixed GPTQ (INT4 attention, INT8 MLP) | ~24 GiB | **Same architecture routing issue** + too large (24 GiB leaves almost no room for KV cache on 32 GiB VRAM) |

No pure GPTQ INT4 quantization (all layers at 4-bit) exists for Mistral Devstral Small 2 24B on HuggingFace. The only GPTQ variant uses mixed INT4/INT8, resulting in ~24 GiB on disk -- nearly as large as the FP8 original (~25 GiB). Both AWQ and GPTQ use the same compressed-tensors loading path in vLLM v0.15.1, so GPTQ offers no architectural routing advantage.

### Other quantization formats (not vLLM-compatible)

| Model | Format | Status |
|-------|--------|--------|
| [`Firworks/Devstral-Small-2-24B-Instruct-2512-nvfp4`](https://huggingface.co/Firworks/Devstral-Small-2-24B-Instruct-2512-nvfp4) | NVFP4 (FP4 with dual scaling) | **Does not load in vLLM v0.15.1** (model card confirms failure) |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit`](https://huggingface.co/mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit) | Apple MLX 4-bit | Apple Silicon only (not vLLM-compatible). Also reports [gibberish output](https://huggingface.co/mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit/discussions/1), suggesting cross-framework architecture handling issues. |
| [`DeathGodlike/Devstral-Small-2-24B-Instruct-2512_EXL3`](https://huggingface.co/DeathGodlike/Devstral-Small-2-24B-Instruct-2512_EXL3) | ExLlamaV3 (EXL3) | ExLlamaV3 inference engine only (not vLLM-compatible) |
| [`unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF`](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) | GGUF | llama.cpp / Ollama only. vLLM has experimental GGUF support (`--load-format gguf`) but it converts weights back to torch tensors (losing llama.cpp kernel optimizations) and is described as "highly experimental and under-optimized." |
| [`maxence-bouvier/Devstral-Small-2-24B-Instruct-SINQ-4bit`](https://huggingface.co/maxence-bouvier/Devstral-Small-2-24B-Instruct-SINQ-4bit) | SINQ (Sinkhorn-Normalized Quantization) | Requires custom SINQ fork + GemLite. Not vLLM-compatible. Reports ~1 tok/s on NVIDIA GeForce RTX 3090. |

### BitsAndBytes NF4 on-the-fly quantization

vLLM v0.15.1 supports BitsAndBytes for on-the-fly NF4 quantization (`--quantization bitsandbytes --load-format bitsandbytes`). However, `--load-format bitsandbytes` and `--load-format mistral` are mutually exclusive flags. BitsAndBytes requires the HuggingFace config path (`config-format: "hf"`), which hits the same `text_config.model_type: "ministral3"` / wrong text backbone problem. Additionally, there is no official BF16 checkpoint of Mistral Devstral Small 2 24B (BF16 would be ~48 GiB for 24 billion parameters) -- the official model ships in FP8 format only.

### Mistral AI official releases

Mistral AI has **not** released any quantized version of Mistral Devstral Small 2 24B. The only official model is [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) in FP8 format (~25 GiB on disk). All quantized versions are community-created.

## What works

- Short responses (< ~50 tokens) are coherent
- Tool calling (function name and JSON arguments) is correct
- Model loading succeeds without errors
- vLLM startup logs show no warnings about architecture mismatches

## What does not work

- Any response requiring more than ~50-100 tokens degenerates into repetitive gibberish
- The degeneration pattern: starts coherent -> repeats fragments -> echoes prompt words -> emits streams of punctuation and disconnected words

## Assessment of existing workarounds in this repository

Every patch in this repository addresses a real bug in a logically correct way. The chain of workarounds is well-reasoned -- each fix is a necessary response to a genuine vLLM v0.15.1 limitation. The problem is that the final link in the chain (the text backbone class selection) is broken at the vLLM level, not at the config level, and no amount of config patching can fix it.

### `config_override.json`: patching `text_config.model_type` from `"ministral3"` to `"mistral"`

**Correctly solves Bug 2** (transformers v4.57.6 `KeyError: 'ministral3'`). There is no other way to get past this error on the HuggingFace config path. This patch is not misguided -- it is the only option available. However, it is also what triggers Bug 3: because `text_config.model_type` is now `"mistral"`, the [Pixtral-12B special case](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/mistral3.py) in `Mistral3ForConditionalGeneration.__init__` fires and forces `MistralForCausalLM` as the text backbone.

### `config_override.json`: adding `text_config.llama_4_scaling`

**Correctly solves the missing query scaling bug.** `MistralAttention.__init__` in [`vllm/model_executor/models/mistral.py`](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/mistral.py) reads `getattr(config, "llama_4_scaling", None)`, and the HuggingFace config path does not construct this attribute from `rope_parameters`. The patch works -- it changed the gibberish pattern from total degeneration into punctuation soup to coherent starts followed by repetitive loops. This is a real fix for a real bug, but insufficient on its own because the wrong text backbone class (`MistralForCausalLM` instead of `Ministral3ForCausalLM`) has other architectural differences beyond just query scaling.

### `config.yaml`: `config-format: "hf"` + `load-format: "safetensors"`

**Not a choice -- it is a constraint.** AWQ models produced by [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) ship only HuggingFace-format sharded safetensors (`model-*.safetensors`), not Mistral-native consolidated safetensors (`consolidated-*.safetensors`). The `load-format: "mistral"` flag requires consolidated safetensors, which do not exist in the AWQ repository. This is the only loading path available.

### `config.yaml`: `hf-config-path: "/workspace/config_override"`

**The correct mechanism** to feed the patched `config_override.json` to vLLM instead of the AWQ model's original `config.json`.

### `config.yaml`: `limit-mm-per-prompt: '{"image": 0}'`

**Valid optimization regardless of quantization format.** Saves ~1 GiB of VRAM by skipping the dead Pixtral vision tower inherited from the Mistral Small 3.1 24B Instruct base model.

### `config.yaml`: `generation-config: "vllm"` with explicit `override-generation-config`

**Correctly works around the AWQ repo's incomplete `generation_config.json`** (missing `temperature` and `do_sample`). Not misguided.

### Why Bug 3 cannot be fixed from config

The Pixtral-12B special case in `Mistral3ForConditionalGeneration.__init__` is triggered by two conditions that are both consequences of correct earlier fixes:

- `config.text_config.architectures is None` -- true because the [`cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit`](https://huggingface.co/cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit) repository's `config.json` does not set this field (not something controllable from this repository)
- `config.text_config.model_type == "mistral"` -- true because we patched it (which we had to do to fix Bug 2)

Setting `text_config.architectures` explicitly in `config_override.json` would bypass the if-statement, but there is nowhere useful to point it. The correct value `Ministral3ForCausalLM` does not exist in vLLM v0.15.1's model registry. Setting it to `["MistralForCausalLM"]` just makes explicit what the special case already does. There is no correct text backbone class available in vLLM v0.15.1 for the `ministral3` architecture family.

**The repository's patches are a well-reasoned chain where each fix is correct, but the final link in the chain -- the text backbone class -- is broken at the vLLM level and cannot be resolved by config patching alone.**

## Resolution path

The fundamental problem is that **vLLM v0.15.1 does not have the `Ministral3ForCausalLM` text backbone class** in its model registry, and the workarounds needed to load any quantized model (patching `text_config.model_type`, forcing `config-format: "hf"`) cause the Pixtral-12B special case to fire and select the wrong text backbone class.

### Option 1: Use the official FP8 model with Mistral-native loading (confirmed working)

Switch from the AWQ 4-bit model to the official [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) FP8 model with `config-format: "mistral"`, `load-format: "mistral"`, `tokenizer-mode: "mistral"`. This bypasses all three bugs entirely because the Mistral-native loading path reads `params.json` (which has correct top-level `llama_4_scaling` and `yarn` fields), uses the `mistral-common` Tekken tokenizer, and routes through `Mistral3ForConditionalGeneration` correctly.

**Trade-off:** FP8 weights use ~25 GiB VRAM vs. ~14 GiB for AWQ 4-bit, leaving only ~5-6 GiB for BF16 KV cache (~32,000-38,000 tokens of context) instead of ~15 GiB (~98,768 tokens). However, FP8 weights are higher quality than INT4 AWQ -- the community-measured perplexity degradation for INT4 AWQ on this model is +10.5% over the original FP8 (perplexity 5.0161 vs. 4.5408, [measured by btbtyler09](https://huggingface.co/btbtyler09/Devstral-Small-2-24B-Instruct-INT4-INT8-Mixed-GPTQ)).

### Option 2: Wait for vLLM to merge the transformers v5 bump (timeline unknown)

[vllm-project/vllm#30566](https://github.com/vllm-project/vllm/pull/30566) ("Update to transformers v5") has been open since December 12, 2025. This PR bumps the transformers dependency from `>= 4.56.0, < 5` to `>= 5.0.0`, which would enable vLLM to natively recognize the `ministral3` model type. As of February 6, 2026, the PR is labeled "ready" but has not been approved by code owners and has ongoing merge conflicts. There is no published timeline for merging. vLLM v0.15.1 (February 4, 2026) is the latest release -- no v0.15.2 or v0.16.0 exists.

### Option 3: Fix the Mistral config adapter to avoid the Pixtral misroute (upstream issue)

[vllm-project/vllm#29904](https://github.com/vllm-project/vllm/issues/29904) tracks the bug where the Mistral config adapter in `vllm/transformers_utils/configs/mistral.py` unconditionally routes any model with `vision_encoder` in `params.json` to `PixtralForConditionalGeneration`. Fixing this to correctly route `Mistral3ForConditionalGeneration` models would allow AWQ models to use `config-format: "mistral"` (if they also had Mistral-native consolidated safetensors, which they currently do not). The issue has been open since December 2, 2025 with no fix or workaround posted.

### Option 4: Manually register `Ministral3ForCausalLM` in the container (fragile)

Monkey-patch or modify the `vllm/vllm-openai:v0.15.1` Docker image to add a `Ministral3ForCausalLM` entry to the model registry. This would require implementing or backporting the `Ministral3ForCausalLM` class from transformers v5.0.0. This approach is fragile, untested, and not recommended for production use.

## Upstream issues

- [vllm-project/vllm#29904](https://github.com/vllm-project/vllm/issues/29904) -- Mistral Large 3 675B being detected as `PixtralForConditionalGeneration` (same root cause as Bug 1 above; open since December 2, 2025, no resolution)
- [vllm-project/vllm#30566](https://github.com/vllm-project/vllm/pull/30566) -- PR to update vLLM to transformers v5.0.0 (open since December 12, 2025, not merged as of February 6, 2026)
- [vllm-project/vllm#29757](https://github.com/vllm-project/vllm/pull/29757) -- PR that added Mistral Large 3 and Ministral 3 support via the Mistral-native config path (merged December 2, 2025, included since vLLM v0.12.0; does **not** add `Ministral3ForCausalLM` to the model registry)
- [vllm-project/vllm#29968](https://github.com/vllm-project/vllm/issues/29968) -- Ministral 3 streaming tool calls fail with `JSONDecodeError` (open since December 2025)
- [vllm-project/vllm#33916](https://github.com/vllm-project/vllm/issues/33916) -- `IndexError` in streaming tool calls with the Mistral tool call parser (open since February 5, 2026)
- [vllm-project/llm-compressor#1652](https://github.com/vllm-project/llm-compressor/issues/1652) -- GPTQ-quantized Mistral Small models fail to load in vLLM due to vision tower layers being incorrectly quantized (fixed in [llm-compressor PR #1871](https://github.com/vllm-project/llm-compressor/pull/1871))
