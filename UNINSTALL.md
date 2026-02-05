# Uninstall / Cleanup

## Stop and remove the persistent Docker container

```bash
docker stop devstral_vllm || true
docker rm devstral_vllm || true
```

## Optionally delete the Docker image (re-downloads on next install)

```bash
docker rmi vllm/vllm-openai:v0.15.1
```

## Optional Hugging Face cache cleanup (removes downloaded model weights)

This removes the cached AWQ 4-bit weights for Mistral Devstral Small 2 24B Instruct 2512:

```bash
rm -rf ~/.cache/huggingface/hub/models--cyankiwi--Devstral-Small-2-24B-Instruct-2512-AWQ-4bit
```

To remove **all** cached Hugging Face models (not just Devstral):

```bash
rm -rf ~/.cache/huggingface/hub/*
```

## Optional: remove Mistral Vibe CLI v2.0.2 global config

```bash
rm -rf ~/.vibe
```

## Optional: uninstall Mistral Vibe CLI v2.0.2

```bash
uv tool uninstall mistral-vibe
```
