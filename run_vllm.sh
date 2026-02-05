#!/usr/bin/env bash
set -Eeuo pipefail

# Create a persistent vLLM v0.15.1 container for Mistral Devstral Small 2 24B (AWQ 4-bit).
# Not a system service; just a normal Docker container you can start/stop.
#
# Run once to create. After that: docker start devstral_vllm
# The container has --restart unless-stopped, so it survives reboots.

IMAGE="vllm/vllm-openai:v0.15.1"
NAME="devstral_vllm"
PORT="8000"             # container port (OpenAI-compatible API)
# Bind to the Docker bridge gateway (172.17.0.1) so the API is reachable by
# other containers (e.g., OpenWebUI) and from the host at 172.17.0.1:8000,
# but NOT exposed to the entire LAN. This matches the networking pattern from
# https://github.com/BigBIueWhale/personal_server.
BIND="172.17.0.1"
CONFIG="$(pwd)/config.yaml"
HF_CACHE="${HOME}/.cache/huggingface"

echo "Pulling image ${IMAGE} ..."
docker pull "${IMAGE}"

echo "Ensuring Hugging Face cache directory and config exist ..."
mkdir -p "${HF_CACHE}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config.yaml in $(pwd)"
  exit 1
fi

# Remove any stopped container with same name (idempotent install)
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Container ${NAME} already exists; removing to recreate ..."
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

echo "Creating container ${NAME} ..."
docker create \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  -p "${BIND}:${PORT}:8000" \
  -v "${CONFIG}:/workspace/config.yaml:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  "${IMAGE}" \
  vllm serve --config /workspace/config.yaml

echo "Starting ${NAME} ..."
docker start -a "${NAME}"
