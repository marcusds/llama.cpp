#!/usr/bin/env bash
# llama-swap cmd for deepseek-v4.1-flash: makes the vLLM container's life == this process's life.
# Starts the (already created) container if it is stopped, then blocks following its logs.
# Exits when the container stops, so llama-swap sees the model as stopped; cmdStop does `docker stop`.
set -euo pipefail
NAME=dsv41-vllm-a
docker start "$NAME" >/dev/null
exec docker logs -f --tail 0 "$NAME"
