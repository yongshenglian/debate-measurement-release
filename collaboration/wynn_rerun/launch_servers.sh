#!/bin/bash
# Local OpenAI-compatible vLLM servers for the Wynn et al. re-run, on an internal (no-internet) docker network.
# Usage: launch_servers.sh <gpu_llama> <gpu_mistral>
set -e
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
docker network inspect wynn-net >/dev/null 2>&1 || docker network create --internal wynn-net >/dev/null
R=$PWD/reports/wynn-rerun-20261120-a; mkdir -p $R/cache-llama $R/cache-mistral
docker run -d --name wynn-llama --network wynn-net --ipc host --gpus device=$1 --cpus 8 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -v $PWD/models/debate-or-vote/llama3.1-8b-instruct-nous-d10aef7:/model:ro -v $R/cache-llama:/root/.cache \
  $IMG --model /model --served-model-name llama --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.88 --seed 0
docker run -d --name wynn-mistral --network wynn-net --ipc host --gpus device=$2 --cpus 8 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -v $PWD/models/wynn/mistral-7b-instruct-v0.2-63a8b08:/model:ro -v $R/cache-mistral:/root/.cache \
  $IMG --model /model --served-model-name mistral --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.88 --seed 0
