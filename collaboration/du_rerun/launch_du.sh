#!/bin/bash
# Usage: launch_du.sh <gpu>   (runs Qwen then Llama, 300 questions, 3 rounds)
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
R=$PWD/reports/du-rerun-20261120-a; mkdir -p $R/out $R/cache
T=$PWD/collaboration/wynn_rerun/talk-aint-cheap/data/GSM8k/gsm8k_test.jsonl
docker run -d --name du-rerun --network none --ipc host --gpus device=$1 --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v $PWD/source/related-work/llm_multiagent_debate:/du:ro -v $PWD/collaboration/du_rerun:/code:ro -v $T:/data/test.jsonl:ro \
  -v $PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354:/m/qwen:ro -v $PWD/models/debate-or-vote/llama3.1-8b-instruct-nous-d10aef7:/m/llama:ro \
  -v $R/out:/out -v $R/cache:/root/.cache --entrypoint bash $IMG -c "
  for m in qwen llama; do timeout 3600s python3 /code/gen_du.py /m/\$m /data/test.jsonl 300 3 /out/du_\$m.json > /out/du_\$m.log 2>&1; echo \"\$m exit \$?\" >> /out/status.txt; done"
