#!/bin/bash
# Study I full runs (reports/debate-indep-20261121-a/PROTOCOL.md). One job per GPU.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
I=$PWD/reports/debate-indep-20261121-a; H=$PWD/reports/debate-hard-20261120-a
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/source/related-work/DMAD:/dmad:ro -v $PWD/collaboration/when_debate:/code:ro -v $H/out:$H/out:ro"
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
arm() { docker run --rm --name ind-$2 --gpus device=$1 $COMMON -v $QW:/model:ro -v $I:/r -v $I/cache-g$1:/root/.cache --entrypoint bash $IMG \
  -c "timeout 7200s python3 /code/i_gen.py $2 /r/math_hard300.csv /r/out/$2.json > /r/out/$2.log 2>&1"; }
rev() { docker run --rm --name ind-review-$2 --gpus device=$1 $COMMON -v $3:/model:ro -v $I:/r -v $I/cache-g$1:/root/.cache --entrypoint bash $IMG \
  -c "timeout 7200s python3 /code/i_reverse.py review $2 /r/tasks.jsonl /r/out/qwen_solve.jsonl /r/out/gemma_solve.jsonl /r/out/$2_review.jsonl > /r/out/$2_review.log 2>&1"; }
arm 0 H_full_indep & arm 1 H_ans_indep & rev 2 qwen $QW & rev 3 gemma $GE &
wait
echo ALL DONE
