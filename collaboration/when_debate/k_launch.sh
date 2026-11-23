#!/bin/bash
# Study K runs (reports/framing-rep-20261122-a/PROTOCOL.md). Usage: k_launch.sh <out subdir: out|smoke> [LIMIT]
# Phase 1 (solves, 4 GPUs in parallel): GPUs 0-2 Gemma samples 0/1/2, GPU 3 Qwen sample 0. Phase 2 (judge, Gemma): shards 0-3.
# GPU-hour cap by construction: 4 x (3,000 s + 10,200 s) = 14.7 GPU-h. Output is written in chunks, so a timeout keeps finished work.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
K=$PWD/reports/framing-rep-20261122-a; O=$1; LIM=${2:+-e LIMIT=$2}
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/collaboration/when_debate:/code:ro -v $K/data:/data:ro -v $K/$O:/o"
gpufree() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done; }
dock() { # gpu name model timeout cmd
  gpufree $1
  docker run --rm --name k-$O-$2 --gpus device=$1 $COMMON $LIM -v $3:/model:ro -v $K/cache-g$1:/root/.cache --entrypoint bash $IMG -c "timeout $4 $5"
  echo "$2 exit $? $(date -u +%FT%TZ)"; }
mkdir -p $K/cache-g0 $K/cache-g1 $K/cache-g2 $K/cache-g3 $K/$O
echo "start $(date -u +%FT%TZ)"
dock 0 solve-g0 $GE 3000s "python3 /code/k_gen.py solve gemma 0 /data/tasks.jsonl /o/gemma_solve_s0.jsonl > /o/gemma_solve_s0.log 2>&1" &
dock 1 solve-g1 $GE 3000s "python3 /code/k_gen.py solve gemma 1 /data/tasks.jsonl /o/gemma_solve_s1.jsonl > /o/gemma_solve_s1.log 2>&1" &
dock 2 solve-g2 $GE 3000s "python3 /code/k_gen.py solve gemma 2 /data/tasks.jsonl /o/gemma_solve_s2.jsonl > /o/gemma_solve_s2.log 2>&1" &
dock 3 solve-q0 $QW 3000s "python3 /code/k_gen.py solve qwen 0 /data/tasks.jsonl /o/qwen_solve.jsonl > /o/qwen_solve.log 2>&1" &
wait
echo "solves done $(date -u +%FT%TZ)"
for s in 0 1 2 3; do
  dock $s judge-$s $GE 10200s "python3 /code/k_gen.py judge $s /data/tasks.jsonl /o/qwen_solve.jsonl /o/gemma_solve_s0.jsonl /o/judge_$s.jsonl > /o/judge_$s.log 2>&1" &
done
wait
echo "ALL DONE $(date -u +%FT%TZ)"
