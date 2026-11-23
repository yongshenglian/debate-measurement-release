#!/bin/bash
# Study J runs (reports/neutral-judge-20261121-a/PROTOCOL.md). Usage: j_launch.sh <out subdir: out|smoke> [LIMIT]
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
R=$PWD/reports; J=$R/neutral-judge-20261121-a; O=$1; LIM=${2:+-e LIMIT=$2}
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/collaboration/when_debate:/code:ro -v $R:/rep:ro -v $J/$O:/o"
declare -A TS=( [gsm8k]="reverse-peer-20261120-a/tasks.jsonl reverse-peer-20261120-a/out" [math]="debate-hard-20261120-a/data/tasks.jsonl debate-hard-20261120-a/out" [mmlu]="mmlupro-20261121-a/data/tasks.jsonl mmlupro-20261121-a/out-neutral" )
lane() { # gpu judge model
  for ts in gsm8k math mmlu; do set -- $1 $2 $3 ${TS[$ts]}
    docker run --rm --name judge-$O-$2-$ts --gpus device=$1 $COMMON $LIM -v $3:/model:ro -v $J/cache-g$1:/root/.cache --entrypoint bash $IMG \
      -c "timeout 5400s python3 /code/j_gen.py $ts $2 /rep/$4 /rep/$5/qwen_solve.jsonl /rep/$5/gemma_solve.jsonl /o/${ts}_$2.jsonl > /o/${ts}_$2.log 2>&1"
    echo "$ts $2 exit $?"; done; }
mkdir -p $J/cache-g0 $J/cache-g1 $J/$O
lane 0 gemma $GE & lane 1 qwen $QW & wait
echo ALL DONE
