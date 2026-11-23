#!/bin/bash
# Study H full runs (reports/debate-hard-20261120-a/PROTOCOL.md). One sequential queue per GPU.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
H=$PWD/reports/debate-hard-20261120-a; mkdir -p $H/out
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/source/related-work/DMAD:/dmad:ro -v $PWD/collaboration/when_debate:/code:ro"
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
arm() { # gpu arm
  docker run --rm --name hard-$2 --gpus device=$1 $COMMON -v $QW:/model:ro -v $H:/r -v $H/cache-g$1:/root/.cache --entrypoint bash $IMG \
    -c "timeout 7200s python3 /code/hard_gen.py $2 /r/data/math_hard300.csv /r/out/$2.json > /r/out/$2.log 2>&1"
}
rev() { # gpu stage model model_dir args...
  local g=$1 st=$2 m=$3 md=$4; shift 4
  docker run --rm --name hard-$st-$m --gpus device=$g $COMMON -v $md:/model:ro -v $H:/r -v $H/cache-g$g:/root/.cache --entrypoint bash $IMG \
    -c "timeout 7200s python3 /code/hard_reverse.py $st $m $* > /r/out/${m}_$st.log 2>&1"
}
waitfor() { until grep -q "\"stage\"" "$1" 2>/dev/null; do sleep 20; done; }
( arm 0 H_full; arm 0 H_vote20 ) &
( arm 1 H_ans ) &
( rev 2 solve qwen $QW /r/data/tasks.jsonl /r/out/qwen_solve.jsonl; waitfor $H/out/gemma_solve.log
  rev 2 review qwen $QW /r/data/tasks.jsonl /r/out/qwen_solve.jsonl /r/out/gemma_solve.jsonl /r/out/qwen_review.jsonl; arm 2 H_dovword ) &
( rev 3 solve gemma $GE /r/data/tasks.jsonl /r/out/gemma_solve.jsonl; waitfor $H/out/qwen_solve.log
  rev 3 review gemma $GE /r/data/tasks.jsonl /r/out/qwen_solve.jsonl /r/out/gemma_solve.jsonl /r/out/gemma_review.jsonl ) &
wait
echo ALL DONE
