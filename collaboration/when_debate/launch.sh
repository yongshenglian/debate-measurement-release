#!/bin/bash
# Full runs for studies S (debate settings) and R (reverse peer). One sequential queue per GPU.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
S=$PWD/reports/debate-settings-20261120-a; V=$PWD/reports/reverse-peer-20261120-a
mkdir -p $S/out $V/out
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/source/related-work/DMAD:/dmad:ro -v $PWD/collaboration/when_debate:/code:ro"
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
arm() { # gpu arm
  mkdir -p $S/cache-g$1
  docker run --rm --name set-$2 --gpus device=$1 $COMMON -v $QW:/model:ro -v $S:/r -v $S/cache-g$1:/root/.cache --entrypoint bash $IMG \
    -c "timeout 5400s python3 /code/settings_gen.py $2 /r/gsm8k_test300.csv /r/out/$2.json > /r/out/$2.log 2>&1"
}
rev() { # gpu stage model model_dir args...
  local g=$1 st=$2 m=$3 md=$4; shift 4; mkdir -p $V/cache-g$g
  docker run --rm --name rev-$st-$m --gpus device=$g $COMMON -v $md:/model:ro -v $V:/r -v $V/cache-g$g:/root/.cache --entrypoint bash $IMG \
    -c "timeout 5400s python3 /code/reverse_gen.py $st $m $* > /r/out/${m}_$st.log 2>&1"
}
waitfor() { until grep -q "\"stage\"" "$1" 2>/dev/null; do sleep 20; done; }  # the log line is printed after the jsonl is fully written
( arm 0 S2_dmad; arm 0 S3_dmad_dovword; arm 0 S5_dmad_512 ) &
( arm 1 S1_dov; arm 1 S7_dov_chat; arm 1 S4_dmad_raw ) &
( rev 2 solve qwen $QW /r/tasks.jsonl /r/out/qwen_solve.jsonl; waitfor $V/out/gemma_solve.log
  rev 2 review qwen $QW /r/tasks.jsonl /r/out/qwen_solve.jsonl /r/out/gemma_solve.jsonl /r/out/qwen_review.jsonl; arm 2 S6_dmad_ansonly ) &
( rev 3 solve gemma $GE /r/tasks.jsonl /r/out/gemma_solve.jsonl; waitfor $V/out/qwen_solve.log
  rev 3 review gemma $GE /r/tasks.jsonl /r/out/qwen_solve.jsonl /r/out/gemma_solve.jsonl /r/out/gemma_review.jsonl ) &
wait
echo ALL DONE
