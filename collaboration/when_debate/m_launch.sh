#!/bin/bash
# Study M runs (reports/mmlupro-20261121-a/PROTOCOL.md). GPU 3 starts when free; GPUs 0-2 after study T's launcher is done.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
M=$PWD/reports/mmlupro-20261121-a; TOUT=$PWD/reports/twins-hard-20261121-a/launch.out
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONPATH=/extra -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/envs/dmad-extra:/extra:ro -v $PWD/source/related-work/DMAD:/dmad:ro -v $PWD/collaboration/when_debate:/code:ro"
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
gpufree() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done; }
tdone() { until grep -q "ALL DONE" $TOUT 2>/dev/null; do sleep 30; done; gpufree $1; }
waitfor() { until grep -q "\"stage\"" "$1" 2>/dev/null; do sleep 20; done; }
dock() { # gpu name model env cmd
  docker run --rm --name mmlu-$2 --gpus device=$1 $COMMON $4 -v $3:/model:ro -v $M:/r -v $M/cache-g$1:/root/.cache --entrypoint bash $IMG -c "timeout 7200s $5"; }
mkdir -p $M/cache-g0 $M/cache-g1 $M/cache-g2 $M/cache-g3
for m in qwen gemma; do ln -sf ../out-neutral/${m}_solve.jsonl $M/out-indep/${m}_solve.jsonl; done
REV="/r/data/tasks.jsonl /r/out-neutral/qwen_solve.jsonl /r/out-neutral/gemma_solve.jsonl"
( gpufree 3
  dock 3 solve-gemma $GE "" "python3 /code/m_reverse.py solve gemma /r/data/tasks.jsonl /r/out-neutral/gemma_solve.jsonl > /r/out-neutral/gemma_solve.log 2>&1"
  waitfor $M/out-neutral/qwen_solve.log
  dock 3 rev-gemma-n $GE "" "python3 /code/m_reverse.py review gemma $REV /r/out-neutral/gemma_review.jsonl > /r/out-neutral/gemma_review.log 2>&1"
  dock 3 rev-gemma-i $GE "-e INDEP=1" "python3 /code/m_reverse.py review gemma $REV /r/out-indep/gemma_review.jsonl > /r/out-indep/gemma_review.log 2>&1" ) &
( tdone 0
  dock 0 solve-qwen $QW "" "python3 /code/m_reverse.py solve qwen /r/data/tasks.jsonl /r/out-neutral/qwen_solve.jsonl > /r/out-neutral/qwen_solve.log 2>&1"
  waitfor $M/out-neutral/gemma_solve.log
  dock 0 rev-qwen-n $QW "" "python3 /code/m_reverse.py review qwen $REV /r/out-neutral/qwen_review.jsonl > /r/out-neutral/qwen_review.log 2>&1"
  dock 0 rev-qwen-i $QW "-e INDEP=1" "python3 /code/m_reverse.py review qwen $REV /r/out-indep/qwen_review.jsonl > /r/out-indep/qwen_review.log 2>&1"
  dock 0 vote20 $QW "" "python3 /code/hard_gen.py H_vote20 /r/data/mmlupro300.csv /r/out-debate/H_vote20.json > /r/out-debate/H_vote20.log 2>&1" ) &
( tdone 1
  dock 1 full $QW "" "python3 /code/hard_gen.py H_full /r/data/mmlupro300.csv /r/out-debate/H_full.json > /r/out-debate/H_full.log 2>&1"
  dock 1 full-indep $QW "" "python3 /code/i_gen.py H_full_indep /r/data/mmlupro300.csv /r/out-debate/H_full_indep.json > /r/out-debate/H_full_indep.log 2>&1" ) &
( tdone 2
  dock 2 ans $QW "" "python3 /code/hard_gen.py H_ans /r/data/mmlupro300.csv /r/out-debate/H_ans.json > /r/out-debate/H_ans.log 2>&1"
  dock 2 ans-indep $QW "" "python3 /code/i_gen.py H_ans_indep /r/data/mmlupro300.csv /r/out-debate/H_ans_indep.json > /r/out-debate/H_ans_indep.log 2>&1" ) &
wait
echo ALL DONE
