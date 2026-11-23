#!/bin/bash
# Study T pipeline (reports/twins-hard-20261121-a/PROTOCOL.md): continuations (3 shards) -> build -> reviews.
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
T=$PWD/reports/twins-hard-20261121-a; H=$PWD/reports/debate-hard-20261120-a
QW=$PWD/models/debate-or-vote/qwen2.5-7b-instruct-a09a354; GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16
run() { # gpu name model args...
  local g=$1 n=$2 md=$3; shift 3; mkdir -p $T/cache-g$g
  docker run --rm --name twin-$n --gpus device=$g --network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
    -v $PWD/collaboration/twins_hard:/tcode:ro -v $md:/model:ro -v $T:/r -v $T/cache-g$g:/root/.cache --entrypoint bash $IMG \
    -c "timeout 5400s python3 /tcode/t_gen.py $* > /r/out/$n.log 2>&1"
}
mkdir -p $T/out $T/build
for i in 0 1 2; do run $i edit$i $QW edit /r/data/bases.shard$i.jsonl /r/out/edits.shard$i.jsonl & done; wait
cat $T/out/edits.shard{0,1,2}.jsonl > $T/out/edits.jsonl
python3 collaboration/twins_hard/t_build.py jobs $T/data/tasks.jsonl $T/data/labels.json $H/out/qwen_solve.jsonl $H/out/gemma_solve.jsonl $T/data/bases.jsonl $T/out/edits.jsonl $T/build || exit 1
python3 - <<PY
import json
for m, k in (("qwen", 1), ("gemma", 2)):
    J = [json.loads(l) for l in open("$T/build/" + m + "_jobs.jsonl")]
    for i in range(k):
        open(f"$T/build/{m}_jobs.part{i}.jsonl", "w").writelines(json.dumps(j) + "\n" for j in J if j["tid"] % k == i)
PY
run 0 review-qwen $QW review /r/build/qwen_jobs.part0.jsonl /r/out/qwen_review.jsonl &
run 1 review-gemma0 $GE review /r/build/gemma_jobs.part0.jsonl /r/out/gemma_review.part0.jsonl &
run 2 review-gemma1 $GE review /r/build/gemma_jobs.part1.jsonl /r/out/gemma_review.part1.jsonl &
wait
cat $T/out/gemma_review.part0.jsonl $T/out/gemma_review.part1.jsonl > $T/out/gemma_review.jsonl
echo ALL DONE
