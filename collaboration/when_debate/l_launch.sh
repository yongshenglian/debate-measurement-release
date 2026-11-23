#!/bin/bash
# Study L runs (reports/framing-gen-20261122-a/PROTOCOL.md): study K with a different judge.
# Usage: l_launch.sh <judge: qwen38|inkling> <out subdir: out-<judge>|smoke-<judge>> [LIMIT]
# Phase 1: the judge's own solutions, samples 0/1/2 on GPUs 0-2. Phase 2: judge shards 0-3 on GPUs 0-3.
# The peer is Qwen2.5-7B sample 0 from study K (read-only). GPU-hour cap by construction: 3 x 4,800 s + 4 x 10,200 s = 15.3 GPU-h.
cd /home/lian/swarmworld/research
L=$PWD/reports/framing-gen-20261122-a; J=$1; O=$2; LIM=${3:+-e LIMIT=$3}
KD=$PWD/reports/framing-rep-20261122-a
case $J in
  qwen38)  IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
           REPO=/mnt/sdb/lian_models/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4; REV=63768c10df38c0395e12ef49edac1bd539eaeeea
           CTK='{"enable_thinking": false}'; LKW='{}'
           EXTRA="-e PYTHONPATH=/extra -v $PWD/envs/dmad-extra:/extra:ro" ;;   # as study K
  inkling) IMG=$(cat $L/inkling_image.digest)
           REPO=/mnt/sdb/lian_models/hub/models--thinkingmachines--Inkling-2-Mini; REV=b85c3cfa615fc413cc9af3379f491b963f3b586a
           CTK='{"reasoning_effort": 0.0}'; LKW='{}'
           EXTRA="" ;;   # Python 3.14 image; the harness does not need the study K extras (pandas for Python 3.12)
  *) echo "unknown judge $J"; exit 2 ;;
esac
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONDONTWRITEBYTECODE=1 $EXTRA
  --env-file $L/$O/env.list
  -v $PWD/collaboration/when_debate:/code:ro -v $KD/data:/data:ro -v $KD/out/qwen_solve.jsonl:/peer/qwen_solve.jsonl:ro
  -v $REPO:/mrepo:ro -v $L/$O:/o"
gpufree() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done; }
dock() { # gpu name timeout cmd
  gpufree $1
  docker run --rm --name l-$O-$2 --gpus device=$1 $COMMON $LIM -v $L/cache-$J-g$1:/root/.cache --entrypoint bash $IMG -c "timeout $3 $4"
  echo "$2 exit $? $(date -u +%FT%TZ)"; }
mkdir -p $L/cache-$J-g0 $L/cache-$J-g1 $L/cache-$J-g2 $L/cache-$J-g3 $L/$O
printf 'MODEL=/mrepo/snapshots/%s\nCTK=%s\nLLM_KW=%s\n' "$REV" "$CTK" "$LKW" > $L/$O/env.list
echo "start $J $(date -u +%FT%TZ) image $IMG model $REPO@$REV ctk $CTK"
for s in 0 1 2; do
  dock $s solve-s$s 4800s "python3 /code/l_gen.py solve $s /data/tasks.jsonl /o/judge_solve_s$s.jsonl > /o/judge_solve_s$s.log 2>&1" &
done
wait
echo "solves done $(date -u +%FT%TZ)"
for s in 0 1 2 3; do
  dock $s judge-$s 10200s "python3 /code/l_gen.py judge $s /data/tasks.jsonl /peer/qwen_solve.jsonl /o/judge_solve_s0.jsonl /o/judge_$s.jsonl > /o/judge_$s.log 2>&1" &
done
wait
echo "ALL DONE $(date -u +%FT%TZ)"
