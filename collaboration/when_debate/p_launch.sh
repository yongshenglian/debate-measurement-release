#!/bin/bash
# Study P runs (reports/private-revision-20261123-a/PROTOCOL.md): private revision on the K (Gemma) and L (Qwen3.8) disagreement sets.
# Usage: p_launch.sh <judge: gemma|qwen38> <out subdir> [LIMIT]. Shards 0-3 on GPUs 0-3.
# GPU-hour cap by construction: gemma 4 x 2,400 s = 2.7 GPU-h; qwen38 4 x 5,400 s = 6.0 GPU-h.
cd /home/lian/swarmworld/research
P=$PWD/reports/private-revision-20261123-a; J=$1; O=$2; LIM=${3:+-e LIMIT=$3}
KD=$PWD/reports/framing-rep-20261122-a; LD=$PWD/reports/framing-gen-20261122-a
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
EXTRA="-e PYTHONPATH=/extra -v $PWD/envs/dmad-extra:/extra:ro"
case $J in
  gemma)  REPO=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16; MPATH=/mrepo; CTK='{}'; JS=$KD/out/gemma_solve_s0.jsonl; TO=2400s ;;
  qwen38) REPO=/mnt/sdb/lian_models/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4; MPATH=/mrepo/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea
          CTK='{"enable_thinking": false}'; JS=$LD/out-qwen38/judge_solve_s0.jsonl; TO=5400s ;;
  *) echo "unknown judge $J"; exit 2 ;;
esac
mkdir -p $P/$O $P/cache-$J-g0 $P/cache-$J-g1 $P/cache-$J-g2 $P/cache-$J-g3
printf 'MODEL=%s\nCTK=%s\nLLM_KW={}\n' "$MPATH" "$CTK" > $P/$O/env.list
COMMON="--network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONDONTWRITEBYTECODE=1 $EXTRA
  --env-file $P/$O/env.list
  -v $PWD/collaboration/when_debate:/code:ro -v $KD/data:/data:ro -v $KD/out/qwen_solve.jsonl:/peer/qwen_solve.jsonl:ro
  -v $JS:/judge/judge_solve_s0.jsonl:ro -v $REPO:/mrepo:ro -v $P/$O:/o"
gpufree() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done; }
echo "start $J $(date -u +%FT%TZ) model $REPO ctk $CTK"
for s in 0 1 2 3; do
  ( gpufree $s
    docker run --rm --name p-$O-$s --gpus device=$s $COMMON $LIM -v $P/cache-$J-g$s:/root/.cache --entrypoint bash $IMG \
      -c "timeout $TO python3 /code/p_gen.py $s /data/tasks.jsonl /peer/qwen_solve.jsonl /judge/judge_solve_s0.jsonl /o/self_$s.jsonl > /o/self_$s.log 2>&1"
    echo "self-$s exit $? $(date -u +%FT%TZ)" ) &
done
wait
echo "ALL DONE $(date -u +%FT%TZ)"
