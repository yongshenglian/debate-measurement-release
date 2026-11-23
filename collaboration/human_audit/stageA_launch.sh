#!/bin/bash
# Human audit stage A (reports/human-audit-20261122/PROTOCOL.md): blind Gemma reading of all 39,000 part-A outputs,
# 4 shards on GPUs 0-3, started after study K's launcher prints ALL DONE. Per-container cap 3,600 s (4 GPU-h total at most).
cd /home/lian/swarmworld/research
IMG=vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
A=$PWD/reports/human-audit-20261122/stageA; KOUT=$PWD/reports/framing-rep-20261122-a/launch.out
GE=$PWD/models/heterogeneous/gemma-3-27b-it-w4a16; GEN=$PWD/reports/debate-vote-gsm8k-llama-20261119-a/blind-extraction
until grep -q "ALL DONE" $KOUT 2>/dev/null; do sleep 60; done
gpufree() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done; }
mkdir -p $A/cache-g0 $A/cache-g1 $A/cache-g2 $A/cache-g3
echo "start $(date -u +%FT%TZ)"
for s in 0 1 2 3; do
  ( gpufree $s
    docker run --rm --name audit-read-$s --gpus device=$s --network none --ipc host --cpus 6 --memory 96g -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
      -v $GE:/model:ro -v $GEN:/gen:ro -v $A:/a -v $A/cache-g$s:/root/.cache --entrypoint bash $IMG \
      -c "timeout 3600s python3 /gen/generate.py chat /a/jobs_$s.jsonl /a/out_$s.jsonl > /a/out_$s.log 2>&1"
    echo "shard $s exit $? $(date -u +%FT%TZ)" ) &
done
wait
cat $A/out_0.jsonl $A/out_1.jsonl $A/out_2.jsonl $A/out_3.jsonl > $A/out.jsonl
echo "ALL DONE $(date -u +%FT%TZ) $(wc -l < $A/out.jsonl) lines"
