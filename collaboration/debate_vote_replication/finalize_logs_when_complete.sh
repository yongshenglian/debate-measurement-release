#!/usr/bin/env bash
set -u

project_root=/home/lian/swarmworld/research
run_root="$project_root/reports/debate-vote-gsm8k-20261117-a"
work_root="$project_root/collaboration/debate_vote_replication"
data_root="$project_root/datasets/debate-or-vote/gsm8k-740312a"
env_root="$project_root/envs/debate-or-vote-py310"
image='pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2'
finalizer_log="$run_root/finalizer.log"
audit_output="$run_root/final-log-source-audit.json"

printf '%s finalizer_started\n' "$(date --iso-8601=seconds)" >> "$finalizer_log"
while true; do
  ready=0
  invalid=0
  for arm in dense sparse central single; do
    history=$(find "$run_root/$arm/out/history" -name '*.jsonl' -type f 2>/dev/null | head -1)
    if [[ -n "$history" ]]; then rows=$(wc -l < "$history"); else rows=0; fi
    (( rows == 300 )) && ready=$((ready + 1))
    (( rows > 300 )) && invalid=1
  done
  if (( invalid )); then
    printf '%s invalid_more_than_300_rows\n' "$(date --iso-8601=seconds)" >> "$finalizer_log"
    exit 2
  fi
  (( ready == 4 )) && break
  sleep 300
done

mkdir -p "$run_root/logs" "$run_root/container-inspect"
for entry in dense:0 sparse:1 central:2 single:3; do
  IFS=: read -r arm gpu <<< "$entry"
  container="debate-vote-gsm8k-20261117-a-${arm}-gpu${gpu}"
  docker wait "$container" >> "$finalizer_log"
  status=$(docker inspect "$container" --format '{{.State.Status}}|{{.State.ExitCode}}|{{.State.OOMKilled}}')
  if [[ "$status" != 'exited|0|false' ]]; then
    printf '%s bad_container_state %s %s\n' "$(date --iso-8601=seconds)" "$container" "$status" >> "$finalizer_log"
    exit 3
  fi
  docker inspect "$container" > "$run_root/container-inspect/$arm.json"
  docker logs "$container" > "$run_root/logs/$arm.log" 2>&1
done

if [[ -e "$audit_output" ]]; then
  printf '%s audit_output_already_exists\n' "$(date --iso-8601=seconds)" >> "$finalizer_log"
  exit 4
fi
docker run --rm --network none \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -v "$env_root:/env:ro" -v "$data_root:/data" \
  -v "$run_root:/run" -v "$work_root:/work:ro" \
  "$image" /env/bin/python /work/verify_gsm8k_source_v2.py \
  --run /run --data-dir /data --logs /run/logs \
  --expected-size 300 --require-complete > "$audit_output" 2>> "$finalizer_log"
status=$?
printf '%s final_log_audit_exit_%s\n' "$(date --iso-8601=seconds)" "$status" >> "$finalizer_log"
exit "$status"
