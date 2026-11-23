#!/usr/bin/env bash
set -u

run_root=/run
audit_output="$run_root/paired-audit-v3.json"
audit_log="$run_root/audit-watcher-v3.log"

printf '%s watcher_started\n' "$(date --iso-8601=seconds)" >> "$audit_log"
while true; do
  ready=0
  invalid=0
  for arm in dense sparse central single; do
    result_file=$(find "$run_root/$arm/out/history" -name '*.jsonl' -type f 2>/dev/null | head -1)
    if [[ -z "$result_file" ]]; then records=0; else records=$(wc -l < "$result_file"); fi
    (( records == 300 )) && ready=$((ready + 1))
    (( records > 300 )) && invalid=1
  done
  if (( invalid == 1 )); then
    printf '%s invalid_more_than_300_rows\n' "$(date --iso-8601=seconds)" >> "$audit_log"
    exit 2
  fi
  (( ready == 4 )) && break
  sleep 300
done
sleep 30
if [[ -e "$audit_output" ]]; then
  printf '%s audit_output_already_exists\n' "$(date --iso-8601=seconds)" >> "$audit_log"
  exit 3
fi
python /work/analyze_v3.py --run "$run_root" --expected-rows 300 --output "$audit_output" >> "$audit_log" 2>&1
status=$?
printf '%s audit_exit_%s\n' "$(date --iso-8601=seconds)" "$status" >> "$audit_log"
exit "$status"
