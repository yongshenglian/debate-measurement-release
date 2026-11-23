#!/usr/bin/env bash
set -u

run_root="/home/lian/swarmworld/research/reports/debate-vote-gsm8k-20261117-a"
monitor_log="$run_root/monitor.tsv"
interval_seconds="${1:-300}"

if [[ ! -e "$monitor_log" ]]; then
  printf 'timestamp\tarm\tcontainer_state\trecords\tgpu\tmemory_mib\tutil_pct\ttemp_c\thw_thermal\n' > "$monitor_log"
fi

while true; do
  timestamp=$(date --iso-8601=seconds)
  all_done=1
  for entry in \
    'dense:debate-vote-gsm8k-20261117-a-dense-gpu0:0' \
    'sparse:debate-vote-gsm8k-20261117-a-sparse-gpu1:1' \
    'central:debate-vote-gsm8k-20261117-a-central-gpu2:2' \
    'single:debate-vote-gsm8k-20261117-a-single-gpu3:3'; do
    IFS=: read -r arm container gpu <<< "$entry"
    state=$(docker inspect --format '{{.State.Status}}:{{.State.ExitCode}}:{{.State.OOMKilled}}' "$container" 2>/dev/null || printf 'missing:-1:false')
    [[ "$state" == running:* ]] && all_done=0
    result_file=$(find "$run_root/$arm/out/history" -name '*.jsonl' -type f 2>/dev/null | head -1)
    if [[ -n "$result_file" ]]; then
      records=$(wc -l < "$result_file")
    else
      records=0
    fi
    gpu_fields=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader,nounits 2>/dev/null || printf 'NA, NA, NA, NA')
    IFS=, read -r memory util temp hw_thermal <<< "$gpu_fields"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$timestamp" "$arm" "$state" "$records" "$gpu" \
      "${memory// /}" "${util// /}" "${temp// /}" "${hw_thermal// /}" >> "$monitor_log"
  done
  (( all_done == 1 )) && break
  sleep "$interval_seconds"
done
