#!/usr/bin/env bash
set -u

run_root=/run
monitor_log="$run_root/monitor.tsv"
interval_seconds="${1:-300}"

while true; do
  timestamp=$(date --iso-8601=seconds)
  complete=0
  for entry in dense:0 sparse:1 central:2 single:3; do
    IFS=: read -r arm gpu <<< "$entry"
    result_file=$(find "$run_root/$arm/out/history" -name '*.jsonl' -type f 2>/dev/null | head -1)
    if [[ -n "$result_file" ]]; then
      records=$(wc -l < "$result_file")
    else
      records=0
    fi
    (( records >= 300 )) && complete=$((complete + 1))
    gpu_fields=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader,nounits 2>/dev/null || printf 'NA, NA, NA, NA')
    IFS=, read -r memory util temp hw_thermal <<< "$gpu_fields"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$timestamp" "$arm" "$records" "$gpu" \
      "${memory// /}" "${util// /}" "${temp// /}" "${hw_thermal// /}" >> "$monitor_log"
  done
  (( complete == 4 )) && break
  sleep "$interval_seconds"
done
