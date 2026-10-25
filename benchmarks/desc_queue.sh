#!/bin/bash
# Serial, resumable job queue for desc_matched.py on a shared machine (CPU only, one job at a time).
# usage: desc_queue.sh jobs.txt     (lines: "<tag> <desc_matched.py arguments>"; '#' comments)
# Gates (checked before every job): no other desc_matched job of ours, MemAvailable >= MIN_GB, load1 < LOAD_MAX.
# A job is skipped when results/<tag>/record.json exists.  Before each job wait until MemAvailable >= MIN_GB.
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=benchmarks
PY=${PY:-$HOME/desc-env/bin/python}
MIN_GB=${MIN_GB:-10}      # expected RSS of a job plus a 4 GB margin
LOAD_MAX=${LOAD_MAX:-28}  # keep total machine load below the core count
mkdir -p logs results
while read -r tag args; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  d=results/$tag
  [ -f "$d/record.json" ] && continue
  # also never overlap with a job of ours still running
  while pgrep -f benchmarks/desc_matched.py > /dev/null \
     || [ "$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)" -lt "$MIN_GB" ] \
     || [ "$(awk '{printf "%d", $1}' /proc/loadavg)" -ge "$LOAD_MAX" ]; do sleep 60; done
  rm -rf "$d"
  start=$(date +%s)
  echo "START $tag $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)" >> logs/queue.log
  nice -n 10 "$PY" benchmarks/desc_matched.py $args --output-dir "$d" > "logs/$tag.log" 2>&1
  echo "DONE $tag rc=$? wall=$(( $(date +%s)-start ))s $(date -Is)" >> logs/queue.log
done < "$1"
echo "QUEUE_FINISHED $1 $(date -Is)" >> logs/queue.log
