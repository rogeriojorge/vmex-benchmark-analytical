#!/bin/bash
# usage: matrix_score_loop.sh <dest-dir> <vmex-path> [jobs]
# Repeatedly pulls finished runs from the office box and scores every run that has a WOUT and no
# volume score yet, until the office queue reports QUEUE_DONE for all phases listed in $STOP_MARKS.
DEST=$1; VMEXP=$2; JOBS=${3:-2}
HERE=$(cd $(dirname $0) && pwd)
export PYTHONPATH=$VMEXP OMP_NUM_THREADS=2
while true; do
  $HERE/matrix_collect.sh $DEST > /dev/null 2>&1
  pending=()
  for d in $DEST/runs/*/; do
    id=$(basename $d)
    [ -f $d/volume_score_v1.json ] && continue
    grep -q '"status": "running"' $d/receipt.json && continue
    grep -q '"wout": {' $d/receipt.json || continue
    pending+=("$id")
  done
  if [ ${#pending[@]} -gt 0 ]; then
    for case in $(printf '%s\n' "${pending[@]}" | sed -E 's/-(cold|ladder|seed|restart)[^-]*-ns[0-9]+-(vmex|vmec2000)$//' | sort -u); do
      dirs=()
      for id in "${pending[@]}"; do [[ $id == ${case}-* && ${id#${case}-} =~ ^(cold|ladder|seed|restart) ]] && dirs+=("$DEST/runs/$id"); done
      echo "$(date +%T) scoring ${#dirs[@]} runs of $case"
      python $HERE/score_solved.py "${dirs[@]}" --case $case >> $DEST/score_$case.log 2>&1 &
      while [ $(jobs -r | wc -l) -ge $JOBS ]; do sleep 5; done
    done
    wait
  fi
  grep -q "QUEUE_DONE.*$STOP_MARK" $DEST/logs/queue*.log 2>/dev/null && [ ${#pending[@]} -eq 0 ] && break
  sleep 300
done
