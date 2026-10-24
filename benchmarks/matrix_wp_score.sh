#!/bin/bash
# usage: matrix_wp_score.sh <dest-dir> <vmex-path>
# Pull the work-precision runs and score repeat 0 and every tolerance-sweep run (repeats of one configuration
# write bitwise-identical WOUTs, which matrix_wp_tables.py checks by hash). Surface scores are skipped.
DEST=$1; export PYTHONPATH=$2 OMP_NUM_THREADS=2
HERE=$(cd $(dirname $0) && pwd)
mkdir -p $DEST/wp_runs
rsync -a --exclude cache office:vab-matrixG/runs_wp/ $DEST/wp_runs/
for d in $DEST/wp_runs/*/; do id=$(basename $d); [ -f $DEST/../logs/wp_$id.log ] && true; done
rsync -a office:vab-matrixG/logs/ $DEST/wp_logs/
for d in $DEST/wp_runs/*/; do
  id=$(basename $d)
  case $id in *-vmex) [ -f $DEST/wp_logs/wp_$id.log ] && cp $DEST/wp_logs/wp_$id.log $d/stdout.log;; esac
done
for case in $(ls $DEST/wp_runs | sed -E 's/-(cold|ladder)_.*$//' | sort -u); do
  dirs=()
  for d in $DEST/wp_runs/${case}-*/; do
    id=$(basename $d)
    [ -f $d/volume_score_v1.json ] && continue
    grep -q '"wout": {' $d/receipt.json || continue
    [[ $id == *_rep0-* || $id == *_ftol* ]] && dirs+=("$d")
  done
  [ ${#dirs[@]} -gt 0 ] && python $HERE/score_solved.py "${dirs[@]}" --case $case --no-surface
done
