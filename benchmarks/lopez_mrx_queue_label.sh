#!/bin/bash
# resolution ladder of the exact-chart (label) vacuum solve; one job at a time
MRXPATH=${MRXPATH:-$HOME/local/vmex-bench-work/mrx-lopez}
PY=${PY:-$HOME/local/vmex-bench-work/venv-mrxJ/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../results/lopez_mrx/ladder
cd /tmp
run() { # name ns... p
  name=$1; shift
  PYTHONPATH=$MRXPATH $PY $HERE/lopez_mrx_run.py --out $OUT/$name.json "$@" 2>&1 | tail -1
}
for tw in -1 0; do
  for ns in "6 8 12" "8 12 24" "10 16 32" "12 20 48" "16 24 64"; do
    n=$(echo $ns | tr ' ' '_')
    run label_tw${tw}_p3_$n --chart label --twist $tw --ns $ns --p 3 --turns 20
  done
done
for p in 2 4; do
  for ns in "8 12 24" "12 20 48"; do
    n=$(echo $ns | tr ' ' '_')
    run label_tw-1_p${p}_$n --chart label --twist -1 --ns $ns --p $p --turns 20
  done
done
