#!/bin/bash
# one-turn-map structure of the true vacuum field in the VMEC-deck domains; one job at a time (RSS ~13 GB)
MRXPATH=${MRXPATH:-$HOME/local/vmex-bench-work/mrx-lopez}
PY=${PY:-$HOME/local/vmex-bench-work/venv-mrxJ/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../results/lopez_mrx/cylham
CACHE=${CACHE:-/tmp}
cd /tmp
for mn in "6 24" "8 40" "10 56" "10 72"; do
  set -- $mn
  PYTHONPATH=$MRXPATH $PY $HERE/lopez_mrx_run.py --chart cyl --p 3 --batch 20000 --turns 10 --steps 96 --ham \
    --mpol $1 --ntor $2 --cache $CACHE/cylF_$1_$2.npz --ns 6 8 96 --out $OUT/cyl_$1_$2_6_8_96.json 2>&1 | tail -1
done
