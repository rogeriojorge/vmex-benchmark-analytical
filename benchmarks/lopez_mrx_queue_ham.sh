#!/bin/bash
# deformed-domain scan with the one-turn-map effective Hamiltonian at two MRX resolutions (twisted chart), one job at a time
MRXPATH=${MRXPATH:-$HOME/local/vmex-bench-work/mrx-lopez}
PY=${PY:-$HOME/local/vmex-bench-work/venv-mrxJ/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../results/lopez_mrx/ham
cd /tmp
run() { # tag ns eps m n [extra]
  tag=$1; ns=$2; e=$3; m=$4; n=$5; shift 5
  PYTHONPATH=$MRXPATH $PY $HERE/lopez_mrx_perturb.py --eps $e --m $m --n $n --ns $ns --turns 300 --out $OUT/$tag "$@" 2>&1 | tail -1
}
for ns in "8 12 24" "12 20 48"; do
  t=ns${ns// /_}
  run base_$t "$ns" 0 1 1
  for c in "0p01 0.01 3 1" "0p05 0.05 3 1" "0p1 0.1 3 1" "0p01 0.01 2 1" "0p05 0.05 2 1" "0p1 0.1 2 1" "0p2 0.2 2 1" "0p05 0.05 1 1" "0p2 0.2 1 1" "0p05 0.05 2 2" "0p05 0.05 3 2" "0p1 0.1 2 0"; do
    set -- $c
    run e$1_m$3n$4_$t "$ns" $2 $3 $4
  done
done
run isl_e0p1_m3n1_ns12_20_48 "12 20 48" 0.1 3 1 --islands
