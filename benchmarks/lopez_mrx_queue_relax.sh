#!/bin/bash
# sheared-through-iota=1 start fields, relaxed by MRX's ideal Newton descent, with and without a resonant seed
source $HOME/mrxJ-office/mrxJ/benchmarks/office_env.sh
HERE=$HOME/mrxJ-office/mrxJ/benchmarks
OUT=$HOME/mrxJ-office/mrxJ/results/lopez_mrx/relax
cd /tmp
for seed in 0 1e-4 1e-3; do
  $PY $HERE/lopez_mrx_relax.py --eps 0.002 --seed $seed --newton --steps 20 --chunk 5 --out $OUT/eps2e-3_seed$seed 2>&1 | tail -1
done
