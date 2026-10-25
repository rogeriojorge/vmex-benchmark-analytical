#!/bin/bash
# cylindrical-chart (VMEC-like) runs on office: MRX resolution ladder and deck-truncation scan; one job at a time
source $HOME/mrxJ-office/mrxJ/benchmarks/office_env.sh
HERE=$HOME/mrxJ-office/mrxJ/benchmarks
OUT=$HOME/mrxJ-office/mrxJ/results/lopez_mrx/cyl
cd /tmp
run() { name=$1; shift; $PY $HERE/lopez_mrx_run.py --chart cyl --p 3 --batch 20000 --turns 10 --steps 96 --out $OUT/$name.json "$@" 2>&1 | tail -1; }
C=$HOME/mrxJ-office
run cyl_10_72_6_8_48  --mpol 10 --ntor 72 --cache $C/cylF_10_72.npz --ns 6 8 48
run cyl_10_72_6_8_96  --mpol 10 --ntor 72 --cache $C/cylF_10_72.npz --ns 6 8 96
run cyl_10_72_6_10_144 --mpol 10 --ntor 72 --cache $C/cylF_10_72.npz --ns 6 10 144
for mn in "6 24" "8 40" "10 56"; do
  set -- $mn
  run cyl_$1_$2_6_8_96 --mpol $1 --ntor $2 --cache $C/cylF_$1_$2.npz --ns 6 8 96
done
