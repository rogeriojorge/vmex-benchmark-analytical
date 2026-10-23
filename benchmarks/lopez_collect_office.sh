#!/bin/bash
# on office: copy logs next to runs and score any VMEC2000 WOUT not yet scored
cd ~/vab-lopezE
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 PYTHONPATH=$HOME/vab-lopezE/benchmarks
for d in results/lopez/solves/*/; do n=$(basename $d); [ -f logs/$n.log ] && cp logs/$n.log $d/stdout.log; done
for d in results/lopez/solves_vmec2000/*/ results/lopez/runs/*/; do
  if [ -f $d/receipt.json ] && ls $d/wout_*.nc >/dev/null 2>&1 && [ ! -f $d/lopez_score_v1.json ]; then
    ~/trace2venv/bin/python benchmarks/score_lopez.py --run-dir $d 2>&1 | tail -1
  fi
done
