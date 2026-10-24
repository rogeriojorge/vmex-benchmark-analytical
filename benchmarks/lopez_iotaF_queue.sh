#!/bin/bash
# usage: lopez_iotaF_queue.sh <queue-file> <lane-name>   (run on office, CPU only, from ~/vab-iotaF)
# queue lines: name|protocol|closure|ns|niter|ftol|threads|extra lopez_solve.py arguments
cd ~/vab-iotaF
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$HOME/vab-iotaF/pylib:$HOME/vab-iotaF/vmex-3b73d6f:$HOME/vab-iotaF/benchmarks
mkdir -p logs results/iotaF/solves
while IFS='|' read name proto clos ns niter ftol thr extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  export OMP_NUM_THREADS=$thr
  echo "START $name $(date +%T) free=$(free -g | awk '/Mem/{print $7}')G" >> logs/lane_$2.log
  ~/trace2venv/bin/python benchmarks/lopez_solve.py --protocol $proto --closure $clos --ns $ns --niter $niter --ftol $ftol $extra --tag "_${name##*_}" --out results/iotaF/solves/$name > logs/$name.log 2>&1
  echo "END $name rc=$? $(date +%T)" >> logs/lane_$2.log
done < "$1"
echo LANE_DONE >> logs/lane_$2.log
