#!/bin/bash
# usage: runner.sh <queue-file>   lines: name|protocol|closure|ns|niter|ftol|tcon0(or -)|threads
cd ~/vab-lopezE
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=$HOME/vab-lopezE/pylib:$HOME/vab-lopezE/vmex-3b73d6f:$HOME/vab-lopezE/benchmarks
while IFS='|' read name proto clos ns niter ftol tcon thr extra; do
  [ -z "$name" ] && continue
  export OMP_NUM_THREADS=$thr
  targ=""; [ "$tcon" != "-" ] && targ="--tcon0 $tcon"
  echo "START $name $(date +%T)" >> logs/queue.log
  ~/trace2venv/bin/python benchmarks/lopez_solve.py --protocol $proto --closure $clos --ns $ns --niter $niter --ftol $ftol $targ $extra --tag "_${name##*_}" --out results/lopez/solves/$name > logs/$name.log 2>&1
  echo "END $name rc=$? $(date +%T)" >> logs/queue.log
done < "$1"
echo QUEUE3_DONE >> logs/queue.log
