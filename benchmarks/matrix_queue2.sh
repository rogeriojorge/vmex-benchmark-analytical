#!/bin/bash
# usage: matrix_queue.sh <queue-file> [omp-threads]    runs the lines of a queue file serially on the office box
# CPU only; waits while less than MIN_GB of memory is available; one run at a time.
BASE=$HOME/vab-matrixG
cd $BASE/bench
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=${2:-4} OPENBLAS_NUM_THREADS=${2:-4}
export PYTHONPATH=$BASE/vmex-main:$BASE/bench/benchmarks
XV=/home/rjorge/local/refcodes/VMEC2000/build/build/bin/xvmec
MIN_GB=${MIN_GB:-6}
mkdir -p $BASE/logs $BASE/runs $BASE/cache
while IFS='|' read case code proto ns tag extra; do
  [ -z "$case" ] && continue
  id="${case}-${proto}${tag}-ns${ns}-${code}"
  extra=${extra//@RUNS@/$BASE/runs}
  [ -d $BASE/runs/$id ] && { echo "SKIP $id exists" >> $BASE/logs/queue_d.log; continue; }
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_GB" ]; do sleep 60; done
  echo "START $id $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print "avail="$7"G"}')" >> $BASE/logs/queue_d.log
  export VMEX_COMPILATION_CACHE_DIR=$BASE/cache/$id
  $BASE/venv/bin/python benchmarks/matrix_solve.py --case $case --code $code --protocol $proto --ns $ns \
      ${tag:+--tag "$tag"} $extra --xvmec $XV --out-parent $BASE/runs > $BASE/logs/$id.log 2>&1
  echo "END $id rc=$? $(date +%T)" >> $BASE/logs/queue_d.log
  rm -rf $BASE/cache/$id
done < "$1"
echo "QUEUE_DONE $1 $(date +%T)" >> $BASE/logs/queue_d.log
