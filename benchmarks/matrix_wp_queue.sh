#!/bin/bash
# usage: matrix_wp_queue.sh <queue-file> [omp-threads]
# Work-precision timing runs. Lines: case|code|protocol|ns|tag|extra|cachegroup
# VMEX runs of one cachegroup share a persistent compilation cache: the first is cold (empty cache),
# later ones warm.  Load average before and after is in every receipt.
BASE=$HOME/vab-matrixG
cd $BASE/bench
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=${2:-4} OPENBLAS_NUM_THREADS=${2:-4}
export PYTHONPATH=$BASE/vmex-main:$BASE/bench/benchmarks
XV=/home/rjorge/local/refcodes/VMEC2000/build/build/bin/xvmec
MIN_GB=${MIN_GB:-6}
WPDIR=${WPDIR:-runs_wp}   # a second pass on a quieter box uses WPDIR=runs_wp2
mkdir -p $BASE/logs $BASE/$WPDIR $BASE/cache_$WPDIR
while IFS='|' read case code proto ns tag extra group; do
  [ -z "$case" ] && continue
  id="${case}-${proto}${tag}-ns${ns}-${code}"
  [ -d $BASE/$WPDIR/$id ] && { echo "SKIP $id exists" >> $BASE/logs/${WPDIR}_queue.log; continue; }
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_GB" ]; do sleep 60; done
  echo "START $id $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)" >> $BASE/logs/${WPDIR}_queue.log
  export VMEX_COMPILATION_CACHE_DIR=$BASE/cache_$WPDIR/$group
  $BASE/venv/bin/python benchmarks/matrix_solve.py --case $case --code $code --protocol $proto --ns $ns \
      ${tag:+--tag "$tag"} $extra --xvmec $XV --out-parent $BASE/$WPDIR > $BASE/logs/${WPDIR}_$id.log 2>&1
  echo "END $id rc=$? $(date +%T)" >> $BASE/logs/${WPDIR}_queue.log
done < "$1"
echo "QUEUE_DONE $1 $(date +%T)" >> $BASE/logs/${WPDIR}_queue.log
