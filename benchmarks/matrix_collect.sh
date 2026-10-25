#!/bin/bash
# usage: matrix_collect.sh <dest-dir>   pull runs and logs from the office box; VMEX stdout goes to runs/<id>/stdout.log
DEST=$1
mkdir -p $DEST/runs $DEST/logs
rsync -a --exclude cache office:vab-matrixG/runs/ $DEST/runs/
rsync -a office:vab-matrixG/logs/ $DEST/logs/
for d in $DEST/runs/*/; do
  id=$(basename $d)
  case $id in *-vmex) [ -f $DEST/logs/$id.log ] && cp $DEST/logs/$id.log $d/stdout.log;; esac
done
