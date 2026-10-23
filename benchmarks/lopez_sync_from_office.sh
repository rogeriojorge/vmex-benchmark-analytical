#!/bin/bash
cd /Users/rogerio/local/vmex-bench-work/lopezE
ssh office '~/vab-lopezE/collect.sh' 
rsync -a --prune-empty-dirs --include='*/' --include='receipt.json' --include='lopez_score_v1.json' --include='stdout.log' --exclude='*' office:vab-lopezE/results/lopez/solves/ results/lopez/solves/
rsync -a --prune-empty-dirs --include='*/' --include='receipt.json' --include='lopez_score_v1.json' --include='stdout.log' --exclude='*' office:vab-lopezE/results/lopez/solves_vmec2000/ results/lopez/solves_vmec2000/
rsync -a --prune-empty-dirs --include='*/' --include='receipt.json' --include='lopez_score_v1.json' --include='stdout.log' --exclude='*' office:vab-lopezE/results/lopez/runs/ results/lopez/runs/
python3 benchmarks/lopez_tables.py
