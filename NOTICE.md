# References and attribution

The benchmark code is MIT licensed (see `LICENSE`). The integer and sheared exact equilibria follow M. Landreman, arXiv:2609.26742, and the supplement at https://github.com/landreman/analytic_3d_equilibria. The Solov'ev cases, including the up-down asymmetric one, are derived in `benchmarks/analytic.py`.

No source of VMEX, VMEC2000, VMEC++ or DESC is included. Those codes, their assets and licenses remain with their repositories.

`benchmarks/upstream/landreman_sheared_desc.py` is vendored unchanged (git blob `e54499a810da26518cff4f0d3f1940b45e8d1789`) from M. Landreman's `analytic_3d_equilibria` supplement, MIT licensed, Copyright (c) 2026 Matt Landreman; its licence text is `benchmarks/upstream/LICENSE_landreman`. It is run as published and is not imported by any benchmark code.
