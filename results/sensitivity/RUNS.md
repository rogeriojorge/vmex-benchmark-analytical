# Stage 5 run log (office, CPU only, JAX_PLATFORMS=cpu, OMP_NUM_THREADS=4, nice 10, one job at a time)

VMEX `caf09ee7faca2e1c74d139ef6910f05e02654514` (0.12.0), clone `~/vab-sensI/vmex-caf09ee`, venv Python 3.12.13, JAX/jaxlib 0.11.2, solvax 0.31.0.
Every record carries its command, script and map SHA-256, versions and environment. Raw arrays are the `.npz` next to each `.json`.

| record | command (abridged) | outcome |
|---|---|---|
| `integer_axisymmetric_ns{33,65,129}.json` | `sensitivity_vmex.py --case integer_axisymmetric --ns N --steps 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 (129: five steps) --vol 24 32 2` | accepted roots |
| `integer_axisymmetric_ns257.json` | five steps, `--vol 24 32 2 --vol-fine 10 32 2` | accepted |
| `integer_axisymmetric_ns{33,65,129}_fine.json` | no FD, `--vol-fine 10 32 2` | accepted; JVP only |
| `integer_axisymmetric_ns129_quad.json` | no FD, `--vol-fine 16 48 2` | quadrature check: fine volume metric moves 2.4% (c), 2.8% (delta) |
| `integer_axisymmetric_ns{17,33}_dense.json` | `--dense --steps 1e-4 --vol 16 24 2` | dense control, accepted |
| `integer_axisymmetric_ns65_tcon0_0.json` | `--tcon0 0` | accepted (diagnostic) |
| `integer_axisymmetric_ns129_tcon0_0.json` | `--tcon0 0 --ftol 1e-12` | root |F| 5e-7: REJECTED, kept as a record |
| `integer_3d_ns{17,33}.json` | `--case integer_3d --params c delta a --seed exact --ftol 1e-10` | accepted roots |
| `integer_3d_ns17_aug.json` | `--augment --steps 1e-4` | diagnostic: tangent + frozen-entry drift, not supported |
| `integer_3d_ns25.json` | as above with `--augment` | root |F| 2e-6: REJECTED |
| `integer_3d_ns33_tcon0_0.json` | `--tcon0 0` | root |F| 2e-6: REJECTED |

Runs that produced no record (reasons are part of the evidence):

* integer 3-D NS17, `--ftol 1e-14`, exact seed, default TCON0, local: `VmecConvergenceError: MORE ITERATIONS REQUIRED` after the iteration budget; the host solver does not reach a 1e-14 force-square sum on the 3-D deck. Rerun at ftol 1e-10 (302 iterations), where the library anchor then Newton-refines to |F| 3e-14.
* integer axisymmetric NS129, `--tcon0 0`, `--ftol 1e-14`: same error; rerun at `--ftol 1e-12` gives the rejected record above.
* integer 3-D NS65 (twice, default TCON0 and `--tcon0 0`): process killed by signal 9, maximum resident set size 52,545,828 kB (the host has 62 GB), before any output. NS17 peaks at 11.1 GB and NS25 at 15.8 GB. Not retried.
* The first axisymmetric NS129 launch was stopped by hand after 15 minutes to rerun with fewer FD steps; no record.
