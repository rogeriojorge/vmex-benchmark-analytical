# DESC matched runs: generated tables

Scores: exact-chart whole-volume relative L2 (level q2); `B native` is DESC's own-grid score.

## Factor isolation

### sheared_A, M = 8

| run | seed | closure | optimizer | grid | N | L | tol | status | its | nfev | cost0 | cost | solve s | B | J | force | B native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | zero | iota | lsq-exact | default | 8 | 10 | ours | cap | 150 | 179 | 1.73e-01 | 4.68e-09 | 393 | 2.48e-04 | 1.04e-03 | 1.60e-03 | 2.48e-04 |
| base_plus_L | zero | iota | lsq-exact | default | 8 | 8 | ours | converged | 26 | 29 | 1.69e-01 | 6.44e-08 | 37 | 2.75e-04 | 2.26e-03 | 2.33e-03 | 2.75e-04 |
| base_plus_N | zero | iota | lsq-exact | default | 16 | 10 | ours | cap | 150 | 199 | 1.73e-01 | 2.02e-09 | 1452 | 8.59e-05 | 3.18e-04 | 3.21e-04 | 8.59e-05 |
| base_plus_closure | zero | current | lsq-exact | default | 8 | 10 | ours | converged | 146 | 185 | 1.68e-01 | 5.38e-09 | 433 | 2.10e-04 | 8.89e-04 | 1.70e-03 | 2.10e-04 |
| base_plus_grid | zero | iota | lsq-exact | quadrature | 8 | 10 | ours | cap | 150 | 187 | 2.19e-01 | 6.81e-09 | 956 | 2.94e-04 | 1.18e-03 | 1.83e-03 | 2.94e-04 |
| base_plus_optimizer | zero | iota | matrix-free | default | 8 | 10 | ours | cap | 149 | 150 | 1.73e-01 | 5.38e-08 | 222 | 1.34e-03 | 6.39e-03 | 8.95e-03 | 1.34e-03 |
| base_plus_seed | exact | iota | lsq-exact | default | 8 | 10 | ours | cap | 150 | 172 | 1.04e-06 | 4.69e-09 | 502 | 2.49e-04 | 9.54e-04 | 1.55e-03 | 2.49e-04 |
| base_plus_tol | zero | iota | lsq-exact | default | 8 | 10 | published | converged | 8 | 13 | 1.73e-01 | 5.38e-09 | 42 | 2.44e-04 | 8.93e-04 | 1.48e-03 | 2.44e-04 |
| pub | exact | current | matrix-free | quadrature | 16 | 8 | published | converged | 3 | 4 | 2.10e-06 | 3.50e-07 | 69 | 1.12e-04 | 1.07e-03 | 1.66e-03 | 1.12e-04 |
| pub_minus_L | exact | current | matrix-free | quadrature | 16 | 10 | published | converged | 3 | 4 | 1.57e-06 | 3.39e-08 | 25 | 8.75e-05 | 3.41e-04 | 5.27e-04 | 8.75e-05 |
| pub_minus_N | exact | current | matrix-free | quadrature | 8 | 8 | published | converged | 3 | 4 | 2.11e-06 | 3.47e-07 | 22 | 1.17e-04 | 1.11e-03 | 1.65e-03 | 1.17e-04 |
| pub_minus_closure | exact | iota | matrix-free | quadrature | 16 | 8 | published | converged | 4 | 5 | 2.95e-06 | 3.46e-07 | 41 | 1.21e-04 | 1.32e-03 | 1.65e-03 | 1.21e-04 |
| pub_minus_grid | exact | current | matrix-free | default | 16 | 8 | published | converged | 3 | 4 | 1.93e-06 | 7.01e-07 | 34 | 1.21e-04 | 1.15e-03 | 1.62e-03 | 1.21e-04 |
| pub_minus_optimizer | exact | current | lsq-exact | quadrature | 16 | 8 | published | converged | 2 | 6 | 1.99e-06 | 1.16e-07 | 123 | 1.76e-04 | 2.24e-03 | 1.72e-03 | 1.76e-04 |
| pub_minus_seed | zero | current | matrix-free | quadrature | 16 | 8 | published | converged | 20 | 21 | 2.09e-01 | 1.26e-06 | 240 | 2.49e-03 | 1.32e-02 | 1.27e-02 | 2.49e-03 |
| pub_minus_tol | exact | current | matrix-free | quadrature | 16 | 8 | ours | cap | 149 | 150 | 2.10e-06 | 1.40e-07 | 633 | 1.46e-04 | 1.97e-03 | 1.04e-03 | 1.46e-04 |

### sheared_A, M = 10

| run | seed | closure | optimizer | grid | N | L | tol | status | its | nfev | cost0 | cost | solve s | B | J | force | B native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | zero | iota | lsq-exact | default | 10 | 12 | ours | cap | 150 | 184 | 1.75e-01 | 1.42e-10 | 1045 | 1.93e-05 | 1.09e-04 | 1.59e-04 | 1.93e-05 |
| base_plus_L | zero | iota | lsq-exact | default | 10 | 10 | ours | converged | 137 | 178 | 1.72e-01 | 1.14e-09 | 607 | 2.26e-05 | 3.56e-04 | 2.95e-04 | 2.26e-05 |
| base_plus_closure | zero | current | lsq-exact | default | 10 | 12 | ours | cap | 150 | 159 | 1.70e-01 | 2.92e-10 | 1056 | 1.78e-05 | 1.49e-04 | 2.05e-04 | 1.77e-05 |
| base_plus_grid | zero | iota | lsq-exact | quadrature | 10 | 12 | ours | cap | 150 | 214 | 2.22e-01 | 2.16e-10 | 5643 | 1.97e-05 | 1.16e-04 | 1.84e-04 | 1.97e-05 |
| base_plus_optimizer | zero | iota | matrix-free | default | 10 | 12 | ours | cap | 149 | 150 | 1.75e-01 | 7.59e-08 | 346 | 1.59e-03 | 7.85e-03 | 1.03e-02 | 1.59e-03 |
| base_plus_seed | exact | iota | lsq-exact | default | 10 | 12 | ours | cap | 150 | 160 | 2.62e-08 | 1.38e-10 | 828 | 1.43e-05 | 8.33e-05 | 1.53e-04 | 1.43e-05 |
| base_plus_tol | zero | iota | lsq-exact | default | 10 | 12 | published | converged | 8 | 13 | 1.75e-01 | 2.11e-10 | 66 | 1.97e-05 | 1.22e-04 | 1.70e-04 | 1.97e-05 |
| pub | exact | current | matrix-free | quadrature | 20 | 10 | published | converged | 1 | 2 | 7.01e-08 | 1.50e-08 | 21 | 1.74e-05 | 1.89e-04 | 3.32e-04 | 1.74e-05 |
| pub_minus_L | exact | current | matrix-free | quadrature | 20 | 12 | published | converged | 1 | 2 | 4.08e-08 | 3.60e-09 | 22 | 1.37e-05 | 9.62e-05 | 1.72e-04 | 1.37e-05 |
| pub_minus_N | exact | current | matrix-free | quadrature | 10 | 10 | published | converged | 1 | 2 | 7.07e-08 | 1.51e-08 | 17 | 1.80e-05 | 1.89e-04 | 3.32e-04 | 1.80e-05 |
| pub_minus_closure | exact | iota | matrix-free | quadrature | 20 | 10 | published | converged | 1 | 2 | 8.25e-08 | 1.63e-08 | 18 | 1.76e-05 | 1.90e-04 | 3.47e-04 | 1.76e-05 |
| pub_minus_grid | exact | current | matrix-free | default | 20 | 10 | published | converged | 1 | 2 | 1.83e-07 | 1.45e-07 | 18 | 1.95e-05 | 1.91e-04 | 3.40e-04 | 1.95e-05 |
| pub_minus_optimizer | exact | current | lsq-exact | quadrature | 20 | 10 | published | converged | 2 | 5 | 6.69e-08 | 4.28e-09 | 395 | 2.55e-05 | 3.69e-04 | 5.20e-04 | 2.55e-05 |
| pub_minus_seed | zero | current | matrix-free | quadrature | 20 | 10 | published | converged | 24 | 25 | 2.12e-01 | 1.06e-06 | 204 | 2.46e-03 | 1.43e-02 | 1.25e-02 | 2.46e-03 |
| pub_minus_tol | exact | current | matrix-free | quadrature | 20 | 10 | ours | cap | 149 | 150 | 7.01e-08 | 3.87e-09 | 1206 | 1.63e-05 | 2.52e-04 | 1.72e-04 | 1.63e-05 |

### sheared_issan, M = 8

| run | seed | closure | optimizer | grid | N | L | tol | status | its | nfev | cost0 | cost | solve s | B | J | force | B native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pub | exact | current | matrix-free | quadrature | 16 | 8 | published | converged | 2 | 3 | 6.45e-08 | 4.97e-09 | 18 | 8.37e-06 | 9.09e-05 | 1.25e-04 | 8.40e-06 |
| pub_minus_L | exact | current | matrix-free | quadrature | 16 | 10 | published | converged | 1 | 2 | 8.23e-09 | 1.27e-09 | 14 | 5.47e-06 | 3.05e-05 | 6.27e-05 | 5.47e-06 |
| pub_minus_N | exact | current | matrix-free | quadrature | 8 | 8 | published | converged | 3 | 4 | 3.03e-06 | 9.55e-07 | 17 | 2.12e-04 | 5.45e-04 | 1.59e-03 | 2.13e-04 |
| pub_minus_closure | exact | iota | matrix-free | quadrature | 16 | 8 | published | converged | 3 | 4 | 9.55e-08 | 3.96e-09 | 19 | 8.37e-06 | 1.11e-04 | 1.11e-04 | 8.41e-06 |
| pub_minus_grid | exact | current | matrix-free | default | 16 | 8 | published | converged | 39 | 40 | 5.54e-05 | 3.98e-05 | 72 | 1.95e-03 | 1.20e-02 | 8.22e-03 | 1.95e-03 |
| pub_minus_seed | zero | current | matrix-free | quadrature | 16 | 8 | published | converged | 28 | 29 | 6.32e-01 | 1.39e-06 | 120 | 4.62e-03 | 2.16e-02 | 2.33e-02 | 4.62e-03 |
| pub_minus_tol | exact | current | matrix-free | quadrature | 16 | 8 | ours | cap | 149 | 150 | 6.45e-08 | 1.63e-09 | 599 | 9.97e-06 | 1.17e-04 | 7.96e-05 | 9.92e-06 |

## Matched M ladders (work-precision rows)

| case | protocol | M | N | L | status | its | nfev | solve s | solve cpu s | total s | B | J | force | B native | J native | outside pts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| integer_3d | strict | 4 | 4 | 6 | converged | 54 | 62 | 18 | 64 | 200 | 1.35e-03 | 1.60e-02 | 5.06e-02 | 1.35e-03 | 1.60e-02 | 46 |
| integer_3d | strict | 6 | 6 | 8 | converged | 47 | 60 | 38 | 173 | 274 | 1.31e-04 | 1.46e-03 | 1.14e-02 | 1.31e-04 | 1.46e-03 | 0 |
| integer_3d | strict | 8 | 8 | 10 | converged | 23 | 37 | 69 | 347 | 371 | 1.02e-05 | 1.49e-04 | 1.27e-03 | 1.02e-05 | 1.49e-04 | 0 |
| integer_3d | strict | 10 | 10 | 12 | converged | 6 | 15 | 85 | 451 | 467 | 7.33e-07 | 1.64e-05 | 1.45e-04 | 7.33e-07 | 1.64e-05 | 0 |
| integer_3d | strict | 12 | 12 | 14 | converged | 5 | 13 | 271 | 1353 | 796 | 7.01e-08 | 1.88e-06 | 1.69e-05 | 7.01e-08 | 1.88e-06 | 0 |
| integer_3d | loose | 4 | 4 | 6 | converged | 14 | 20 | 17 | 41 | 192 | 1.22e-03 | 1.70e-02 | 2.84e-02 | 1.22e-03 | 1.71e-02 | 46 |
| integer_3d | loose | 6 | 6 | 8 | converged | 6 | 9 | 20 | 51 | 243 | 1.16e-04 | 1.33e-03 | 9.96e-03 | 1.16e-04 | 1.33e-03 | 0 |
| integer_3d | loose | 8 | 8 | 10 | converged | 3 | 6 | 25 | 90 | 303 | 9.44e-06 | 1.44e-04 | 1.15e-03 | 9.44e-06 | 1.44e-04 | 0 |
| integer_3d | loose | 10 | 10 | 12 | converged | 1 | 5 | 38 | 166 | 343 | 1.36e-06 | 4.67e-05 | 4.32e-04 | 1.36e-06 | 4.67e-05 | 0 |
| integer_3d | exact | 4 | 4 | 6 | converged | 8 | 9 | 18 | 36 | 194 | 1.42e-03 | 1.56e-02 | 6.05e-02 | 1.43e-03 | 1.56e-02 | 46 |
| integer_3d | exact | 6 | 6 | 8 | converged | 4 | 8 | 19 | 48 | 253 | 1.10e-04 | 1.24e-03 | 9.07e-03 | 1.10e-04 | 1.24e-03 | 0 |
| integer_3d | exact | 8 | 8 | 10 | converged | 4 | 8 | 27 | 108 | 306 | 5.94e-06 | 9.85e-05 | 7.70e-04 | 5.94e-06 | 9.85e-05 | 0 |
| integer_3d | exact | 10 | 10 | 12 | converged | 1 | 5 | 29 | 144 | 323 | 8.22e-07 | 3.16e-05 | 2.93e-04 | 8.22e-07 | 3.16e-05 | 0 |
| integer_3d | pub | 4 | 8 | 4 | cap/stopped | 149 | 150 | 134 | 385 | 308 | 2.33e-03 | 3.21e-02 | 5.74e-02 | 2.35e-03 | 3.22e-02 | 42 |
| integer_3d | pub | 6 | 12 | 6 | converged | 11 | 12 | 30 | 168 | 256 | 1.91e-04 | 4.19e-03 | 4.82e-03 | 1.91e-04 | 4.19e-03 | 0 |
| integer_3d | pub | 8 | 16 | 8 | converged | 9 | 10 | 45 | 366 | 345 | 1.54e-05 | 4.96e-04 | 8.29e-04 | 1.54e-05 | 4.96e-04 | 0 |
| integer_3d | pub | 10 | 20 | 10 | converged | 3 | 4 | 33 | 262 | 433 | 1.31e-06 | 6.94e-05 | 2.95e-04 | 1.31e-06 | 6.94e-05 | 0 |
| integer_3d_tau | strict | 6 | 6 | 8 | converged | 68 | 77 | 145 | 827 | 264 | 4.40e-05 | 6.06e-04 | 4.44e-03 | 4.40e-05 | 6.06e-04 | 0 |
| integer_3d_tau | strict | 8 | 8 | 10 | converged | 6 | 14 | 111 | 585 | 285 | 1.83e-06 | 3.69e-05 | 2.56e-04 | 1.83e-06 | 3.69e-05 | 0 |
| integer_3d_tau | strict | 10 | 10 | 12 | converged | 4 | 12 | 700 | 2403 | 1021 | 1.24e-07 | 3.60e-06 | 2.66e-05 | 1.24e-07 | 3.60e-06 | 0 |
| integer_3d_tau | loose | 6 | 6 | 8 | converged | 4 | 7 | 20 | 78 | 115 | 3.17e-05 | 4.10e-04 | 2.81e-03 | 3.17e-05 | 4.10e-04 | 0 |
| integer_3d_tau | loose | 8 | 8 | 10 | converged | 2 | 6 | 42 | 227 | 167 | 1.62e-06 | 3.61e-05 | 2.61e-04 | 1.62e-06 | 3.61e-05 | 0 |
| integer_3d_tau | loose | 10 | 10 | 12 | converged | 0 | 1 | 60 | 241 | 213 | 2.01e-07 | 6.13e-06 | 5.99e-05 | 2.01e-07 | 6.13e-06 | 0 |
| integer_3d_tau | exact | 6 | 6 | 8 | converged | 5 | 7 | 21 | 83 | 120 | 4.10e-05 | 5.59e-04 | 4.07e-03 | 4.10e-05 | 5.59e-04 | 0 |
| integer_3d_tau | exact | 8 | 8 | 10 | converged | 1 | 5 | 41 | 189 | 192 | 2.02e-06 | 4.60e-05 | 3.50e-04 | 2.02e-06 | 4.60e-05 | 0 |
| integer_3d_tau | exact | 10 | 10 | 12 | converged | 0 | 1 | 67 | 264 | 249 | 7.23e-08 | 2.26e-06 | 1.89e-05 | 7.23e-08 | 2.26e-06 | 0 |
| integer_3d_tau_mirror | strict | 6 | 6 | 8 | converged | 62 | 68 | 121 | 703 | 246 | 4.40e-05 | 6.06e-04 | 4.44e-03 | 4.40e-05 | 6.06e-04 | 0 |
| integer_3d_tau_mirror | strict | 8 | 8 | 10 | converged | 6 | 14 | 213 | 855 | 581 | 1.82e-06 | 3.67e-05 | 2.54e-04 | 1.82e-06 | 3.67e-05 | 0 |
| integer_3d_tau_mirror | strict | 10 | 10 | 12 | converged | 4 | 12 | 543 | 2062 | 764 | 1.28e-07 | 3.68e-06 | 2.72e-05 | 1.28e-07 | 3.68e-06 | 0 |
| integer_3d_tau_mirror | loose | 6 | 6 | 8 | converged | 4 | 7 | 20 | 78 | 116 | 3.18e-05 | 4.11e-04 | 2.81e-03 | 3.18e-05 | 4.11e-04 | 0 |
| integer_3d_tau_mirror | loose | 8 | 8 | 10 | converged | 2 | 6 | 43 | 230 | 168 | 1.64e-06 | 3.64e-05 | 2.63e-04 | 1.64e-06 | 3.64e-05 | 0 |
| integer_3d_tau_mirror | loose | 10 | 10 | 12 | converged | 0 | 1 | 59 | 241 | 227 | 2.01e-07 | 6.13e-06 | 5.99e-05 | 2.01e-07 | 6.13e-06 | 0 |
| integer_3d_tau_mirror | exact | 6 | 6 | 8 | converged | 5 | 7 | 26 | 96 | 145 | 4.11e-05 | 5.61e-04 | 4.08e-03 | 4.11e-05 | 5.61e-04 | 0 |
| integer_3d_tau_mirror | exact | 8 | 8 | 10 | converged | 1 | 5 | 41 | 190 | 191 | 2.01e-06 | 4.59e-05 | 3.48e-04 | 2.01e-06 | 4.59e-05 | 0 |
| integer_3d_tau_mirror | exact | 10 | 10 | 12 | converged | 0 | 1 | 68 | 264 | 249 | 7.23e-08 | 2.26e-06 | 1.89e-05 | 7.23e-08 | 2.26e-06 | 0 |
| sheared_A | strict | 6 | 6 | 8 | converged | 26 | 32 | 34 | 120 | 284 | 7.26e-04 | 2.60e-03 | 3.46e-03 | 7.26e-04 | 2.60e-03 | 0 |
| sheared_A | strict | 8 | 8 | 10 | cap/stopped | 150 | 179 | 393 | 2321 | 594 | 2.48e-04 | 1.04e-03 | 1.60e-03 | 2.48e-04 | 1.04e-03 | 0 |
| sheared_A | strict | 10 | 10 | 12 | cap/stopped | 150 | 184 | 1045 | 7697 | 1258 | 1.93e-05 | 1.09e-04 | 1.59e-04 | 1.93e-05 | 1.09e-04 | 0 |
| sheared_A | loose | 6 | 6 | 8 | converged | 7 | 11 | 21 | 56 | 257 | 7.15e-04 | 2.51e-03 | 3.51e-03 | 7.15e-04 | 2.51e-03 | 0 |
| sheared_A | loose | 8 | 8 | 10 | converged | 8 | 13 | 34 | 155 | 316 | 2.44e-04 | 8.93e-04 | 1.48e-03 | 2.44e-04 | 8.93e-04 | 0 |
| sheared_A | loose | 10 | 10 | 12 | converged | 8 | 13 | 79 | 483 | 437 | 1.97e-05 | 1.22e-04 | 1.70e-04 | 1.97e-05 | 1.22e-04 | 0 |
| sheared_A | exact | 6 | 6 | 8 | converged | 5 | 10 | 17 | 46 | 220 | 7.10e-04 | 3.20e-03 | 3.21e-03 | 7.10e-04 | 3.20e-03 | 0 |
| sheared_A | exact | 8 | 8 | 10 | converged | 4 | 10 | 24 | 106 | 276 | 1.14e-04 | 5.67e-04 | 8.36e-04 | 1.14e-04 | 5.67e-04 | 0 |
| sheared_A | exact | 10 | 10 | 12 | converged | 2 | 7 | 46 | 233 | 407 | 2.04e-05 | 9.52e-05 | 1.55e-04 | 2.04e-05 | 9.52e-05 | 0 |
| sheared_A | pub | 6 | 12 | 6 | converged | 11 | 12 | 30 | 168 | 268 | 1.16e-03 | 9.42e-03 | 7.86e-03 | 1.16e-03 | 9.42e-03 | 0 |
| sheared_A | pub | 8 | 16 | 8 | converged | 3 | 4 | 21 | 140 | 333 | 1.12e-04 | 1.07e-03 | 1.66e-03 | 1.12e-04 | 1.07e-03 | 0 |
| sheared_A | pub | 10 | 20 | 10 | converged | 1 | 2 | 17 | 105 | 429 | 1.74e-05 | 1.89e-04 | 3.31e-04 | 1.74e-05 | 1.89e-04 | 0 |
| sheared_issan | strict | 6 | 6 | 8 | converged | 110 | 138 | 61 | 335 | 294 | 5.94e-03 | 2.08e-02 | 3.56e-02 | 5.94e-03 | 2.08e-02 | 0 |
| sheared_issan | strict | 8 | 8 | 10 | cap/stopped | 150 | 174 | 580 | 2439 | 1015 | 3.55e-03 | 1.28e-02 | 2.26e-02 | 3.55e-03 | 1.28e-02 | 0 |
| sheared_issan | loose | 6 | 6 | 8 | converged | 15 | 20 | 25 | 83 | 267 | 5.28e-03 | 1.89e-02 | 3.10e-02 | 5.28e-03 | 1.89e-02 | 0 |
| sheared_issan | loose | 8 | 8 | 10 | converged | 9 | 15 | 37 | 174 | 273 | 7.40e-04 | 3.41e-03 | 6.00e-03 | 7.40e-04 | 3.41e-03 | 0 |
| sheared_issan | loose | 10 | 10 | 12 | converged | 10 | 16 | 74 | 504 | 370 | 4.24e-03 | 1.54e-02 | 2.69e-02 | 4.24e-03 | 1.54e-02 | 0 |
| sheared_issan | exact | 6 | 6 | 8 | converged | 11 | 19 | 22 | 72 | 247 | 3.11e-03 | 1.21e-02 | 1.77e-02 | 3.11e-03 | 1.21e-02 | 0 |
| sheared_issan | exact | 8 | 8 | 10 | converged | 4 | 11 | 29 | 121 | 308 | 3.55e-04 | 2.59e-03 | 4.69e-03 | 3.54e-04 | 2.59e-03 | 0 |
| sheared_issan | exact | 10 | 10 | 12 | converged | 1 | 7 | 32 | 159 | 338 | 7.19e-05 | 2.80e-04 | 5.08e-04 | 7.21e-05 | 2.80e-04 | 0 |
| sheared_issan | pub | 6 | 12 | 6 | converged | 8 | 9 | 24 | 129 | 260 | 1.34e-04 | 1.10e-03 | 1.00e-03 | 1.34e-04 | 1.11e-03 | 0 |
| sheared_issan | pub | 8 | 16 | 8 | converged | 2 | 3 | 18 | 102 | 327 | 8.39e-06 | 9.09e-05 | 1.25e-04 | 8.40e-06 | 9.11e-05 | 0 |
| sheared_issan | pub | 10 | 20 | 10 | converged | 1 | 2 | 17 | 105 | 418 | 7.76e-07 | 8.61e-06 | 1.65e-05 | 7.76e-07 | 8.65e-06 | 0 |
| sheared_issan_tau | strict | 6 | 6 | 8 | converged | 90 | 111 | 304 | 1189 | 527 | 1.48e-02 | 5.04e-02 | 1.12e-01 | 1.48e-02 | 5.04e-02 | 0 |
| sheared_issan_tau | loose | 6 | 6 | 8 | converged | 11 | 16 | 30 | 144 | 145 | 1.45e-02 | 4.96e-02 | 1.13e-01 | 1.45e-02 | 4.96e-02 | 0 |
| sheared_issan_tau | loose | 8 | 8 | 10 | converged | 9 | 17 | 110 | 658 | 266 | 5.49e-03 | 1.94e-02 | 4.57e-02 | 5.49e-03 | 1.94e-02 | 0 |
| sheared_issan_tau | exact | 6 | 6 | 8 | converged | 26 | 36 | 51 | 297 | 174 | 1.34e-02 | 4.74e-02 | 1.02e-01 | 1.34e-02 | 4.74e-02 | 0 |
| sheared_issan_tau | exact | 8 | 8 | 10 | converged | 21 | 29 | 206 | 1341 | 347 | 2.29e-03 | 1.20e-02 | 2.20e-02 | 2.29e-03 | 1.20e-02 | 0 |
| sheared_issan_tau_mirror | strict | 6 | 6 | 8 | converged | 77 | 91 | 198 | 1075 | 324 | 1.48e-02 | 5.04e-02 | 1.12e-01 | 1.48e-02 | 5.04e-02 | 0 |
| sheared_issan_tau_mirror | loose | 6 | 6 | 8 | converged | 12 | 15 | 36 | 168 | 160 | 1.40e-02 | 4.82e-02 | 1.08e-01 | 1.40e-02 | 4.82e-02 | 0 |
| sheared_issan_tau_mirror | loose | 8 | 8 | 10 | converged | 9 | 17 | 108 | 653 | 246 | 5.45e-03 | 1.92e-02 | 4.55e-02 | 5.45e-03 | 1.92e-02 | 0 |
| sheared_issan_tau_mirror | exact | 6 | 6 | 8 | converged | 30 | 43 | 54 | 335 | 161 | 1.37e-02 | 4.80e-02 | 1.05e-01 | 1.37e-02 | 4.79e-02 | 0 |
| sheared_issan_tau_mirror | exact | 8 | 8 | 10 | converged | 23 | 31 | 176 | 1257 | 309 | 2.47e-03 | 1.26e-02 | 2.36e-02 | 2.47e-03 | 1.26e-02 | 0 |
| solovev_asymmetric | strict | 4 | 0 | 6 | converged | 10 | 15 | 11 | 21 | 47 | 8.21e-06 | 1.13e-04 | 1.31e-04 | 8.21e-06 | 1.13e-04 | 0 |
| solovev_asymmetric | strict | 6 | 0 | 8 | converged | 15 | 21 | 11 | 24 | 47 | 1.10e-07 | 2.42e-06 | 2.80e-06 | 1.10e-07 | 2.42e-06 | 0 |
| solovev_asymmetric | strict | 8 | 0 | 10 | cap/stopped | 150 | 159 | 15 | 48 | 51 | 4.47e-07 | 3.81e-05 | 1.52e-05 | 4.47e-07 | 3.81e-05 | 0 |
| solovev_asymmetric | strict | 10 | 0 | 12 | converged | 12 | 17 | 12 | 28 | 50 | 1.37e-09 | 5.05e-08 | 3.85e-07 | 1.26e-09 | 5.05e-08 | 0 |
| solovev_asymmetric | strict | 12 | 0 | 14 | converged | 14 | 23 | 13 | 33 | 50 | 1.30e-08 | 1.74e-06 | 5.45e-07 | 1.30e-08 | 1.74e-06 | 0 |
| solovev_asymmetric | loose | 4 | 0 | 6 | converged | 5 | 6 | 12 | 22 | 49 | 8.41e-06 | 1.38e-04 | 1.43e-04 | 8.41e-06 | 1.38e-04 | 0 |
| solovev_asymmetric | loose | 6 | 0 | 8 | converged | 10 | 11 | 12 | 24 | 49 | 5.67e-07 | 1.02e-05 | 1.11e-04 | 5.67e-07 | 1.02e-05 | 0 |
| solovev_asymmetric | loose | 8 | 0 | 10 | converged | 8 | 14 | 12 | 27 | 50 | 1.20e-05 | 1.00e-03 | 4.03e-04 | 1.20e-05 | 1.00e-03 | 0 |
| solovev_asymmetric | loose | 10 | 0 | 12 | converged | 8 | 10 | 13 | 28 | 51 | 3.34e-06 | 3.92e-05 | 4.34e-04 | 3.34e-06 | 3.92e-05 | 0 |
| solovev_asymmetric | loose | 12 | 0 | 14 | converged | 9 | 13 | 13 | 31 | 55 | 8.39e-07 | 7.15e-05 | 5.58e-04 | 8.39e-07 | 7.15e-05 | 0 |
| solovev_asymmetric | exact | 4 | 0 | 6 | converged | 2 | 4 | 11 | 22 | 51 | 8.40e-06 | 1.32e-04 | 1.98e-04 | 8.40e-06 | 1.32e-04 | 0 |
| solovev_asymmetric | exact | 6 | 0 | 8 | converged | 0 | 1 | 12 | 22 | 59 | 3.13e-07 | 6.49e-06 | 7.26e-05 | 3.13e-07 | 6.49e-06 | 0 |
| solovev_asymmetric | exact | 8 | 0 | 10 | converged | 0 | 1 | 13 | 23 | 55 | 5.11e-09 | 1.34e-07 | 1.51e-06 | 5.06e-09 | 1.34e-07 | 0 |
| solovev_asymmetric | exact | 10 | 0 | 12 | converged | 0 | 1 | 13 | 23 | 57 | 8.37e-11 | 2.70e-09 | 3.03e-08 | 8.32e-11 | 2.70e-09 | 0 |
| solovev_asymmetric | exact | 12 | 0 | 14 | converged | 0 | 1 | 13 | 23 | 58 | 1.58e-12 | 1.09e-10 | 1.21e-09 | 1.58e-12 | 1.09e-10 | 0 |
| solovev_symmetric | strict | 4 | 0 | 6 | converged | 8 | 9 | 12 | 24 | 59 | 6.51e-06 | 8.87e-05 | 1.09e-04 | 6.51e-06 | 8.87e-05 | 0 |
| solovev_symmetric | strict | 6 | 0 | 8 | converged | 10 | 13 | 13 | 25 | 60 | 8.15e-08 | 2.05e-06 | 2.07e-06 | 8.14e-08 | 2.05e-06 | 0 |
| solovev_symmetric | strict | 8 | 0 | 10 | converged | 11 | 14 | 13 | 28 | 60 | 1.48e-09 | 4.08e-08 | 1.79e-07 | 1.39e-09 | 4.08e-08 | 0 |
| solovev_symmetric | strict | 10 | 0 | 12 | converged | 11 | 20 | 12 | 28 | 60 | 1.39e-07 | 1.83e-05 | 7.01e-06 | 1.39e-07 | 1.83e-05 | 0 |
| solovev_symmetric | strict | 12 | 0 | 14 | converged | 10 | 19 | 13 | 29 | 61 | 3.15e-08 | 4.97e-06 | 1.11e-06 | 3.15e-08 | 4.97e-06 | 0 |
| solovev_symmetric | loose | 4 | 0 | 6 | converged | 5 | 6 | 13 | 25 | 67 | 6.70e-06 | 1.00e-04 | 2.43e-04 | 6.70e-06 | 1.00e-04 | 0 |
| solovev_symmetric | loose | 6 | 0 | 8 | converged | 5 | 6 | 14 | 26 | 69 | 1.31e-06 | 2.80e-05 | 3.03e-04 | 1.31e-06 | 2.80e-05 | 0 |
| solovev_symmetric | loose | 8 | 0 | 10 | converged | 6 | 7 | 14 | 27 | 68 | 8.38e-07 | 1.98e-05 | 2.32e-04 | 8.38e-07 | 1.98e-05 | 0 |
| solovev_symmetric | loose | 10 | 0 | 12 | converged | 6 | 10 | 15 | 29 | 70 | 5.85e-07 | 3.51e-05 | 3.18e-04 | 5.85e-07 | 3.51e-05 | 0 |
| solovev_symmetric | loose | 12 | 0 | 14 | converged | 5 | 9 | 15 | 29 | 68 | 2.85e-07 | 2.09e-05 | 2.09e-04 | 2.85e-07 | 2.09e-05 | 0 |
| solovev_symmetric | exact | 4 | 0 | 6 | converged | 2 | 3 | 13 | 25 | 65 | 6.51e-06 | 8.71e-05 | 1.92e-04 | 6.51e-06 | 8.71e-05 | 0 |
| solovev_symmetric | exact | 6 | 0 | 8 | converged | 0 | 1 | 12 | 23 | 66 | 2.22e-07 | 4.60e-06 | 5.16e-05 | 2.22e-07 | 4.60e-06 | 0 |
| solovev_symmetric | exact | 8 | 0 | 10 | converged | 0 | 1 | 12 | 23 | 62 | 3.29e-09 | 8.67e-08 | 9.74e-07 | 3.26e-09 | 8.67e-08 | 0 |
| solovev_symmetric | exact | 10 | 0 | 12 | converged | 0 | 1 | 12 | 24 | 62 | 4.89e-11 | 1.58e-09 | 1.78e-08 | 4.86e-11 | 1.58e-09 | 0 |
| solovev_symmetric | exact | 12 | 0 | 14 | converged | 0 | 1 | 12 | 24 | 63 | 7.38e-13 | 2.83e-11 | 3.18e-10 | 7.34e-13 | 2.83e-11 | 0 |

## Timing repeats

| tag | runs | solve s (each) | median | spread (max/min) | B (each) | load at start (each) |
|---|---|---|---|---|---|---|
| ladder_integer_3d_loose_M8 | 3 | 25.2, 34.7, 42.0 | 34.7 | 1.67 | 9.44e-06, 9.44e-06, 9.44e-06 | 12.9, 25.9, 26.3 |
| ladder_sheared_A_loose_M8 | 3 | 34.3, 38.4, 51.6 | 38.4 | 1.50 | 2.44e-04, 2.44e-04, 2.44e-04 | 10.0, 27.3, 20.7 |
| ladder_sheared_A_pub_M10 | 3 | 17.4, 23.8, 26.1 | 23.8 | 1.50 | 1.74e-05, 1.74e-05, 1.74e-05 | 7.1, 23.3, 27.9 |
| ladder_solovev_asymmetric_loose_M10 | 3 | 12.7, 15.3, 17.4 | 15.3 | 1.37 | 3.34e-06, 3.34e-06, 3.34e-06 | 6.8, 16.9, 23.1 |
