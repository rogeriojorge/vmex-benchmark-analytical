## Timing: repeated runs per configuration

| case | code | protocol | NS | ftol | repeats | wall first s | warm median s | min-max s | iters (total) | fsq | B | load1 | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| integer_3d | vmec2000 | cold | 17 | 1e-08 | 2 | 2.4 | 2.4 | 2.4-2.4 | 174 | 9.8e-09 | 2.31e-03 | 31-33 |  |
| integer_3d | vmec2000 | cold | 33 | 1e-08 | 2 | 3.1 | 3.2 | 3.1-3.2 | 184 | 9.4e-09 | 1.19e-03 | 28-31 |  |
| integer_3d | vmec2000 | cold | 65 | 1e-08 | 2 | 4.7 | 4.7 | 4.7-4.8 | 191 | 9.3e-09 | 6.90e-04 | 24-28 |  |
| integer_3d | vmex | cold | 17 | 1e-08 | 2 | 7.9 | 3.6 | 3.6-7.9 | 174 | 9.8e-09 | 2.31e-03 | 15-17 | noisy |
| integer_3d | vmex | cold | 33 | 1e-08 | 2 | 7.8 | 8.6 | 7.8-8.6 | 184 | 9.4e-09 | 1.19e-03 | 17-29 |  |
| integer_3d | vmex | cold | 65 | 1e-08 | 2 | 18.7 | 9.9 | 9.9-18.7 | 191 | 9.3e-09 | 6.90e-04 | 29-42 | noisy |
| solovev_asymmetric | vmec2000 | cold | 17 | 1e-14 | 3 | 2.1 | 2.1 | 2.0-2.1 | 709 | 1.0e-14 | 2.64e-05 | 13-16 |  |
| solovev_asymmetric | vmec2000 | ladder | 33 | 1e-14 | 3 | 2.5 | 2.6 | 2.5-2.6 | 1077 | 1.0e-14 | 1.72e-05 | 15-15 |  |
| solovev_asymmetric | vmec2000 | ladder | 65 | 1e-14 | 3 | 3.7 | 3.8 | 3.7-3.9 | 1473 | 9.9e-15 | 1.34e-05 | 15-15 |  |
| solovev_asymmetric | vmec2000 | ladder | 129 | 1e-14 | 3 | 5.9 | 5.9 | 5.6-6.0 | 1874 | 1.0e-14 | 8.59e-06 | 15-16 |  |
| solovev_asymmetric | vmex | cold | 17 | 1e-14 | 3 | 5.4 | 2.4 | 2.3-5.4 | 536 | 9.4e-15 | 3.47e-05 | 7-8 | noisy |
| solovev_asymmetric | vmex | ladder | 33 | 1e-14 | 3 | 10.0 | 4.6 | 4.6-10.0 | 806 | 9.9e-15 | 1.53e-05 | 8-8 | noisy |
| solovev_asymmetric | vmex | ladder | 65 | 1e-14 | 3 | 14.6 | 7.0 | 7.0-14.6 | 1080 | 9.7e-15 | 1.23e-05 | 7-8 | noisy |
| solovev_asymmetric | vmex | ladder | 129 | 1e-14 | 3 | 23.2 | 12.3 | 12.1-23.2 | 1394 | 9.9e-15 | 8.83e-06 | 7-13 | noisy |
| solovev_symmetric | vmec2000 | cold | 17 | 1e-14 | 3 | 1.8 | 1.8 | 1.8-1.8 | 462 | 9.8e-15 | 2.43e-05 | 8-8 |  |
| solovev_symmetric | vmec2000 | ladder | 33 | 1e-14 | 3 | 1.9 | 1.9 | 1.9-2.0 | 693 | 9.6e-15 | 7.20e-06 | 7-8 |  |
| solovev_symmetric | vmec2000 | ladder | 65 | 1e-14 | 3 | 2.2 | 2.2 | 2.2-2.2 | 941 | 9.6e-15 | 9.90e-06 | 7-7 |  |
| solovev_symmetric | vmec2000 | ladder | 129 | 1e-14 | 3 | 2.8 | 2.8 | 2.7-2.8 | 1176 | 9.6e-15 | 7.85e-06 | 7-7 |  |
| solovev_symmetric | vmex | cold | 17 | 1e-14 | 3 | 4.3 | 2.0 | 1.9-4.3 | 462 | 9.8e-15 | 2.43e-05 | 6-8 | noisy |
| solovev_symmetric | vmex | ladder | 33 | 1e-14 | 3 | 7.9 | 3.6 | 3.6-7.9 | 693 | 9.6e-15 | 7.20e-06 | 8-9 | noisy |
| solovev_symmetric | vmex | ladder | 65 | 1e-14 | 3 | 11.5 | 5.7 | 5.5-11.5 | 941 | 9.6e-15 | 9.90e-06 | 8-8 | noisy |
| solovev_symmetric | vmex | ladder | 129 | 1e-14 | 3 | 16.0 | 7.6 | 7.5-16.0 | 1176 | 9.6e-15 | 7.85e-06 | 8-8 | noisy |

## Tolerance sweep (every ladder stage at the swept ftol)

| case | code | NS | protocol | ftol (all stages) | wall s | iters (total) | fsq reached | B | B/floor |
|---|---|---|---|---|---|---|---|---|---|
| integer_3d | vmec2000 | 33 | cold | 1e-10 | 4.6 | 366 | 1.0e-10 | 1.44e-03 | 941 |
| integer_3d | vmec2000 | 33 | cold | 1e-09 | 3.7 | 261 | 9.9e-10 | 1.21e-03 | 792 |
| integer_3d | vmec2000 | 33 | cold | 1e-07 | 2.8 | 134 | 9.8e-08 | 1.49e-03 | 975 |
| integer_3d | vmec2000 | 33 | cold | 1e-06 | 2.6 | 107 | 9.2e-07 | 1.86e-03 | 1217 |
| integer_3d | vmex | 33 | cold | 1e-10 | 5.3 | 366 | 1.0e-10 | 1.44e-03 | 941 |
| integer_3d | vmex | 33 | cold | 1e-09 | 4.2 | 261 | 9.9e-10 | 1.21e-03 | 792 |
| integer_3d | vmex | 33 | cold | 1e-07 | 3.9 | 134 | 9.8e-08 | 1.49e-03 | 975 |
| integer_3d | vmex | 33 | cold | 1e-06 | 9.4 (cold cache) | 107 | 9.2e-07 | 1.86e-03 | 1217 |
| solovev_asymmetric | vmec2000 | 65 | ladder | 1e-12 | 3.1 | 1182 | 9.9e-13 | 1.34e-05 | 3178 |
| solovev_asymmetric | vmec2000 | 65 | ladder | 1e-10 | 2.6 | 851 | 9.9e-11 | 1.46e-05 | 3450 |
| solovev_asymmetric | vmec2000 | 65 | ladder | 1e-08 | 2.3 | 565 | 8.7e-09 | 8.92e-05 | 21131 |
| solovev_asymmetric | vmec2000 | 65 | ladder | 1e-06 | 2.1 | 293 | 9.7e-07 | 6.30e-04 | 149272 |
| solovev_asymmetric | vmex | 65 | ladder | 1e-12 | 7.9 | 895 | 9.6e-13 | 1.28e-05 | 3042 |
| solovev_asymmetric | vmex | 65 | ladder | 1e-10 | 7.4 | 688 | 9.3e-11 | 1.33e-05 | 3141 |
| solovev_asymmetric | vmex | 65 | ladder | 1e-08 | 6.8 | 481 | 9.4e-09 | 6.07e-05 | 14372 |
| solovev_asymmetric | vmex | 65 | ladder | 1e-06 | 16.4 (cold cache) | 316 | 1.0e-06 | 1.96e-04 | 46534 |
| solovev_symmetric | vmec2000 | 65 | ladder | 1e-12 | 2.1 | 790 | 8.5e-13 | 1.02e-05 | 2447 |
| solovev_symmetric | vmec2000 | 65 | ladder | 1e-10 | 2.0 | 599 | 1.0e-10 | 1.11e-05 | 2652 |
| solovev_symmetric | vmec2000 | 65 | ladder | 1e-08 | 2.0 | 423 | 8.0e-09 | 5.62e-05 | 13461 |
| solovev_symmetric | vmec2000 | 65 | ladder | 1e-06 | 1.8 | 186 | 5.2e-07 | 5.66e-04 | 135657 |
| solovev_symmetric | vmex | 65 | ladder | 1e-12 | 5.3 | 790 | 8.5e-13 | 1.02e-05 | 2447 |
| solovev_symmetric | vmex | 65 | ladder | 1e-10 | 5.0 | 599 | 1.0e-10 | 1.11e-05 | 2652 |
| solovev_symmetric | vmex | 65 | ladder | 1e-08 | 5.1 | 423 | 8.0e-09 | 5.62e-05 | 13461 |
| solovev_symmetric | vmex | 65 | ladder | 1e-06 | 11.0 (cold cache) | 186 | 5.2e-07 | 5.66e-04 | 135657 |

## Cheapest configuration reaching each whole-volume B target

| case | code | B target | cheapest configuration | warm wall s (first) | iterations | B | fsq |
|---|---|---|---|---|---|---|---|
| integer_3d | vmex | 1e-02 | cold NS17 ftol 1e-08 | 3.6 (7.9) | 174 | 2.31e-03 | 9.8e-09 |
| integer_3d | vmex | 1e-03 | cold NS65 ftol 1e-08 | 9.9 (18.7) | 191 | 6.90e-04 | 9.3e-09 |
| integer_3d | vmex | 1e-04 | none reached (best B 6.90e-04) | - | - | - | - |
| integer_3d | vmex | 1e-05 | none reached (best B 6.90e-04) | - | - | - | - |
| integer_3d | vmec2000 | 1e-02 | cold NS17 ftol 1e-08 | 2.4 (2.4) | 174 | 2.31e-03 | 9.8e-09 |
| integer_3d | vmec2000 | 1e-03 | cold NS65 ftol 1e-08 | 4.7 (4.7) | 191 | 6.90e-04 | 9.3e-09 |
| integer_3d | vmec2000 | 1e-04 | none reached (best B 6.90e-04) | - | - | - | - |
| integer_3d | vmec2000 | 1e-05 | none reached (best B 6.90e-04) | - | - | - | - |
| solovev_asymmetric | vmex | 1e-02 | cold NS17 ftol 1e-14 | 2.4 (5.4) | 536 | 3.47e-05 | 9.4e-15 |
| solovev_asymmetric | vmex | 1e-03 | cold NS17 ftol 1e-14 | 2.4 (5.4) | 536 | 3.47e-05 | 9.4e-15 |
| solovev_asymmetric | vmex | 1e-04 | cold NS17 ftol 1e-14 | 2.4 (5.4) | 536 | 3.47e-05 | 9.4e-15 |
| solovev_asymmetric | vmex | 1e-05 | ladder NS129 ftol 1e-14 | 12.3 (23.2) | 1394 | 8.83e-06 | 9.9e-15 |
| solovev_asymmetric | vmec2000 | 1e-02 | cold NS17 ftol 1e-14 | 2.1 (2.1) | 709 | 2.64e-05 | 1.0e-14 |
| solovev_asymmetric | vmec2000 | 1e-03 | cold NS17 ftol 1e-14 | 2.1 (2.1) | 709 | 2.64e-05 | 1.0e-14 |
| solovev_asymmetric | vmec2000 | 1e-04 | cold NS17 ftol 1e-14 | 2.1 (2.1) | 709 | 2.64e-05 | 1.0e-14 |
| solovev_asymmetric | vmec2000 | 1e-05 | ladder NS129 ftol 1e-14 | 5.9 (5.9) | 1874 | 8.59e-06 | 1.0e-14 |
| solovev_symmetric | vmex | 1e-02 | cold NS17 ftol 1e-14 | 2.0 (4.3) | 462 | 2.43e-05 | 9.8e-15 |
| solovev_symmetric | vmex | 1e-03 | cold NS17 ftol 1e-14 | 2.0 (4.3) | 462 | 2.43e-05 | 9.8e-15 |
| solovev_symmetric | vmex | 1e-04 | cold NS17 ftol 1e-14 | 2.0 (4.3) | 462 | 2.43e-05 | 9.8e-15 |
| solovev_symmetric | vmex | 1e-05 | ladder NS33 ftol 1e-14 | 3.6 (7.9) | 693 | 7.20e-06 | 9.6e-15 |
| solovev_symmetric | vmec2000 | 1e-02 | ladder NS65 ftol 1e-06 (sweep) | 1.8 (1.8) | 186 | 5.66e-04 | 5.2e-07 |
| solovev_symmetric | vmec2000 | 1e-03 | ladder NS65 ftol 1e-06 (sweep) | 1.8 (1.8) | 186 | 5.66e-04 | 5.2e-07 |
| solovev_symmetric | vmec2000 | 1e-04 | cold NS17 ftol 1e-14 | 1.8 (1.8) | 462 | 2.43e-05 | 9.8e-15 |
| solovev_symmetric | vmec2000 | 1e-05 | ladder NS33 ftol 1e-14 | 1.9 (1.9) | 693 | 7.20e-06 | 9.6e-15 |
