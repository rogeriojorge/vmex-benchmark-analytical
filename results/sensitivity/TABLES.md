## integer_3d

VMEX caf09ee7 (0.12.0), tcon0 1, seed exact, ftol 1e-10, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 17 | cloud | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | 1.5e-11 | 3.800e+00 | FAIL | 4.07e+00 (0.001) | 3.670e-01 |
| 17 | volume | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | 1.5e-11 | 4.824e+00 | FAIL | 4.83e+00 (0.001) | 4.504e-01 |
| 17 | volume_fine | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | nan | 4.697e+00 | FAIL | - | - |
| 17 | cloud | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | 1.5e-11 | 1.927e-01 | FAIL | 2.00e-01 (0.001) | 2.925e-02 |
| 17 | volume | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | 1.5e-11 | 2.295e-01 | FAIL | 2.41e-01 (0.001) | 3.556e-02 |
| 17 | volume_fine | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | nan | 2.260e-01 | FAIL | - | - |
| 17 | cloud | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | 1.5e-11 | 2.110e+00 | FAIL | 2.35e+00 (0.001) | 3.907e-01 |
| 17 | volume | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | 1.5e-11 | 2.871e+00 | FAIL | 2.92e+00 (0.001) | 4.735e-01 |
| 17 | volume_fine | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | nan | 2.799e+00 | FAIL | - | - |
| 25 | cloud | c | 2.1e-06 / 4.0e-06 | 2.8e-05 / 3.6e-14 | 6.2e-11 | 8.684e-01 | REJECTED (root) | 8.77e-01 (0.0001) | 6.399e-01 |
| 25 | volume | c | 2.1e-06 / 4.0e-06 | 2.8e-05 / 3.6e-14 | 1.2e-11 | 1.706e+00 | REJECTED (root) | 1.70e+00 (0.001) | 4.786e-01 |
| 25 | volume_fine | c | 2.1e-06 / 4.0e-06 | 2.8e-05 / 3.6e-14 | nan | 1.483e+00 | REJECTED (root) | - | - |
| 25 | cloud | delta | 2.1e-06 / 4.0e-06 | 1.3e-05 / 2.1e-14 | 6.2e-11 | 5.837e-02 | REJECTED (root) | 7.72e-02 (0.001) | 3.726e-02 |
| 25 | volume | delta | 2.1e-06 / 4.0e-06 | 1.3e-05 / 2.1e-14 | 1.2e-11 | 1.178e-01 | REJECTED (root) | 1.17e-01 (1e-05) | 2.701e-02 |
| 25 | volume_fine | delta | 2.1e-06 / 4.0e-06 | 1.3e-05 / 2.1e-14 | nan | 9.822e-02 | REJECTED (root) | - | - |
| 25 | cloud | a | 2.1e-06 / 4.0e-06 | 2.4e-05 / 3.1e-14 | 6.2e-11 | 7.300e-01 | REJECTED (root) | 6.26e-01 (0.001) | 4.777e-01 |
| 25 | volume | a | 2.1e-06 / 4.0e-06 | 2.4e-05 / 3.1e-14 | 1.2e-11 | 1.463e+00 | REJECTED (root) | 1.43e+00 (0.001) | 4.638e-01 |
| 25 | volume_fine | a | 2.1e-06 / 4.0e-06 | 2.4e-05 / 3.1e-14 | nan | 1.282e+00 | REJECTED (root) | - | - |
| 33 | cloud | c | 7.8e-14 / 2.1e-12 | 2.6e-05 / 2.9e-14 | 7.7e-11 | 1.947e+00 | FAIL | 1.60e+00 (0.001) | 4.578e-01 |
| 33 | volume | c | 7.8e-14 / 2.1e-12 | 2.6e-05 / 2.9e-14 | 2.5e-11 | 1.637e+00 | FAIL | 1.25e+00 (0.0001) | 4.631e-01 |
| 33 | volume_fine | c | 7.8e-14 / 2.1e-12 | 2.6e-05 / 2.9e-14 | nan | 1.585e+00 | FAIL | - | - |
| 33 | cloud | delta | 7.8e-14 / 2.1e-12 | 1.8e-05 / 2.3e-14 | 7.7e-11 | 1.167e-01 | FAIL | 1.02e-01 (0.0001) | 2.051e-02 |
| 33 | volume | delta | 7.8e-14 / 2.1e-12 | 1.8e-05 / 2.3e-14 | 2.5e-11 | 7.474e-02 | FAIL | 6.69e-02 (0.001) | 2.660e-02 |
| 33 | volume_fine | delta | 7.8e-14 / 2.1e-12 | 1.8e-05 / 2.3e-14 | nan | 8.356e-02 | FAIL | - | - |
| 33 | cloud | a | 7.8e-14 / 2.1e-12 | 2.5e-05 / 2.6e-14 | 7.7e-11 | 1.378e+00 | FAIL | 1.08e+00 (0.0001) | 4.195e-01 |
| 33 | volume | a | 7.8e-14 / 2.1e-12 | 2.5e-05 / 2.6e-14 | 2.5e-11 | 1.143e+00 | FAIL | 7.80e-01 (1e-05) | 4.356e-01 |
| 33 | volume_fine | a | 7.8e-14 / 2.1e-12 | 2.5e-05 / 2.6e-14 | nan | 1.145e+00 | FAIL | - | - |

## integer_3d_aug

VMEX caf09ee7 (0.12.0), tcon0 1, seed exact, ftol 1e-10, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 17 | cloud | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | 1.5e-11 | 3.800e+00 | FAIL | 4.09e+00 (0.0001) | 3.787e-01 |
| 17 | volume | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | 1.5e-11 | 4.824e+00 | FAIL | 4.85e+00 (0.0001) | 4.546e-01 |
| 17 | volume_fine | c | 3.0e-14 / 3.9e-13 | 4.1e-05 / 4.1e-14 | nan | 4.697e+00 | FAIL | - | - |
| 17 | cloud | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | 1.5e-11 | 1.927e-01 | FAIL | 2.01e-01 (0.0001) | 2.933e-02 |
| 17 | volume | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | 1.5e-11 | 2.295e-01 | FAIL | 2.42e-01 (0.0001) | 3.575e-02 |
| 17 | volume_fine | delta | 3.0e-14 / 3.9e-13 | 1.6e-05 / 1.6e-14 | nan | 2.260e-01 | FAIL | - | - |
| 17 | cloud | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | 1.5e-11 | 2.110e+00 | FAIL | 2.36e+00 (0.0001) | 4.072e-01 |
| 17 | volume | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | 1.5e-11 | 2.871e+00 | FAIL | 2.93e+00 (0.0001) | 4.881e-01 |
| 17 | volume_fine | a | 3.0e-14 / 3.9e-13 | 2.5e-05 / 2.5e-14 | nan | 2.799e+00 | FAIL | - | - |

## integer_3d_tcon0_0

VMEX caf09ee7 (0.12.0), tcon0 0, seed exact, ftol 1e-10, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 33 | cloud | c | 1.8e-06 / 5.1e-06 | 8.4e-03 / 8.4e-12 | 2.2e-08 | 1.603e+02 | REJECTED (root) | 1.60e+02 (0.001) | 9.527e-02 |
| 33 | volume | c | 1.8e-06 / 5.1e-06 | 8.4e-03 / 8.4e-12 | 7.6e-09 | 1.084e+02 | REJECTED (root) | 1.08e+02 (1e-05) | 6.787e-01 |
| 33 | volume_fine | c | 1.8e-06 / 5.1e-06 | 8.4e-03 / 8.4e-12 | nan | 2.190e+02 | REJECTED (root) | - | - |
| 33 | cloud | delta | 1.8e-06 / 5.1e-06 | 3.0e-03 / 3.0e-12 | 2.2e-08 | 1.117e+01 | REJECTED (root) | 1.12e+01 (0.001) | 5.858e-03 |
| 33 | volume | delta | 1.8e-06 / 5.1e-06 | 3.0e-03 / 3.0e-12 | 7.6e-09 | 8.265e+00 | REJECTED (root) | 8.26e+00 (0.001) | 7.496e-03 |
| 33 | volume_fine | delta | 1.8e-06 / 5.1e-06 | 3.0e-03 / 3.0e-12 | nan | 1.714e+01 | REJECTED (root) | - | - |
| 33 | cloud | a | 1.8e-06 / 5.1e-06 | 6.8e-03 / 6.8e-12 | 2.2e-08 | 1.159e+02 | REJECTED (root) | 1.16e+02 (1e-05) | 3.105e-01 |
| 33 | volume | a | 1.8e-06 / 5.1e-06 | 6.8e-03 / 6.8e-12 | 7.6e-09 | 8.740e+01 | REJECTED (root) | 8.74e+01 (0.001) | 5.751e-02 |
| 33 | volume_fine | a | 1.8e-06 / 5.1e-06 | 6.8e-03 / 6.8e-12 | nan | 1.809e+02 | REJECTED (root) | - | - |

## integer_axisymmetric

VMEX caf09ee7 (0.12.0), tcon0 1, seed cold, ftol 1e-14, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 33 | cloud | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 5.8e-12 | 9.660e-03 | FAIL | 1.27e-07 (3e-05) | 9.660e-03 |
| 33 | volume | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 5.9e-13 | 1.058e-02 | FAIL | 2.20e-08 (0.0001) | 1.058e-02 |
| 33 | cloud | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 5.8e-12 | 1.129e-03 | FAIL | 2.57e-08 (0.0001) | 1.129e-03 |
| 33 | volume | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 5.9e-13 | 1.314e-03 | FAIL | 3.18e-09 (0.0003) | 1.314e-03 |
| 65 | cloud | c | 5.1e-14 / 5.0e-12 | 4.1e-06 / 4.2e-15 | 7.3e-12 | 1.176e-02 | FAIL | 1.25e-07 (3e-05) | 1.176e-02 |
| 65 | volume | c | 5.1e-14 / 5.0e-12 | 4.1e-06 / 4.2e-15 | 5.8e-12 | 8.351e-03 | FAIL | 6.02e-08 (3e-05) | 8.351e-03 |
| 65 | cloud | delta | 5.1e-14 / 5.0e-12 | 3.0e-06 / 3.0e-15 | 7.3e-12 | 1.582e-03 | FAIL | 2.53e-08 (0.0001) | 1.582e-03 |
| 65 | volume | delta | 5.1e-14 / 5.0e-12 | 3.0e-06 / 3.0e-15 | 5.8e-12 | 1.004e-03 | FAIL | 3.17e-09 (0.0001) | 1.004e-03 |
| 129 | cloud | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 2.1e-11 | 8.514e-03 | FAIL | 1.00e-07 (1e-05) | 8.515e-03 |
| 129 | volume | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 1.1e-11 | 6.078e-03 | FAIL | 5.04e-08 (3e-05) | 6.078e-03 |
| 129 | cloud | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 2.1e-11 | 1.306e-03 | FAIL | 1.42e-08 (0.0001) | 1.306e-03 |
| 129 | volume | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 1.1e-11 | 7.976e-04 | FAIL | 1.64e-09 (0.0001) | 7.976e-04 |
| 257 | cloud | c | 4.9e-13 / 2.0e-10 | 6.7e-06 / 6.7e-15 | 3.7e-11 | 1.410e-02 | FAIL | 1.30e-07 (0.0001) | 1.410e-02 |
| 257 | volume | c | 4.9e-13 / 2.0e-10 | 6.7e-06 / 6.7e-15 | 2.2e-10 | 1.103e-03 | FAIL | 2.35e-07 (0.0001) | 1.103e-03 |
| 257 | volume_fine | c | 4.9e-13 / 2.0e-10 | 6.7e-06 / 6.7e-15 | nan | 2.910e-03 | FAIL | - | - |
| 257 | cloud | delta | 4.9e-13 / 2.0e-10 | 5.8e-06 / 5.8e-15 | 3.7e-11 | 1.753e-03 | FAIL | 2.43e-08 (0.0001) | 1.753e-03 |
| 257 | volume | delta | 4.9e-13 / 2.0e-10 | 5.8e-06 / 5.8e-15 | 2.2e-10 | 8.997e-05 | FAIL | 3.50e-08 (0.001) | 8.997e-05 |
| 257 | volume_fine | delta | 4.9e-13 / 2.0e-10 | 5.8e-06 / 5.8e-15 | nan | 3.860e-04 | FAIL | - | - |

## integer_axisymmetric_dense

VMEX caf09ee7 (0.12.0), tcon0 1, seed cold, ftol 1e-14, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 17 | cloud | c | 7.6e-15 / 1.7e-13 | 1.4e-06 / 1.4e-15 | 3.3e-11 | 4.619e-03 | FAIL | 1.18e-07 (0.0001) | 4.619e-03 |
| 17 | volume | c | 7.6e-15 / 1.7e-13 | 1.4e-06 / 1.4e-15 | 3.1e-12 | 6.202e-03 | FAIL | 3.31e-07 (0.0001) | 6.202e-03 |
| 17 | cloud | delta | 7.6e-15 / 1.7e-13 | 1.6e-06 / 1.6e-15 | 3.3e-11 | 3.895e-04 | FAIL | 3.61e-09 (0.0001) | 3.895e-04 |
| 17 | volume | delta | 7.6e-15 / 1.7e-13 | 1.6e-06 / 1.6e-15 | 3.1e-12 | 9.005e-04 | FAIL | 8.57e-09 (0.0001) | 9.005e-04 |
| 33 | cloud | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 5.8e-12 | 9.660e-03 | FAIL | 5.05e-07 (0.0001) | 9.660e-03 |
| 33 | volume | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 2.2e-12 | 9.011e-03 | FAIL | 1.20e-07 (0.0001) | 9.011e-03 |
| 33 | cloud | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 5.8e-12 | 1.129e-03 | FAIL | 2.51e-08 (0.0001) | 1.129e-03 |
| 33 | volume | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 2.2e-12 | 1.112e-03 | FAIL | 2.54e-09 (0.0001) | 1.112e-03 |

Dense control NS 17: 568 active dofs, sigma_min 3.98e-05, cond 5.24e+07, forward-vs-reverse Jacobian 4.3e-16, tangent rel. error {'c': 3.626248326421932e-12, 'delta': 6.641970485786055e-12}, reverse rel. error {'c': 3.27721349288962e-13, 'delta': 8.501821733787961e-11}.

Dense control NS 33: 1160 active dofs, sigma_min 2.83e-05, cond 3.21e+08, forward-vs-reverse Jacobian 4.0e-16, tangent rel. error {'c': 1.2007392035235517e-11, 'delta': 3.2482529727399954e-11}, reverse rel. error {'c': 7.922086343829956e-13, 'delta': 2.107021605804422e-10}.

## integer_axisymmetric_fine

VMEX caf09ee7 (0.12.0), tcon0 1, seed cold, ftol 1e-14, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 33 | cloud | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 5.8e-12 | 9.660e-03 | FAIL | - | - |
| 33 | volume | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | 5.9e-13 | 1.058e-02 | FAIL | - | - |
| 33 | volume_fine | c | 1.9e-14 / 1.0e-12 | 3.1e-06 / 2.9e-15 | nan | 1.007e-02 | FAIL | - | - |
| 33 | cloud | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 5.8e-12 | 1.129e-03 | FAIL | - | - |
| 33 | volume | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | 5.9e-13 | 1.314e-03 | FAIL | - | - |
| 33 | volume_fine | delta | 1.9e-14 / 1.0e-12 | 2.2e-06 / 2.2e-15 | nan | 1.169e-03 | FAIL | - | - |
| 65 | cloud | c | 5.1e-14 / 5.0e-12 | 4.1e-06 / 4.2e-15 | 7.3e-12 | 1.176e-02 | FAIL | - | - |
| 65 | volume | c | 5.1e-14 / 5.0e-12 | 4.1e-06 / 4.2e-15 | 5.8e-12 | 8.351e-03 | FAIL | - | - |
| 65 | volume_fine | c | 5.1e-14 / 5.0e-12 | 4.1e-06 / 4.2e-15 | nan | 8.067e-03 | FAIL | - | - |
| 65 | cloud | delta | 5.1e-14 / 5.0e-12 | 3.0e-06 / 3.0e-15 | 7.3e-12 | 1.582e-03 | FAIL | - | - |
| 65 | volume | delta | 5.1e-14 / 5.0e-12 | 3.0e-06 / 3.0e-15 | 5.8e-12 | 1.004e-03 | FAIL | - | - |
| 65 | volume_fine | delta | 5.1e-14 / 5.0e-12 | 3.0e-06 / 3.0e-15 | nan | 1.009e-03 | FAIL | - | - |
| 129 | cloud | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 2.1e-11 | 8.514e-03 | FAIL | - | - |
| 129 | volume | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 1.1e-11 | 6.078e-03 | FAIL | - | - |
| 129 | volume_fine | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | nan | 5.159e-03 | FAIL | - | - |
| 129 | cloud | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 2.1e-11 | 1.306e-03 | FAIL | - | - |
| 129 | volume | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 1.1e-11 | 7.976e-04 | FAIL | - | - |
| 129 | volume_fine | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | nan | 6.778e-04 | FAIL | - | - |

## integer_axisymmetric_quad

VMEX caf09ee7 (0.12.0), tcon0 1, seed cold, ftol 1e-14, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 129 | cloud | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 2.1e-11 | 8.514e-03 | FAIL | - | - |
| 129 | volume | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | 1.1e-11 | 6.078e-03 | FAIL | - | - |
| 129 | volume_fine | c | 1.7e-13 / 3.3e-11 | 4.9e-06 / 4.9e-15 | nan | 5.035e-03 | FAIL | - | - |
| 129 | cloud | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 2.1e-11 | 1.306e-03 | FAIL | - | - |
| 129 | volume | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | 1.1e-11 | 7.976e-04 | FAIL | - | - |
| 129 | volume_fine | delta | 1.7e-13 / 3.3e-11 | 4.1e-06 / 4.1e-15 | nan | 6.590e-04 | FAIL | - | - |

## integer_axisymmetric_tcon0_0

VMEX caf09ee7 (0.12.0), tcon0 0, seed cold, ftol 1e-14, JAX 0.11.2; volume region s <= 0.9.

| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |
|---|---|---|---|---|---|---|---|---|---|
| 65 | cloud | c | 5.2e-14 / 6.1e-12 | 3.7e-06 / 3.8e-15 | 1.4e-09 | 3.408e-03 | FAIL | 9.73e-08 (0.0001) | 3.408e-03 |
| 65 | volume | c | 5.2e-14 / 6.1e-12 | 3.7e-06 / 3.8e-15 | 2.0e-10 | 2.069e-03 | FAIL | 3.61e-08 (0.0001) | 2.069e-03 |
| 65 | volume_fine | c | 5.2e-14 / 6.1e-12 | 3.7e-06 / 3.8e-15 | nan | 2.109e-03 | FAIL | - | - |
| 65 | cloud | delta | 5.2e-14 / 6.1e-12 | 3.0e-06 / 3.0e-15 | 1.4e-09 | 6.865e-04 | FAIL | 1.92e-08 (0.0001) | 6.865e-04 |
| 65 | volume | delta | 5.2e-14 / 6.1e-12 | 3.0e-06 / 3.0e-15 | 2.0e-10 | 4.394e-04 | FAIL | 8.99e-09 (0.0001) | 4.394e-04 |
| 65 | volume_fine | delta | 5.2e-14 / 6.1e-12 | 3.0e-06 / 3.0e-15 | nan | 4.456e-04 | FAIL | - | - |
| 129 | cloud | c | 5.2e-07 / 1.2e-06 | 5.1e-06 / 5.1e-15 | 2.5e-08 | 2.030e-03 | REJECTED (root) | 1.18e-02 (0.0001) | 1.205e-02 |
| 129 | volume | c | 5.2e-07 / 1.2e-06 | 5.1e-06 / 5.1e-15 | 2.0e-09 | 1.182e-03 | REJECTED (root) | 7.05e-03 (0.0001) | 7.019e-03 |
| 129 | volume_fine | c | 5.2e-07 / 1.2e-06 | 5.1e-06 / 5.1e-15 | nan | 1.360e-03 | REJECTED (root) | - | - |
| 129 | cloud | delta | 5.2e-07 / 1.2e-06 | 4.2e-06 / 4.2e-15 | 2.5e-08 | 2.537e-04 | REJECTED (root) | 1.28e-03 (0.0001) | 1.346e-03 |
| 129 | volume | delta | 5.2e-07 / 1.2e-06 | 4.2e-06 / 4.2e-15 | 2.0e-09 | 1.542e-04 | REJECTED (root) | 1.08e-03 (0.0001) | 1.090e-03 |
| 129 | volume_fine | delta | 5.2e-07 / 1.2e-06 | 4.2e-06 / 4.2e-15 | nan | 1.674e-04 | REJECTED (root) | - | - |

## Continuum limit (JVP metric versus NS)

| series | NS | metric | pairwise order |
|---|---|---|---|
| integer_3d/c/cloud | [17, 33] | 3.800e+00, 1.947e+00 | 0.96 |
| integer_3d/c/volume | [17, 33] | 4.824e+00, 1.637e+00 | 1.56 |
| integer_3d/c/volume_fine | [17, 33] | 4.697e+00, 1.585e+00 | 1.57 |
| integer_3d/delta/cloud | [17, 33] | 1.927e-01, 1.167e-01 | 0.72 |
| integer_3d/delta/volume | [17, 33] | 2.295e-01, 7.474e-02 | 1.62 |
| integer_3d/delta/volume_fine | [17, 33] | 2.260e-01, 8.356e-02 | 1.44 |
| integer_3d/a/cloud | [17, 33] | 2.110e+00, 1.378e+00 | 0.61 |
| integer_3d/a/volume | [17, 33] | 2.871e+00, 1.143e+00 | 1.33 |
| integer_3d/a/volume_fine | [17, 33] | 2.799e+00, 1.145e+00 | 1.29 |
| integer_axisymmetric/c/cloud | [33, 65, 129, 257] | 9.660e-03, 1.176e-02, 8.514e-03, 1.410e-02 | -0.28, 0.47, -0.73 |
| integer_axisymmetric/c/volume | [33, 65, 129, 257] | 1.058e-02, 8.351e-03, 6.078e-03, 1.103e-03 | 0.34, 0.46, 2.46 |
| integer_axisymmetric/delta/cloud | [33, 65, 129, 257] | 1.129e-03, 1.582e-03, 1.306e-03, 1.753e-03 | -0.49, 0.28, -0.42 |
| integer_axisymmetric/delta/volume | [33, 65, 129, 257] | 1.314e-03, 1.004e-03, 7.976e-04, 8.997e-05 | 0.39, 0.33, 3.15 |
| integer_axisymmetric_dense/c/cloud | [17, 33] | 4.619e-03, 9.660e-03 | -1.06 |
| integer_axisymmetric_dense/c/volume | [17, 33] | 6.202e-03, 9.011e-03 | -0.54 |
| integer_axisymmetric_dense/delta/cloud | [17, 33] | 3.895e-04, 1.129e-03 | -1.54 |
| integer_axisymmetric_dense/delta/volume | [17, 33] | 9.005e-04, 1.112e-03 | -0.30 |
| integer_axisymmetric_fine/c/cloud | [33, 65, 129] | 9.660e-03, 1.176e-02, 8.514e-03 | -0.28, 0.47 |
| integer_axisymmetric_fine/c/volume | [33, 65, 129] | 1.058e-02, 8.351e-03, 6.078e-03 | 0.34, 0.46 |
| integer_axisymmetric_fine/c/volume_fine | [33, 65, 129] | 1.007e-02, 8.067e-03, 5.159e-03 | 0.32, 0.65 |
| integer_axisymmetric_fine/delta/cloud | [33, 65, 129] | 1.129e-03, 1.582e-03, 1.306e-03 | -0.49, 0.28 |
| integer_axisymmetric_fine/delta/volume | [33, 65, 129] | 1.314e-03, 1.004e-03, 7.976e-04 | 0.39, 0.33 |
| integer_axisymmetric_fine/delta/volume_fine | [33, 65, 129] | 1.169e-03, 1.009e-03, 6.778e-04 | 0.21, 0.57 |
| integer_axisymmetric_tcon0_0/c/cloud | [65] | 3.408e-03 |  |
| integer_axisymmetric_tcon0_0/c/volume | [65] | 2.069e-03 |  |
| integer_axisymmetric_tcon0_0/c/volume_fine | [65] | 2.109e-03 |  |
| integer_axisymmetric_tcon0_0/delta/cloud | [65] | 6.865e-04 |  |
| integer_axisymmetric_tcon0_0/delta/volume | [65] | 4.394e-04 |  |
| integer_axisymmetric_tcon0_0/delta/volume_fine | [65] | 4.456e-04 |  |
| integer_3d/c/volume_fine(combined tags) | [17, 33] | 4.697e+00, 1.585e+00 | 1.57 |
| integer_3d/delta/volume_fine(combined tags) | [17, 33] | 2.260e-01, 8.356e-02 | 1.44 |
| integer_3d/a/volume_fine(combined tags) | [17, 33] | 2.799e+00, 1.145e+00 | 1.29 |
| integer_axisymmetric/c/volume_fine(combined tags) | [33, 65, 129, 257] | 1.007e-02, 8.067e-03, 5.159e-03, 2.910e-03 | 0.32, 0.65, 0.83 |
| integer_axisymmetric/delta/volume_fine(combined tags) | [33, 65, 129, 257] | 1.169e-03, 1.009e-03, 6.778e-04, 3.860e-04 | 0.21, 0.57, 0.81 |
