# environment for MRX runs on a CPU box (float64): source this with PY set to a python that has jax, diffrax, equinox,
# scipy, h5py, netCDF4, e.g.  PY=/path/to/venv/bin/python source office_env.sh ; MRX clone expected at ~/mrxJ-office/mrx-lopez
export PYTHONPATH=$HOME/mrxJ-office/mrx-lopez
export MRX_DTYPE=float64 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
: "${PY:?set PY to a python with jax, diffrax and equinox}"
export PY
