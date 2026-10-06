#!/usr/bin/env bash
# Reproduce every table and figure of the paper (about 25 minutes on a laptop).
set -e
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_all.py results
python verify.py results expected_results
