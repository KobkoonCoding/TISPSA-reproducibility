"""Reproducibility package for the paper
"A double inertial S-iteration algorithm with Tikhonov regularization for common fixed points and
forward-backward splitting, with applications to medical image deblurring and medical data classification".
"""
import os as _os

# The results of the paper were computed with one CPU thread; fix the thread count of the BLAS/OpenMP libraries
# before NumPy is imported by any module of the package.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "3.6.0"
