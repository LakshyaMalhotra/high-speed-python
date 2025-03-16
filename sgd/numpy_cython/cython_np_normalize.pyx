# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np

def normalize(np.ndarray[np.float64_t, ndim=2] features):
    cdef np.ndarray[np.float64_t, ndim=1] feat_min = np.min(features, axis=0)
    cdef np.ndarray[np.float64_t, ndim=1] feat_max = np.max(features, axis=0)
    return (features - feat_min) / (
        feat_max - feat_min + 1e-8
    )