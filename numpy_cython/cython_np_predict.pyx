# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np

def predict(np.ndarray[np.float64_t, ndim=2]features, 
            np.ndarray[np.float64_t, ndim=2]w, 
            float b):
    # cdef double[:, :] out = features @ w + b
    # return out
    return features @ w + b