# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
import cython_np_predict

def calculate_gradients(np.ndarray[np.float64_t, ndim=2] x, 
                        np.ndarray[np.float64_t, ndim=2] y, 
                        np.ndarray[np.float64_t, ndim=2] w, 
                        float b):
    cdef double grad_b
    error = cython_np_predict.predict(x, w, b) - y
    grad_w = 2 * x.T @ error
    grad_b = 2 * np.sum(error)
    return grad_w, grad_b