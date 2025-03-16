# Speed-up Your Python Code

Code repository for the blog post on optimizing Python code performance. In this project, I explore various techniques to enhance the efficiency of Python code, specifically focusing on implementing Stochastic Gradient Descent (SGD).

## Overview

- A baseline implementation of SGD in pure Python
- A highly efficient, vectorized implementation using NumPy
- Experiments with compilation techniques using Cython and Numba

## Key Findings

1. **Pure Python Implementation**: The baseline implementation provides a clear and straightforward approach but lacks performance efficiency.
2. **Vectorized Implementation with NumPy**: This approach leverages NumPy's ability to perform operations on entire arrays at once, resulting in significant performance gains.
3. **Compilation with Cython and Numba**: While these techniques can be powerful, they did not yield substantial improvements in this specific case.

## Repository Structure

- `sgd_pure_python.py`: Baseline implementation of SGD in pure Python
- `sgd_numpy.py`: Vectorized implementation of SGD using NumPy
- `sgd_numpy_cython.py`: SGD implementation using Cython
- `sgd_numpy_numba.py`: SGD implementation using Numba

## How to Run

1. **Baseline Implementation**:
   ```bash
   python sgd_pure_python.py
   ```

2. **NumPy Implementation**:
   ```bash
   python sgd_numpy.py
   ```

3. **Cython Implementation**:
   First, cd to the directory containing Cython code:
   ```bash
   cd numpy_cython
   ```
   Then run the script:
   ```bash
   python sgd_numpy_cython.py
   ```

4. **Numba Implementation**:
   ```bash
   python sgd_numpy_numba.py
   ```

## Conclusion

The vectorized implementation with NumPy proved to be the most effective in optimizing the performance of the SGD algorithm. This project highlights the importance of leveraging specialized libraries like NumPy for numerical computations in Python.

For more details, check out the full blog post: [Speed-up Your Python Code](#your_blog_link_here#)

## License

This project is licensed under the MIT License.
