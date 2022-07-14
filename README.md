# Powell's Method

Altered version of SciPy implementation of modified Powell's derivative free optimisation algorithm. 

See [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_powell.html) and the original implementation as detailed in [Numerical Methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_powell.html) (Chapter 10).

## Modifications

* Prints more information at each line-search and iteration to aid understanding of convergence
* Returns additionally the parameter and function values at the end of each linesearch and iteration.

