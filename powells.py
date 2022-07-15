################################################################
# Minimisation of a generic function using the derivative free #
# Powell's method. Implementation taken from SciPy, modified   #
# to return details of the iterations.                         #
# Alex Lipp 13/7/2022 (alex@lipp.org.uk)                       #
################################################################

from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf, asfarray, isinf)
import numpy as np
from datetime import datetime
import time as time 

# Optimisation functions

def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def brent(func, args=(), brack=None, tol=1.48e-8, full_output=0, maxiter=500):
    """
    Given a function of one variable and a possible bracket, return
    the local minimum of the function isolated to a fractional precision
    of tol.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.
    args : tuple, optional
        Additional arguments (if present).
    brack : tuple, optional
        Either a triple (xa,xb,xc) where xa<xb<xc and func(xb) <
        func(xa), func(xc) or a pair (xa,xb) which are used as a
        starting interval for a downhill bracket search (see
        `bracket`). Providing the pair (xa,xb) does not always mean
        the obtained solution will satisfy xa<=x<=xb.
    tol : float, optional
        Stop if between iteration change is less than `tol`.
    full_output : bool, optional
        If True, return all output args (xmin, fval, iter,
        funcalls).
    maxiter : int, optional
        Maximum number of iterations in solution.

    Returns
    -------
    xmin : ndarray
        Optimum point.
    fval : float
        Optimum value.
    iter : int
        Number of iterations.
    funcalls : int
        Number of objective function evaluations made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Brent' `method` in particular.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    Does not ensure that the minimum lies in the range specified by
    `brack`. See `fminbound`.

    Examples
    --------
    We illustrate the behaviour of the function when `brack` is of
    size 2 and 3 respectively. In the case where `brack` is of the
    form (xa,xb), we can see for the given values, the output need
    not necessarily lie in the range (xa,xb).

    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.brent(f,brack=(1,2))
    >>> minimum
    0.0
    >>> minimum = optimize.brent(f,brack=(-1,0.5,2))
    >>> minimum
    -2.7755575615628914e-17

    """
    options = {'xtol': tol,
               'maxiter': maxiter}
    res = _minimize_scalar_brent(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['nfev']
    else:
        return res['x']

class Brent:
    #need to rethink design of __init__
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500,
                 full_output=0):
        self.func = func
        self.args = args
        self.tol = tol
        self.maxiter = maxiter
        self._mintol = 1.0e-11
        self._cg = 0.3819660
        self.xmin = None
        self.fval = None
        self.iter = 0
        self.funcalls = 0

    # need to rethink design of set_bracket (new options, etc.)
    def set_bracket(self, brack=None):
        self.brack = brack

    def get_bracket_info(self):
        #set up
        func = self.func
        args = self.args
        brack = self.brack
        ### BEGIN core bracket_info code ###
        ### carefully DOCUMENT any CHANGES in core ##
        if brack is None:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
        elif len(brack) == 2:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                       xb=brack[1], args=args)
        elif len(brack) == 3:
            xa, xb, xc = brack
            if (xa > xc):  # swap so xa < xc can be assumed
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError("Not a bracketing interval.")
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError("Not a bracketing interval.")
            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        ### END core bracket_info code ###

        return xa, xb, xc, fa, fb, fc, funcalls

    def optimize(self):
        # set up for optimization
        func = self.func
        xa, xb, xc, fa, fb, fc, funcalls = self.get_bracket_info()
        _mintol = self._mintol
        _cg = self._cg
        #################################
        #BEGIN CORE ALGORITHM
        #################################
        x = w = v = xb
        fw = fv = fx = func(*((x,) + self.args))
        if (xa < xc):
            a = xa
            b = xc
        else:
            a = xc
            b = xa
        deltax = 0.0
        funcalls += 1
        iter = 0
        while (iter < self.maxiter):
            tol1 = self.tol * np.abs(x) + _mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            # check for convergence
            if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
                break
            # XXX In the first iteration, rat is only bound in the true case
            # of this conditional. This used to cause an UnboundLocalError
            # (gh-4140). It should be set before the if (but to what?).
            if (np.abs(deltax) <= tol1):
                if (x >= xmid):
                    deltax = a - x       # do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax
            else:                              # do a parabolic step
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if (tmp2 > 0.0):
                    p = -p
                tmp2 = np.abs(tmp2)
                dx_temp = deltax
                deltax = rat
                # check parabolic fit
                if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                        (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                    rat = p * 1.0 / tmp2        # if parabolic step is useful.
                    u = x + rat
                    if ((u - a) < tol2 or (b - u) < tol2):
                        if xmid - x >= 0:
                            rat = tol1
                        else:
                            rat = -tol1
                else:
                    if (x >= xmid):
                        deltax = a - x  # if it's not do a golden section step
                    else:
                        deltax = b - x
                    rat = _cg * deltax

            if (np.abs(rat) < tol1):            # update by at least tol1
                if rat >= 0:
                    u = x + tol1
                else:
                    u = x - tol1
            else:
                u = x + rat
            fu = func(*((u,) + self.args))      # calculate new output value
            funcalls += 1

            if (fu > fx):                 # if it's bigger than current
                if (u < x):
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            else:
                if (u >= x):
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu

            iter += 1
        #################################
        #END CORE ALGORITHM
        #################################

        self.xmin = x
        self.fval = fx
        self.iter = iter
        self.funcalls = funcalls

    def get_result(self, full_output=False):
        if full_output:
            return self.xmin, self.fval, self.iter, self.funcalls
        else:
            return self.xmin    
    
def _minimize_scalar_brent(func, brack=None, args=(),
                           xtol=1.48e-8, maxiter=500,
                           **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    """
    tol = xtol
    if tol < 0:
        raise ValueError('tolerance should be >= 0, got %r' % tol)

    brent = Brent(func=func, args=args, tol=tol,
                  full_output=True, maxiter=maxiter)
    brent.set_bracket(brack)
    brent.optimize()
    x, fval, nit, nfev = brent.get_result(full_output=True)

    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev,
                          success=success)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimium of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        alpha_min, fret, _, _ = brent(myfunc, full_output=1, tol=tol)
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi


def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1,  callback=None,
                direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func.
    xtol : float, optional
        Line-search error tolerance.
    ftol : float, optional
        Relative error in ``func(xopt)`` acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        If True, ``fopt``, ``xi``, ``direc``, ``iter``, ``funcalls``, and
        ``warnflag`` are returned.
    disp : bool, optional
        If True, print convergence messages.
    callback : callable, optional
        An optional user-supplied function, called after each
        iteration.  Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    direc : ndarray, optional
        Initial fitting step and parameter order set as an (N, N) array, where N
        is the number of fitting parameters in `x0`. Defaults to step size 1.0
        fitting all parameters simultaneously (``np.ones((N, N))``). To
        prevent initial consideration of values in a step or to change initial
        step size, set to 0 or desired step size in the Jth position in the Mth
        block, where J is the position in `x0` and M is the desired evaluation
        step, with steps being evaluated in index order. Step size and ordering
        will change freely as minimization proceeds.

    Returns
    -------
    xopt : ndarray
        Parameter which minimizes `func`.
    fopt : number
        Value of function at minimum: ``fopt = func(xopt)``.
    direc : ndarray
        Current direction set.
    iter : int
        Number of iterations.
    funcalls : int
        Number of function calls made.
    warnflag : int
        Integer warning flag:
            1 : Maximum number of function evaluations.
            2 : Maximum number of iterations.
            3 : NaN result encountered.
            4 : The result is out of the provided bounds.
    allvecs : list
        List of solutions at each iteration.

    See also
    --------
    minimize: Interface to unconstrained minimization algorithms for
        multivariate functions. See the 'Powell' method in particular.

    Notes
    -----
    Uses a modification of Powell's method to find the minimum of
    a function of N variables. Powell's method is a conjugate
    direction method.

    The algorithm has two loops. The outer loop merely iterates over the inner
    loop. The inner loop minimizes over each current direction in the direction
    set. At the end of the inner loop, if certain conditions are met, the
    direction that gave the largest decrease is dropped and replaced with the
    difference between the current estimated x and the estimated x from the
    beginning of the inner-loop.

    The technical conditions for replacing the direction of greatest
    increase amount to checking that

    1. No further gain can be made along the direction of greatest increase
       from that iteration.
    2. The direction of greatest increase accounted for a large sufficient
       fraction of the decrease in the function value from that iteration of
       the inner loop.

    References
    ----------
    Powell M.J.D. (1964) An efficient method for finding the minimum of a
    function of several variables without calculating derivatives,
    Computer Journal, 7 (2):155-162.

    Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
    Numerical Recipes (any edition), Cambridge University Press

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin_powell(f, -1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 2
             Function evaluations: 18
    >>> minimum
    array(0.0)

    """
    opts = {'xtol': xtol,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'direc': direc}

    res = minimize_powell(func, x0, args, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['direc'], res['nit'],
                   res['nfev'], res['status'])
        return retlist
    else:
         return res['x']

def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_array_scalar(x1) and is_array_scalar(x2)):
        raise ValueError("Optimization bounds must be scalars"
                         " or array scalars.")
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.max([np.abs(rat), tol1])
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num)

    return result        

def minimize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm. 
    
    See Numerical Methods (any edition) for details of the algorithm. 
    

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    bounds : `Bounds`
        If bounds are not provided, then an unbounded line search will be used.
        If bounds are provided and the initial guess is within the bounds, then
        every function evaluation throughout the minimization procedure will be
        within the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (or left to default), then some
        function evaluations during the first iteration may be outside the
        bounds, but every function evaluation after the first iteration will be
        within the bounds. If `direc` is not full rank, then some parameters may
        not be optimized and the solution is not guaranteed to be within the
        bounds.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    start=time.time()
    xs = [] # Collate the intermediate x values as we go
    fs = [] # Collate the intermediate f values as we go
    xs_it = [] # Collate the intermediate x values at each iteration
    fs_it = [] # Collate the intermediate f values at each iteration
    maxfun = maxfev
    # we need to use a mutable object here that we can update in the
    # wrapper function
    fcalls, func = wrap_function(func, args)
    x = asarray(x0).flatten()
    xs+=[x]
    xs_it += [x]
    N = len(x)
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    if direc is None:
        direc = eye(N, dtype=float)
    else:
        direc = asarray(direc, dtype=float)
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn("direc input is not full rank, some parameters may "
                          "not be optimized",
                          OptimizeWarning, 3)

    if bounds is None:
        # don't make these arrays of all +/- inf. because
        # _linesearch_powell will do an unnecessary check of all the elements.
        # just keep them None, _linesearch_powell will not have to check
        # all the elements.
        lower_bound, upper_bound = None, None
    else:
        # bounds is standardized in _minimize.py.
        lower_bound, upper_bound = bounds.lb, bounds.ub
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, 3)

    fval = squeeze(func(x))
    fs+=[fval]
    fs_it+=[fval]
    x1 = x.copy()
    iter = 0
    ilist = list(range(N))
    while True:
        fx = fval
        bigind = 0
        delta = 0.0
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Starting iteration",iter+1)
        print("Time",datetime.now())
        print("Search directions:")
        print(direc)
        for i in ilist:
            print("\tMinimising along direction",i+1,"of",N)
            print("\t",datetime.now())
            direc1 = direc[i]
            fx2 = fval
            fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                 tol=xtol * 100,
                                                 lower_bound=lower_bound,
                                                 upper_bound=upper_bound,
                                                 fval=fval)
            xs += [x]
            fs += [fval]
            
            if (fx2 - fval) > delta:
                delta = fx2 - fval
                bigind = i
        iter += 1
        if callback is not None:
            callback(x)
        bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
        if 2.0 * (fx - fval) <= bnd: # Break if 2*(x1-x2)/(x1+x2) < tol. i.e. exit if change in f is too small            
            print("#####",fx - fval,"#####")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Tolerance threshold reached\nQuitting...")
            xs_it += [x]
            fs_it += [fval]
            break
        if fcalls[0] >= maxfun:
            break
        if iter >= maxiter:
            break
        if np.isnan(fx) and np.isnan(fval):
            # Ended up in a nan-region: bail out
            break
        
        # Construct the extrapolated point
        direc1 = x - x1
        x2 = 2*x - x1
        x1 = x.copy()
        fx2 = squeeze(func(x2))

        if (fx > fx2): # If f(x2) is no better than f(x) we keep the original set of directions [See eq. 10.5.7 in Num. Rec]
            t = 2.0*(fx + fx2 - 2.0*fval)
            temp = (fx - fval - delta)
            t *= temp*temp
            temp = fx - fx2
            t -= delta*temp*temp
            # If 2(f0-2fN+fE)[(f0-fN)-Df]^2 >= (f0-fE)^2Df then keep original set of directions. [See eq. 10.5.7 in Num. Rec]
            if t < 0.0: 
                print("\tMinimising along an extrapolated direction")
                print("\t",datetime.now())
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                xs += [x]
                fs += [fval]
                if np.any(direc1):
                    direc[bigind] = direc[-1]
                    direc[-1] = direc1
                    
        xs_it += [x]
        fs_it += [fval]
        print("Completed iteration",iter)
        print("Current parameters =",x)


    warnflag = 0
    # out of bounds is more urgent than exceeding function evals or iters,
    # but I don't want to cause inconsistencies by changing the
    # established warning flags for maxfev and maxiter, so the out of bounds
    # warning flag becomes 3, but is checked for first.
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print("Warning: " + msg)
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']
        if disp:
            print("Warning: " + msg)
    else:
        msg = _status_message['success']
        if disp:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iter)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x,xs=asarray(xs),fs=fs,
                           xs_it = asarray(xs_it), fs_it = asarray(fs_it))
    
    # fval = function value 
    # direc = final direction set 
    # nit = number of iterations 
    # nfev = number of function calls
    # status = exit status 
    # success = Did algorithm exit successfully 
    # message = message of algorithm 
    # x = optimal parameters 
    # xs = optimal x value in every search direction 
    # fs = function values at the above parameter values 
    # xs = optimal x value in end of every *iteration* 
    # fs = function values at the above iteration parameter values
    print("Total runtime:",np.round(time.time()-start,1),"s")
    print("Final result:",result['x'])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return result

#################################
# Below are auxiliary functions #
#################################

def is_array_scalar(x):
    """Test whether `x` is either a scalar or an array scalar.

    """
    return np.size(x) == 1



_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
    
    
# Auxiliary functions for dealing with bounds

def standardize_bounds(bounds):
    """Converts bounds to the form required by the solver."""
    if not isinstance(bounds, Bounds):
        lb, ub = old_bound_to_new(bounds)
        bounds = Bounds(lb, ub)
    return bounds

class Bounds(object):
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : array_like, optional
        Lower and upper bounds on independent variables. Each array must
        have the same size as x or be a scalar, in which case a bound will be
        the same for all the variables. Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def __init__(self, lb, ub, keep_feasible=False):
        self.lb = lb
        self.ub = ub
        self.keep_feasible = keep_feasible

    def __repr__(self):
        if np.any(self.keep_feasible):
            return "{}({!r}, {!r}, keep_feasible={!r})".format(type(self).__name__, self.lb, self.ub, self.keep_feasible)
        else:
            return "{}({!r}, {!r})".format(type(self).__name__, self.lb, self.ub)
        
def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are None they are replaced by
    -np.inf/np.inf.
    """
    lb, ub = zip(*bounds)
    lb = np.array([float(x) if x is not None else -np.inf for x in lb])
    ub = np.array([float(x) if x is not None else np.inf for x in ub])
    return lb, ub        


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper    