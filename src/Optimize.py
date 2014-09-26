import numpy as np
import inspect
import abc


class Function:
    """This class is a function which is used together with the optimisation
    routines. It takes a function with n paramaters, which returns a scalar
    value. Optionally the gradient to the function can be provided, otherwise
    it will be calculated numerically when it is evaluated.

    :param function f: The function which is to be optimised.
    :param function g: The gradient of f. This paramater is optional and if it
    is not provided it will be calculated. If the gradient is provided by the
    user it should return a numpy array of floats.
    :raises TypeError: If f and g does not have the same number of parameters.
    """

    def __init__(self,f,g=None):
        self._f=f
        self._numArgs=len(inspect.getargspec(f)[0])

        if g != None and len(inspect.getargspec(g)[0]) != self._numArgs:
            raise TypeError('f and g does not have the same number of \
            parameters')
        self._g=g

    def __call__(self,params):
        """Evaluates the function for the given paramaters.

        :param array params: A one dimensional array with n-values, where the
        function is to be evaluated. All values must be floats.
        :returns: The value for the function.
        :rtype: float
        :raises TypeError: If params is not a numpy array or if it does not
        contain floats.
        """

        if not isinstance(params,np.ndarray):
            raise TypeError('params is not a numpy array')
        if not issubclass(params.dtype.type, float):
            raise TypeError('params does not contain floats')
        if params.ndim != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise ValueError('the number of elements in params are not \
                    correct')

        return self._f(*params)

    def evalGrad(self,params): #I CLAIM THIS METHOD (Hanna)
        """Evaluates the gradient of the function for the given paramaters.

        :param array params: A one dimensional array with n-values, where the
        function is to be evaluated. All values must be floats.
        :returns: The value for the gradient of the function.
        :rtype: float
        :raises TypeError: If params is not a numpy array or if it does not
        contain floats.
        """

        if not isinstance(params,np.ndarray):
            raise TypeError('params is not a numpy array')
        if not issubclass(params.dtype.type, float):
            raise TypeError('params does not contain floats')
        if params.ndim != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise ValueError('the number of elements in params are not \
                    correct')

        if self._g != None:
            return self._g(*params)

        return self._secondOrderApprox(*params)

    def _secondOrderApprox(self, *params):
        gradient = np.empty(self._numargs)
        delta = 1.e-6
        for n in range(0, self._numargs-1):
            tempParamsLeft = copy(params)
            tempParamsRight = copy(params)
            tempParamsLeft[n]+=delta
            tempParamsRight[n]-=delta
            deltaFunc = self._f(*tempParamsLeft) - self._f(*tempParamsright)
            gradient[n] = deltaFunc/(2*delta)

        return gradient





class OptimBase(object):
    __metaclass__ = abc.ABCMeta


    def __init__(tol=1e-6,maxIterations=200):
    self.tol = tol
    self.maxIterations = maxIterations
    self.currentValues = np.array([0,0,0])
    return

    def __call__(f,startValues):
    pass

    def step(f):
    pass 

class OptimizeNewton(Optimize):
    """This class finds the coordinates for the smallest value of a function.
    """

    def _solveEquations(self,A,b): # Eli
        """Solves a system of equations on the form Ax=b, where A is a matrix
        and x and b are column matrices.

        :param array A: A numpy array of shape (m,n), containing floats.
        :param array b: A numpy array of shape (m,), containing floats.
        :returns: A numpy array of shape (m,) with the solutions for all x.
        :rtype: array
        :raises TypeError: If the matrices are not numpy arrays, containing
        floats and have the right dimensions.
        :raises ValueError: If the number of rows in are not the same as the
        number of elements in b.
        """
        pass

    def _approxHessian(self,f): # Labinot
        """Approximates the hessian for a function f by using a finite
        differences scheme.

        :param Function f: An object of the function class, for which the
        hessian is to be approximated.
        :raises TypeError: If f is not an instance of the Function class.
        :returns: The approximated Hessian. 
        :rtype: array
        """
        pass
