import numpy as np
import scipy.linalg as sl
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
        :raises TypeError: If params is not a numpy array, if params does not
        contain floats or if it is of the wrong size/dimension.
        """

        if not isinstance(params,np.ndarray):
            raise TypeError('params is not a numpy array')
        if not issubclass(params.dtype.type, float):
            raise TypeError('params does not contain floats')
        if params.ndim != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise TypeError('the number of elements in params are not \
                    correct')

        return self._f(*params)

    def evalGrad(self,params): #I CLAIM THIS METHOD (Hanna)
        """Evaluates the gradient of the function for the given paramaters.

        :param array params: A one dimensional array with n-values, where the
        function is to be evaluated. All values must be floats.
        :returns: The value for the gradient of the function.
        :rtype: float
        :raises TypeError: If params is not a numpy array, if params does not
        contain floats or if it is of the wrong size/dimension.
        """

        if not isinstance(params,np.ndarray):
            raise TypeError('params is not a numpy array')
        if not issubclass(params.dtype.type, float):
            raise TypeError('params does not contain floats')
        if params.ndim != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise TypeError('the number of elements in params are not \
                    correct')

        if self._g is not None:
            return self._g(*params)

        return self._secondOrderApprox(*params)

    def _secondOrderApprox(self, *params):
        gradient = np.empty(self._numArgs)
        delta = 1.e-6
        for n in range(0, self._numArgs-1):
            tempParamsLeft = list(params)
            tempParamsRight = list(params)
            tempParamsLeft[n]+=delta
            tempParamsRight[n]-=delta
            deltaFunc = self._f(*tempParamsLeft) - self._f(*tempParamsRight)
            gradient[n] = deltaFunc/(2*delta)

        return gradient








class OptimizeBase(object):

    __metaclass__ = abc.ABCMeta
    """This class is inherited by future implementaions of solvers
    """


    def __init__(self,tol=1e-6,maxIterations=200):

        self._tol=tol
        self._maxIterations = maxIterations
        self._currentValues = np.array([0,0,0])


    def __call__(self,f,startValues):
        pass

    def _step(self,f):
        pass 

    @staticmethod
    def inexactLineSearch(f,S):
        """This method performs an inexact line search based on the method
        proposed by R. Fletcher, *Practical Methods of Optimization*, vol. 1,
        Wiley, New York, 1980.

        :param Function f: The function for which the linesearch is to be
        performed.
        :param array S: The direction along which the step is to be taken, in
        the form of a numpy array containing floats.
        :returns: The step length.
        :rtype: float
        :raises TypeError: If the inparamaters are of the wrong data type, if
        the size of S is not the same as the number of arguments of f or if S
        is not a one dimensional array.
        """
        if(not isinstance(f,Function)):
            raise TypeError('f is not a Function object')
        if(not isinstance(S,np.ndarray)):
            raise TypeError('S is not a numpy array')
        if(not issubclass(S.dtype.type,float)):
            raise TypeError('S does not contain floats')
        if(not S.ndim == 1):
            raise TypeError('S must be one dimensional')
        if(not S.size == f._numArgs):
            raise TypeError('S must be one dimensional')

        return 1

class OptimizeNewton(OptimizeBase):
    """This class finds the coordinates for the smallest value of a function by
    using Newtons method.
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
        :raises ValueError: If the number of rows in A are not the same as the
        number of elements in b.
        """
        if not isinstance(A, np.ndarray):  
            raise TypeError("A must be a numpy array")
        if not isinstance(b, np.ndarray):  
            raise TypeError("b must be a numpy array")
        if not issubclass(A.dtype.type,float):
            raise TypeError("A must be an array of floats") 
        if not issubclass(b.dtype.type,float):
            raise TypeError("b must be an array of floats") 
        if A.shape[0] != len(b):
            raise ValueError("A should have as many rows as b has elements.")
        return sl.solve(A, b)

    def _approxHessian(self,f): # Labinot
        """Approximates the hessian for a function f by using a finite
        differences scheme.

        :param Function f: An object of the function class, for which the
        hessian is to be approximated.
        :raises TypeError: If f is not an instance of the Function class.
        :returns: The approximated Hessian. 
        :rtype: array
        """
        if not isinstance(f, Function):  
            raise TypeError("f must be an instance of the Function class")
            
        delta = self._tol    
        val = self._currentValues
        dim = f._numArgs
        hessian = np.zeros([dim,dim])
        
        for n in xrange(dim):
            for m in xrange(n,dim):               
                dxi = dxj = np.zeros(dim)
                dxi[n] = dxj[m] = delta
            if n == m:
                hessian[n,m] = (f(*(val+dxj)) - 2*f(*val) + f(*(val-dxj)))/(delta**2)
            else:
                hessian[n,m] = (f(*(val+dxi+dxj)) - f(*(val+dxi-dxj)) - f(*(val-dxi+dxj)) 
                + f(*(val-dxi-dxj)))/(2*delta**2)
        hessian = (hessian + np.transpose(hessian))/2
        try:
            sl.cholesky(hessian)
        except sl.LinAlgError:
            print "Matrix is not positive definite"
            return None
        return hessian

class OptimizeDFP(OptimizeBase):

    def _approximateHessian(self, f, hessian):
        if not (isinstance(f, Function)):
            raise TypeError("f must be an instance of the function class")

        delta = np.array([self._currentValues - self._previousValues])
        gamma = np.array([f.evalGrad(self._currentValues) - f.evalGrad(self._previousValues)])

        #denominator = np.dot(delta, np.transpose(gamma))
        term1 = np.dot(np.transpose(delta), delta) / np.dot(delta,np.transpose(gamma))
        term2 = np.dot(np.dot(hessian,np.transpose(gamma)),np.dot(gamma,hessian))
        denominator = np.dot(np.dot(gamma, hessian),np.transpose(gamma))
        return hessian + term1 + term2 / denominator
