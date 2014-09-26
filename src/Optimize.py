import numpy as np
import inspect

class Function:
    """This class is a function which is used together with the optimisation
    routines. It takes a function with n paramaters, which returns a scalar
    value. Optionally the gradient to the function can be provided, otherwise
    it will be calculated numerically when it is evaluated.

    :param function f: The function which is to be optimised.
    :param function g: The gradient of f. This paramater is optional and if it
    is not provided it will be calculated.
    :raises TypeError: If f and g does not have the same number of parameters.
    """

    def __init__(self,f,g=None):
        self._f=f
        self._numArgs=len(inspect.getargspec(f)[0])

        if g != None and len(inspect.getargspec(g)[0]) != self._numArgs:
            raise TypeError('f and g does not have the same number of
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
        if params.ndmin != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise ValueError('the number of elements in params are not
                    correct')

        return self._f(*params)
    
    def evalGrad(self,params):
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
        if params.ndmin != 1:
            raise TypeError('params is not one dimensional')
        if params.shape != (self._numArgs,):
            raise ValueError('the number of elements in params are not
                    correct')

        if self._g != None:
            return g(*params)

        #Calculate the numerecial value of g
