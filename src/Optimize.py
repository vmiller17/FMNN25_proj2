import numpy as np
import scipy.linalg as sl
import inspect
import abc
import sys
from scipy.optimize import minimize_scalar

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

        if (g is not None) and (len(inspect.getargspec(g)[0]) is not self._numArgs):
            raise TypeError('f and g does not have the same number of \
            parameters')
        if g is None:
            self._g = self._approximateGrad
        else:
            self._g = g


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
            raise TypeError('the number of elements in params are not correct')

        return self._g(*params)

    def _approximateGrad(self, *params):
        
        dim = self._numArgs
        delta = 4.8e-8
       
        gradient = np.zeros(dim)
        for n in xrange(dim):
            dx = np.zeros(dim)
            dx[n] = delta
            gradient[n] = (self._f(*(params+dx)) - self._f(*(params-dx)))/(2.*delta)
        return gradient




class OptimizeBase(object):

    __metaclass__ = abc.ABCMeta
    """This class is inherited by future implementaions of solvers
    """


    def __init__(self,tol=1e-6,maxIterations=200):
        
        if (not isinstance(tol,float)): 
           raise TypeError('tol is not a float')            
        if (not isinstance(maxIterations,int)):
            raise TypeError('maxIterations is not a int')
        self._tol=tol
        self._maxIterations = maxIterations
        self.currentValues = None  
        self.currentGrad = None #Has to be initialized here, why?


    def __call__(self,f,startValues):
        """Minimizes the function f with a initial guess x0
        
        :param Function f: An object of the function class which is to be minimized
        :raises TypeError: If f is not an instance of the Function class
        :raises ValueError: If 
        :returns: The point where the function f has its local minimum
        :rtype: array
        """
        
        if not isinstance(f, Function):  
            raise TypeError("f must be an instance of the Function class")
            
        
        self.currentValues = startValues
        nbrOfIter = 0
        self.currentGrad = f.evalGrad(startValues)
        
        while sl.norm(self.currentGrad) > self._tol and nbrOfIter < self._maxIterations:
            self.currentGrad = f.evalGrad(self.currentValues)
            self.currentValues = self._step(f)
            nbrOfIter = nbrOfIter + 1
        return self.currentValues    

    @abc.abstractmethod
    def _step(self,f): 
        """Takes a step towards the solution.
        
        :param Function f: An object of the function class.
        :raises TypeErro: If f is not an instance of the function class or if 
        currentGrad is not an np.array
        :raises ValueError: If currentGrad does not contain floats.
        """
        return 
    
    @abc.abstractmethod
    def _approxHessian(self,f):
        """Gives an approxmation of the Hessian
        """
        return
        

    @staticmethod
    def inexactLineSearch(f,x,S,rho=0.1,sigma=0.7,tau=0.1,chi=9.0):
        """This method performs an inexact line search based on the method
        proposed by R. Fletcher, *Practical Methods of Optimization*, vol. 1,
        Wiley, New York, 1980.

        :param Function f: The function for which the linesearch is to be
        performed.
        :param array x: The current value of x as a one dimensional numpy array
        of floats.
        :param array S: The direction along which the step is to be taken, in
        the form of a numpy array containing floats.
        :param float rho: Tuning parameter for the algarihtm. Default is 0.1
        :param float sigma: Tuning parameter for the algarihtm. Default is 0.7
        :param float tau: Tuning parameter for the algarihtm. Default is 0.1
        :param float chi: Tuning parameter for the algarihtm. Default is 9
        :returns: The step length.
        :rtype: float
        :raises TypeError: If the inparamaters are of the wrong data type, if
        the size of S, or x, is not the same as the number of arguments of f
        or if S or x is not a one dimensional array.
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
            raise TypeError('S must have the same size as the number of \
            arguments of f')
        if(not isinstance(x,np.ndarray)):
            raise TypeError('x is not a numpy array')
        if(not issubclass(x.dtype.type,float)):
            raise TypeError('x does not contain floats')
        if(not x.ndim == 1):
            raise TypeError('x must be one dimensional')
        if(not x.size == f._numArgs):
            raise TypeError('x must have the same size as the number of \
            arguments of f')
        aL = 0
        aU = sys.float_info.max
        fL = f(x + aL*S)
        dfL = f.evalGrad(x + aL*S).dot(S)
        a0 = 2

        #The following loop is very ugly. It is directly copied from the
        #algorihtm in the book, but it should be written better. This is a
        #job for a later time
        stop = False
        f0 = f(x + a0*S)
        df0 = f.evalGrad(x + a0*S).dot(S)
        while((f0 > fL + rho*(a0 - aL)*dfL) or (df0 < sigma*dfL)):
            while(f0 > fL + rho*(a0 - aL)*dfL):
                if(a0 < aU): aU = a0
                a0hat = aL + (dfL*(a0 - aL)**2)/(2*(fL - f0 + (a0 - aL)*dfL))
                if(a0hat < aL + tau*(aU - aL)): a0hat = aL + tau*(aU - aL)
                if(a0hat > aU - tau*(aU - aL)): a0hat = aU - tau*(aU - aL)
                a0 = a0hat
                f0 = f(x + a0*S)


            df0 = f.evalGrad(x + a0*S).dot(S)
            if(df0 < sigma*dfL):
                deltaa0 = (a0 - aL)*df0/(dfL - df0)
                if(deltaa0 < tau*(a0 - aL)): deltaa0 = tau*(a0 - aL)
                if(deltaa0 > chi*(a0 - aL)): deltaa0 = chi*(a0 - aL)
                a0hat = a0 + deltaa0
                aL = a0
                a0 = a0hat
                fL = f0
                dfL = df0
                f0 = f(x + a0*S)

        return a0

    @staticmethod
    def exactLineSearch(f,x,S):
        """
        :param Function f: An object of the Function class. F is called with numpy array of shape (m,).
        :param array x: A numpy array of shape (m,), containing floats. x = currentValues in OptimizeBase.
        :param array S: A numpy array of shape (m,), containing floats. S is the newton direction.
        :returns alpha s.t. fi(alpha)=f(x+alpha*S) is minimized.
        :rtype: float
        :raises TypeError: If the inparamaters are of the wrong data type, if
        the size of S, or x, is not the same as the number of arguments of f
        or if S or x is not a one dimensional array.
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
            raise TypeError('S must have the same size as the number of \
            arguments of f')
        if(not isinstance(x,np.ndarray)):
            raise TypeError('x is not a numpy array')
        if(not issubclass(x.dtype.type,float)):
            raise TypeError('x does not contain floats')
        if(not x.ndim == 1):
            raise TypeError('x must be one dimensional')
        if(not x.size == f._numArgs):
            raise TypeError('x must have the same size as the number of \
            arguments of f')
        def fi(alpha):
            return f(x+alpha*S)
        return minimize_scalar(fi).x




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
            
            
        return sl.solve(A, b) #Should we check with cholesky first to rasie error? 


    def _approxHessian(self,f):
        """Approximates the hessian for a function f by using a finite
        differences scheme.

        :param Function f: An object of the function class, for which the
        hessian is to be approximated.
        :param array currentGrad: the value of the gradient in the current
        point
        :raises TypeError: If f is not an instance of the Function class.
        :returns: The approximated Hessian. 
        :rtype: array
        """
        if not isinstance(f, Function):  
            raise TypeError("f must be an instance of the Function class")
            
        if (self.currentGrad is None): #Problem when testing only this function
            self.currentGrad = f.evalGrad(self.currentValues)
  
        dim = f._numArgs
        hessian = np.zeros([dim,dim])
        delta = 1.e-4
        
        
        for n in xrange(dim):
                dx = np.zeros(dim)
                dx[n]  = delta
                hessian[n] = (self.currentGrad+dx) - (self.currentGrad-dx)

        hessian = (hessian + np.transpose(hessian))/(2*delta)

        
        try:
            sl.cholesky(hessian)
        except sl.LinAlgError:
            print "Matrix is not positive definite" #Raise an exeption here
            raise ValueError('The Hessian is not positive definite')
            return None
            
        return hessian
        
    def _step(self,f): #Added grad here, seems better to pass along then calculate again.
        """Takes a step towards the solution.
        
        :param Function f: An object of the function class.
        :param array currentGrad: A ndarray of the gradient in the currnet point.
        :raises TypeErro: If f is not an instance of the function class or if 
        currentGrad is not an np.array
        :raises ValueError: If currentGrad does not contain floats.
        """
        
        if not isinstance(f, Function):
            raise TypeError('f must be an instance of the Function class')
        if not isinstance(self.currentGrad,np.ndarray):
            raise TypeError('currentGrad is not a numpy array')
        if not issubclass(self.currentGrad.dtype.type, float):
            raise ValueError('currentGrad does not contain floats')

        H = self._approxHessian(f)
        S = np.dot(H,self.currentGrad)   
        alpha = OptimizeBase.exactLineSearch(f,self.currentValues,S)
        val = self.currentValues + alpha*S
        
        return val 

class OptimizeDFP(OptimizeBase):

    def __call__(self, f, startValues):
        """
        Minimizes the function f with a initial guess x0
        :param Function f: An object of the function class which is to be minimized
        :raises TypeError: If f is not an instance of the Function class
        :raises ValueError: If
        :returns: The point where the function f has its local minimum
        :rtype: array
        """

        if not isinstance(f, Function):
            raise TypeError("f must be an instance of the Function class")

        self.currentValues = startValues
        nbrIter = 1
        self._currentGrad = f.evalGrad(startValues)
        self._currentH = np.eye(f._numArgs)
        tempValues = self._step(f)
        self._previousValues = np.copy(self.currentValues)
        self.currentValues = tempValues
        self._previousGrad = np.copy(self._currentGrad)
        self._currentGrad = f.evalGrad(self.currentValues)
        while sl.norm(self._currentGrad) > self._tol and nbrIter < self._maxIterations:
            self._currentH = self._approximateHessian(f)
            self._previousValues = np.copy(self.currentValues)
            self.currentValues = self._step(f)
            self._previousGrad = np.copy(self._currentGrad)
            self._currentGrad = f.evalGrad(self.currentValues)
            nbrIter += 1

        return self.currentValues

    def _step(self, f):
        """Takes a step towards the solution.
        :param Function f: An object of the function class.
        """

        S = np.dot(self._currentH, self._currentGrad)
        alpha = OptimizeBase.inexactLineSearch(f, self.currentValues, S)
        val = self.currentValues + alpha*S
        return val

    def _approxHessian(self, f):
        if not (isinstance(f, Function)):
            raise TypeError("f must be an instance of the function class")

        delta = np.array([self.currentValues - self._previousValues])
        gamma = np.array([f.evalGrad(self.currentValues) - f.evalGrad(self._previousValues)])
        term1 = np.dot(np.transpose(delta), delta) / np.dot(delta,np.transpose(gamma))
        term2 = np.dot(np.dot(hessian,np.transpose(gamma)),np.dot(gamma,hessian))
        denominator = np.dot(np.dot(gamma, hessian),np.transpose(gamma))
        return hessian + term1 + term2 / denominator



class OptimizeBroydenGood(OptimizeBase):
    
        
    def _step(self,f):
    
        if not hasattr(self, '_currentHessInv'):
            self._currentHessInv = np.eye(f._numArgs)
            self._i = 0
        self._i = self._i + 1
        s = -self._currentHessInv.dot(f.evalGrad(self.currentValues))
        alpha = self.inexactLineSearch(f,self.currentValues,s)
        delta = alpha*s
        nextValues = self.currentValues + delta
        gamma = f.evalGrad(nextValues) - f.evalGrad(self.currentValues)
        
       
        u = delta - np.dot(self._currentHessInv, gamma)       
        a = 1/np.dot(u, gamma)        
        nextHessInv = self._currentHessInv + a*np.outer(u,u)
        self._currentHessInv = nextHessInv
        
                     
        return nextValues
        
    def _approxHessian(self,f):
        """Gives an approxmation of the Hessian
        """
        return
        
        
class OptimizeBroydenBad(OptimizeBase):
    

    def _step(self,f):
    
        if not hasattr(self, '_currentHessInv'):
            self._currentHessInv = np.eye(f._numArgs)
            self._i = 0
        self._i = self._i + 1
        s = -self._currentHessInv.dot(f.evalGrad(self.currentValues))
        alpha = self.inexactLineSearch(f,self.currentValues,s)
        delta = alpha*s
        nextValues = self.currentValues + delta
        gamma = f.evalGrad(nextValues) - f.evalGrad(self.currentValues)
        
       
        u = delta - np.dot(self._currentHessInv, gamma)       
        a = 1/np.dot(gamma, gamma);        
        nextHessInv = self._currentHessInv + a*np.outer(u,gamma)
        self._currentHessInv = nextHessInv
        
                     
        return nextValues
        
    def _approxHessian(self,f):
        """Gives an approxmation of the Hessian
        """
        return


class OptimizeBFGS(OptimizeBase): #Erik and Victor claim this

    def _step(self,f):
        if not hasattr(self, '_currentHessInv'):
            self._currentHessInv = np.eye(f._numArgs)
            self._i = 0
        self._i = self._i + 1
        p = -self._currentHessInv.dot(f.evalGrad(self.currentValues))
        alpha = self.inexactLineSearch(f,self.currentValues,p)
        s = alpha*p
        nextValues = self.currentValues + s
        y = f.evalGrad(nextValues) - f.evalGrad(self.currentValues)
        nextHessInv = self._currentHessInv + (s.dot(y) +
            y.dot(self._currentHessInv).dot(y))*(np.outer(s,s))/(s.dot(y))**2 \
                    - (self._currentHessInv.dot(np.outer(y,s)) + \
                        np.outer(s,y).dot(self._currentHessInv))/(s.dot(y))
        self._currentHessInv = nextHessInv
        return nextValues
        
    def _approxHessian(self,f):
        """Gives an approxmation of the Hessian
        """
        return
        
