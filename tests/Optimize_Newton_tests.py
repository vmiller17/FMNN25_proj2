import sys
sys.path = sys.path + ['../src']
from nose.tools import raises,with_setup
import numpy as np
import Optimize

class TestSolveEquations:

    def setUp(self):
        self.optimizer=Optimize.OptimizeNewton()
        self.b=np.array([1., 1., 1.])

    def tearDown(self):
        del self.optimizer
        del self.b

    @raises(TypeError)
    def testBNotArray(self):
        A=np.array([[1.,2.],[2.,1.]])
        self.optimizer._solveEquations(A,2.)

    @raises(TypeError)
    def testANotArray(self):
        A=3.5
        self.optimizer._solveEquations(A,self.b)

    @raises(ValueError)
    def testCorrectDimensions(self):
        A=np.array([[1.,2.],[1.,2.]])
        self.optimizer._solveEquations(A,self.b)

    @raises(TypeError)
    def testAFloatArray(self):
        A=np.array([[1,2],[2,3]])
        self.optimizer._solveEquations(A,self.b)

    @raises(TypeError)
    def testBFloatArray(self):
        A=np.array([[1.,2.],[1.,2.]])
        c=np.array([1,3])
        self.optimizer._solveEquations(A, c)

    @raises(ValueError)
    def testNotPositiveDefinite(self):
        A=np.array([[1.,2.,5.],[0.,-3.,6],[0.,0.,4]])
        c=self.optimizer._solveEquations(A,self.b)

    def testReturnsArray(self):
        A=np.eye(3)
        c=self.optimizer._solveEquations(A,self.b)
        assert isinstance(c, np.ndarray)

    def testCorrectAnswer(self):
        A=np.array([[1.,2.,5.],[0.,3.,6.],[0.,0.,4.]])
        c=self.optimizer._solveEquations(A,self.b)
        assert np.allclose(c,np.array([1./12., -1./6., 1./4.]))

class TestApproximateHessian:
    
    """
    This test class needs to be overlooked. The gradient is sometimes sent as a function
    and sometimes as a ndarray. 
    """

    def setUp(self):
        def f(x,y):
            return x**2 + y**2

        def g(x,y):
            return np.array([2*x,2*y])

        self.function=Optimize.Function(f,g)
        self.optimizer=Optimize.OptimizeNewton()

    def tearDown(self):
        del self.function
        del self.optimizer

    @raises(TypeError)
    def testInputCheck(self):
        self.optimizer._approxHessian(48.)

    def testHessianContainsFloats(self):
       c = self.optimizer._approxHessian(self.function)
       assert issubclass(c[0,0], float)
	
    def testShapeHessian(self):
        assert (self.optimizer._approxHessian(self.function)).shape == (2,2)

    def testApproxHessian(self): #Okand tolerans, kan behova justeras.
        H = np.array([[2.,0.],[0.,2.]])
        approx=self.optimizer._approxHessian(self.function) #the gradient must be provided as a ndarray. Not a function.
        assert np.allclose(H,approx) # To do this we need to choose a point where we want to evaluate the hessian.

    @raises(ValueError)
    def testNotPositiveDefinite(self):
        def f(x,y):
            return x**2 + 3*x*y + y**2

        self.function1=Optimize.Function(f)
        self.optimizer1=Optimize.OptimizeNewton()
        self.optimizer1._approxHessian(f) # Den funktionen ska kolla om f ger upphov till en s.p.d. hessian
		








    
