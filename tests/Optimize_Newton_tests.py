import sys
sys.path = sys.path + ['../']
from nose.tools import raises,with_setup
import numpy as np
import Problem_class

class TestSolveEquations:

    def setUp(self):
        self.optimizer=OptimizeNewton.OptimizeNewton()
        self.b=np.array([1., 1., 1.])

    def tearDown(self):
        del self.optimizer
        del self.b

    @raises(TypeError)
    def testBNotArray(self):
        A=np.array([[1.,2.],[2.,1.]])
        self.optimizer._solveEquations(a,2.)

    @raises(TypeError)
    def testANotArray(self):
        A=3.5
        self.optimizer._solveEquations(A,b)

    @raises(TypeError)
    def testCorrectDimensions(self):
        A=np.array([[1.,2.],[1.,2.]])

    @raises(ValueError)
    def testNotPositiveDefinite(self):
        A=np.array([[1.,2.,5.],[0.,-3.,6],[0.,0.,4]])
        c=self.optimizer._solveEquations(A,self.b)

    def testReturnsArray(self):
        A=np.eye(3)
        c=self.optimizer._solveEquations(A,self.b)
        assert issinstance(c, np.array)

    def testCorrectAnswer(self):
        A=np.array([[1.,2.,5.],[0.,3.,6.],[0.,0.,4.]])
        c=self.optimizer._solveEquations(A,self.b)
        assert np.allclose(c,np.array([1./12., -1./6., 1./4.]))

class TestApproximateHessian:

    def setUp(self):
        def f(x,y):
            return x**2 + 3*x*y + y**2

        def g(x,y):
            return np.array([2*x + 3*y, 3*x + 2*y])

        self.function=Function.Function(f,g)
        self.optimizer=OptimizeNewton.OptimizeNewton()

    def tearDown(self):
        del self.function
        del self.optimizer

    @raises(TypeError)
    def testInputCheck(self):
        self.optimizer._approxHessian(48.)

    
