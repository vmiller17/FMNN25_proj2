import sys
sys.path = sys.path + ['../']
from nose.tools import raises,with_setup
import numpy as np
import Problem_class

class TestNewtonInit:

    @raises(TypeError)
    def testTolNotFloat(self):
        self.testFunction = Newton.Newton(1,1,np.array([1,1]))

    @raises(TypeError)
    def testMaxIterationsNotInt(self):
        self.testFunction = Newton.Newton(1.,1.2,np.array([1,1]))

    @raises(TypeError)
    def testCurrentValuesNotArray(self):
        self.testFunction = Newton.Newton(1.,1.2,[1,1])
	
    @raises(TypeError)
    def testCorrectDimensions(self):
		def gWrong(self,x,y,z)
			return np.array([2*x,2*y,1])
        self.testFunction = Function.Function(f,gWrong)


class TestNewtonCall:

    def setUp(self):
		def f(self,x,y)
			return (x**2)+(y**2)
        self.testFunction = Function.Function(f)

    def tearDown(self):
        del self.testFunction

    def testCorrectValue(self):
        j = self.testFunction.evalGrad(np.array([1.,1.]))
        assert np.allclose(j,np.array([2.,2.]))

	@raises(TypeError)
	def testDimensions(self):
		self.testFunction.evalGrad(np.array([1.,2.,3.]))

	@raises(TypeError)
	def testArrayType(self):
		self.testFunction.evalgrad(np.array([1,2]))


class TestSolveEquations:

    def setUp(self):
        self.optimizer=OptimizeNewton.OptimizeNewton()
        self.b=np.array([1., 1., 1.])

    def tearDown(self):
        del self.optimizer

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
