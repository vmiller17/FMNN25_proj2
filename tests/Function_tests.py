import sys
sys.path = sys.path + ['../']
from nose.tools import raises,with_setup
import numpy as np
import Problem_class

class TestFunctionInit:

	def f(self,x,y)
		return (x**2)+(y**2)

	def g(self,x,y)
		return np.array([2*x,2*y])

    @raises(TypeError)
    def testFNotFunction(self):
        self.testFunction = Function.Function(3.)
	
    @raises(TypeError)
    def testGNotFunction(self):
        self.testFunction = Function.Function(f,2.)

    @raises(TypeError)
    def testCorrectDimensions(self):
		def gWrong(self,x,y,z)
			return np.array([2*x,2*y,1])
        self.testFunction = Function.Function(f,gWrong)

class TestFunctionCall:

    def setUp(self):
		def f(self,x,y)
			return (x**2)+(y**2)
		def g(self,x,y)
			return np.array([2*x,2*y])
        self.testFunction = Function.Function(f,g)

    def tearDown(self):
        del self.testFunction

    @raises(TypeError)
    def testNotFloat(self):
        self.testFunction(np.array([1,2]))

    @raises(TypeError)
    def testWrongDimension(self):
        self.testFunction(np.array([1.,2.,3.]))

    @raises(TypeError)
    def testWrongDimension2(self):
        self.testFunction(np.array([1.,2.],[1.,2.]))

    @raises(TypeError)
    def testNotArray(self):
        self.testFunction([1.,2.])

    def testCorrectValue(self):
        j = self.testFunction(np.array([0.,0.]))
        assert j == 0
        j = self.testFunction(np.array([1.,2.]))
        assert j == 5

class TestFunctionEvalGrad:

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
		self.testFunction.evalGrad(np.array([1,2]))

	def testReturnArrayType(self):
		assert isinstance(self.testFunction.evalGrad(np.array([1.,2.])),np.ndarray)










