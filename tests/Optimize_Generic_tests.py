import sys
sys.path = sys.path + ['../']
from nose.tools import raises,with_setup
import numpy as np
import Function
import Newton

class TestNewtonInit:

    @raises(TypeError)
    def testTolNotFloat(self):
        self.testFunction = Newton.Newton(1,1)

    @raises(TypeError)
    def testMaxIterationsNotInt(self):
        self.testFunction = Newton.Newton(1.,1.2)
	

class TestNewtonCall:

    def setUp(self):
		def f(self,x,y)
			return (x**2)+(y**2)
        self.testFunction = Function.Function(f)
		self.testOptimize = Newton.Newton()

    def tearDown(self):
        del self.testFunction
		del self.testOptimize

	@raise(TypeError)
    def testReturnArray(self):
		val = self.testOptimize(self.testFunction,np.array([1.,1.]))
		assert isinstance(val,np.ndarray)
		assert val.shape == (2,)

    def testCorrectValue(self):
		val = self.testOptimize(self.testFunction,np.array([1.,1.]))
		assert (val == np.array([0.,0.])).all()

    def testArrayContainsFloat(self):
		val = self.testOptimize(self.testFunction,np.array([1.,1.]))
		assert issubclass(val[0],float)

	@raises(TypeError)
	def testDimensions(self):
		self.testOptimize(self.testFunction,np.array([1.,2.,3.]))

	@raises(TypeError)
	def testArrayWithInt(self):
		self.testOptimize(self.testFunction,np.array([1,2]))

	@raises(TypeError)
	def testNotFunction(self):
		self.testOptimize(1.,np.array([1,2]))


class TestNewtonStep:

    def setUp(self):
		def f(self,x,y)
			return (x**2)+(y**2)
        self.testFunction = Function.Function(f)
		self.testOptimize = Newton.Newton()

    def tearDown(self):
        del self.testFunction
		del self.testOptimize

	def testReturnArray(self):
		val = self.testOptimize._step(self.testFunction)
		assert isinstance(val,np.ndarray)
		assert val.shape == (2,)

	@raises(TypeError)
	def testArrayType(self):
		self.testOptimize._step(1.)
