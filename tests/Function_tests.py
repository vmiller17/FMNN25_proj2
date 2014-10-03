import sys
sys.path = sys.path + ['../src']
from nose.tools import raises
import numpy as np
import Optimize


class TestFunctionInit:
    
    def setUp(self):
        def f(x,y):
            return (x**2)+(y**2)
        self.f=f


    def f(self,x,y):
	return (x**2)+(y**2)

    def g(self,x,y):
	return np.array([2*x,2*y])


    @raises(TypeError)
    def testFNotFunction(self):
        self.testFunction = Optimize.Function(3.)
	
    @raises(TypeError)
    def testGNotFunction(self):
        self.testFunction = Optimize.Function(self.f,2.)

    @raises(TypeError)
    def testCorrectDimensions(self):
        def gWrong(self,x,y,z):
            return np.array([2*x,2*y,1])
        self.testFunction = Optimize.Function(self.f,gWrong)#This could be wrong, got merge conflicts. Trying to solve //Victor


class TestFunctionCall:

    def setUp(self):
        def f(x,y):
            return (x**2)+(y**2)
        def g(x,y):
            return np.array([2*x,2*y])
        self.testFunction = Optimize.Function(f,g)


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
        def f(x,y):
            return (x**2)+(y**2)
        self.testFunction = Optimize.Function(f)


    def tearDown(self):
        del self.testFunction

    def testCorrectValue(self):
        j = self.testFunction.evalGrad(np.array([1.,1.]))
        assert np.allclose(j,np.array([2.,2.]))
        
    def testCorrectValue2(self):
        def f2(x,y,z):
            return (x**2)+(y**2)+(z**2)
        self.testFunction = Optimize.Function(f2)
        j = self.testFunction.evalGrad(np.array([3.,10.,7.]))
        assert np.allclose(j,np.array([6.,20.,14.]))
        
    def testCorrectValue3(self):
        def f3(x):
            return (x**2)
        self.testFunction = Optimize.Function(f3)
        j = self.testFunction.evalGrad(np.array([4.]))
        assert np.allclose(j,np.array([8.]))
 
    def testCorrectValue4(self):
        def f4(x,y,z,r):
            return (x**2)+(y**2)+(z**2)+(r**2)
        self.testFunction = Optimize.Function(f4)
        j = self.testFunction.evalGrad(np.array([3.,10.,7.,1.]))
        assert np.allclose(j,np.array([6.,20.,14.,2]))
        
    @raises(TypeError)
    def testDimensions(self):
        self.testFunction.evalGrad(np.array([1.,2.,3.]))

    @raises(TypeError)
    def testArrayWithInt(self):
        self.testFunction.evalGrad(np.array([1,2]))

    def testReturnArrayType(self):
	assert isinstance(self.testFunction.evalGrad(np.array([1.,2.])),np.ndarray)