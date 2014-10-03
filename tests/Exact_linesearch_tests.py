import sys
sys.path = sys.path + ['../src']
from nose.tools import raises
import numpy as np
import Optimize


class TestTypeChecks:
    """These tests checks that the right exceptions are raised when the input
    is wrong."""
    def setUp(self):
        def f(x,y):
            return (x**2)+(y**2)
        self.f = Optimize.Function(f)
        self.x = np.array([1.0,1.0])
    def tearDown(self):
        del self.f

    @raises(TypeError)
    def testFNotFunction(self):
        """Should raise a TypeError because f is not of type Function"""
        def f(x):
            return x
        Optimize.OptimizeBase.exactLineSearch(f,self.x,np.array([1.0]))

    @raises(TypeError)
    def testSNotArray(self):
        """Should raise a TypeError because S is not a numpy array"""
        Optimize.OptimizeBase.exactLineSearch(self.f,self.x,[1.0,2.0])

    @raises(TypeError)
    def testSNotFloats(self):
        """Should raise a TypeError because S does not contain floats"""
        Optimize.OptimizeBase.exactLineSearch(self.f,self.x,np.array([1, 2],
            dtype=int))

    @raises(TypeError)
    def testSWrongSize(self):
        """No TypeError when size of S and arguments of f differs"""
        Optimize.OptimizeBase.exactLineSearch(self.f,self.x,np.array([1,2,3],
            dtype=float))

    @raises(TypeError)
    def testSWrongDimension(self):
        """Should raise a TypeError because S is not one dimensional"""
        Optimize.OptimizeBase.exactLineSearch(self.f,self.x,np.array([[1,2]],
            dtype=float))

    @raises(TypeError)
    def testXNotFloats(self):
        """Should raise a TypeError because x does not contain floats"""
        Optimize.OptimizeBase.exactLineSearch(self.f,np.array([1,2],dtype=int
            ),np.array([[1,2]], dtype=float))

    @raises(TypeError)
    def testXWrongDimension(self):
        """Should raise a TypeError because x is not one dimensional"""
        Optimize.OptimizeBase.exactLineSearch(self.f,np.array([[1.0,2.0]],
            ),np.array([[1,2]], dtype=float))

    @raises(TypeError)
    def testXWrongSize(self):
        """No TypeError when size of x and arguments of f differs"""
        Optimize.OptimizeBase.exactLineSearch(self.f,np.array([1.0],
            ),np.array([[1,2]], dtype=float))

class TestRunningExactLS:
    """These tests evaluates different running scenarios using exact line
    search."""
    def setUp(self):
            def f(x,y):
                return (x**2)+(y**2)
            self.f = Optimize.Function(f)
    def tearDown(self):
        del self.f
        
    def testReturnsAlphaEqualsZeroWhenMinimumIsReached(self):
        assert  np.allclose(Optimize.OptimizeBase.exactLineSearch(self.f, np.array([0.,0.]), np.array([2.,2.])), 0.)
        
    def testFindsMinimumOfOneDFunc(self):
        def g(x):
            return (x-2)**2
        h = Optimize.Function(g)
        assert np.allclose(Optimize.OptimizeBase.exactLineSearch(h, np.array([0.]), np.array([1.])), 2.)