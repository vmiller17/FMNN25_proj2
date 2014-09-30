import sys
sys.path = sys.path + ['../src']
from nose.tools import raises
import numpy as np
import Optimize
import math


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
        Optimize.OptimizeBase.inexactLineSearch(f,self.x,np.array([1.0]))

    @raises(TypeError)
    def testSNotArray(self):
        """Should raise a TypeError because S is not a numpy array"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,self.x,[1.0,2.0])

    @raises(TypeError)
    def testSNotFloats(self):
        """Should raise a TypeError because S does not contain floats"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,self.x,np.array([1, 2],
            dtype=int))

    @raises(TypeError)
    def testSWrongSize(self):
        """No TypeError when size of S and arguments of f differs"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,self.x,np.array([1,2,3],
            dtype=float))

    @raises(TypeError)
    def testSWrongDimension(self):
        """Should raise a TypeError because S is not one dimensional"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,self.x,np.array([[1,2]],
            dtype=float))

    @raises(TypeError)
    def testXNotFloats(self):
        """Should raise a TypeError because x does not contain floats"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,np.array([1,2],dtype=int
            ),np.array([[1,2]], dtype=float))

    @raises(TypeError)
    def testXWrongDimension(self):
        """Should raise a TypeError because x is not one dimensional"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,np.array([[1.0,2.0]],
            ),np.array([[1,2]], dtype=float))

    @raises(TypeError)
    def testXWrongSize(self):
        """No TypeError when size of x and arguments of f differs"""
        Optimize.OptimizeBase.inexactLineSearch(self.f,np.array([1.0],
            ),np.array([[1,2]], dtype=float))

class TestRunningInexactLS:
    """These tests evaluates different running scenarios using inexact line
    search."""

    def setUp(self):
        def f(x,y):
            return 3*y**2 + x**4 - 3 + math.exp((x*y)**2)
        def g(x,y):
            return np.array([4*x**3 + 2*x*(y**2)*math.exp((x*y)**2),
                6*y + 2*y*(x**2)*math.exp((x*y)**2)],dtype=float)
        def H(x,y):
            return np.array([
                [12*x**2+4*(x**2)*(y**4)*math.exp((x*y)**2),
                    4*(x**3)*(y**3)*math.exp((x*y)**2)],
                [4*(y**3)*(x**3)*math.exp((x*y)**2),
                    6 + 4*(x**4)*(y**2)*math.exp((x*y)**2)]
                ],dtype=float)

        self.f = Optimize.Function(f,g)
        self.H = H

    def tearDown(self):
        del self.f
        del self.H

    def testCorrectAnswer(self):
        """Did not return correct answer within 100 iterations"""
        x = np.array([2,2],dtype=float)

        for i in range(100):
            g = self.f.evalGrad(x)
            S = -np.linalg.inv(self.H(*x)).dot(g)
            alpha = Optimize.OptimizeBase.inexactLineSearch(self.f,x,S)
            x = x + alpha*S

        assert np.isclose(self.f(x),-2)

    def testBetterThanNoSearch(self):
        """The inexact line search used more iterations than no line serach"""
        i=0
        x = np.array([2,2],dtype=float)
        while(not np.isclose(self.f(x),-2)):
            g = self.f.evalGrad(x)
            S = -np.linalg.inv(self.H(*x)).dot(g)
            x = x + S
            i = i + 1

        j=0
        x = np.array([2,2],dtype=float)
        while(not np.isclose(self.f(x),-2)):
            g = self.f.evalGrad(x)
            S = - np.linalg.inv(self.H(*x)).dot(g)
            alpha = Optimize.OptimizeBase.inexactLineSearch(self.f,x,S)
            x = x + alpha*S
            j = j + 1
            if (j>i): break

        assert j<i
