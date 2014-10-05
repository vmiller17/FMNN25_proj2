import sys
sys.path = sys.path + ['../src']
from nose.tools import raises
import numpy as np
import Optimize
from Optimize import OptimizeBase
import math

class TestRunningOptimizeBase:
    """These tests evaluates different running scenarios using inexact line
    search."""

    def setUp(self):
        def f(x,y):
            return 3*y**2 + x**4 - 3 + math.exp((x*y)**2)
        def g(x,y):
            return np.array([4*x**3 + 2*x*(y**2)*math.exp((x*y)**2),
                6*y + 2*y*(x**2)*math.exp((x*y)**2)],dtype=float)

        class NewtonOptimizer(OptimizeBase):
            def _step(self,f):
                S = -np.linalg.inv(self.H(*self.currentValues)).dot(self.currentGrad)
                self.currentValues = self.currentValues + S
                return self.currentValues 

            def H(elf,x,y):
                return np.array([
                    [12*x**2+4*(x**2)*(y**4)*math.exp((x*y)**2),
                        4*(x**3)*(y**3)*math.exp((x*y)**2)],
                    [4*(y**3)*(x**3)*math.exp((x*y)**2),
                        6 + 4*(x**4)*(y**2)*math.exp((x*y)**2)]
                    ],dtype=float)
                    
            def _approxHessian(self,f): #Added this just because we have to @abstractmethod
                """Gives an approxmation of the Hessian
                """
                return

        self.f = Optimize.Function(f,g)
        self.Opt = NewtonOptimizer(tol=1e-18)

    def tearDown(self):
        del self.f
        del self.Opt

    def testCorrectAnswer(self):
        """Did not return correct answer within 100 iterations"""
        startValues = np.array([2,2],dtype=float)

        solution = self.Opt(self.f,startValues)

        assert abs(solution[0]) < 1e-6
        assert abs(solution[1]) < 1e-6
