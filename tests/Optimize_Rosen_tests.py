import sys
sys.path = sys.path + ['../src']
import numpy as np
import os
sys.path.append(os.path.abspath("C:\Users\Labinot\FMNN25_proj2\src"))
import Optimize


def rosen(x,y):
    return 100*(y-x**2)**2 + (1-x)**2
    
def testRosenBFGS():

    f = Optimize.Function(rosen)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-8)
    startValues = np.array([1.,1.])
    solution = optimizer(f,startValues)

    assert abs(solution[0]) - 1. < 1e-6
    assert abs(solution[1]) - 1. < 1e-6
    
