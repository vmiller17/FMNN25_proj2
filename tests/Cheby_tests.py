import sys
sys.path = sys.path + ['../src']
import numpy as np
import Optimize
from  scipy       import dot
import scipy.optimize as so

## Chebyquad function.
def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree 
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2. * x * T(x, n - 1) - T(x, n - 2)

def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree 
    Note d/dx T(x,n)= n*U(x,n-1)  
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2. * x
    return 2. * x * U(x, n - 1) - U(x, n - 2) 
    
def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """    
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in xrange(n):
            if i % 2 == 0: 
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)
    
    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2. * xj - 1., i) for xj in x) / n
    return np.array([approx_integral(i) - exint.next() for i in xrange(n)]) 

## Cheby for n = 4.
def chebyquadn4(x1,x2,x3,x4):
    """            
    norm(chebyquad_fcn)**2                
    """
    x = np.array([x1,x2,x3,x4])
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquadn4(x1,x2,x3,x4):
    """
    Evaluation of the gradient function of chebyquad
    """
    x = np.array([x1,x2,x3,x4])
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * np.array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))

## Cheby for n = 8.
def chebyquadn8(x1,x2,x3,x4,x5,x6,x7,x8):
    """            
    norm(chebyquad_fcn)**2                
    """
    x = np.array([x1,x2,x3,x4,x5,x6,x7,x8])
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquadn8(x1,x2,x3,x4,x5,x6,x7,x8):
    """
    Evaluation of the gradient function of chebyquad
    """
    x = np.array([x1,x2,x3,x4,x5,x6,x7,x8])
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * np.array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
## Cheby for n = 11.
def chebyquadn11(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    """            
    norm(chebyquad_fcn)**2                
    """
    x = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquadn11(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    """
    Evaluation of the gradient function of chebyquad
    """
    x = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * np.array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
## Claus cheby.
def chebyquad(x):
    """            
    norm(chebyquad_fcn)**2                
    """
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquad(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * np.array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
## Initial guesses and scipy approximations.
x0n4=np.linspace(0,1,4)
x0n8=np.linspace(0,1,8)
x0n11=np.linspace(0,1,11)
xmin4 = so.fmin_bfgs(chebyquad,x0n4,gradchebyquad, gtol=1e-05)
xmin8 = so.fmin_bfgs(chebyquad,x0n8,gradchebyquad, gtol=1e-05)
xmin11 = so.fmin_bfgs(chebyquad,x0n11,gradchebyquad, gtol=1e-05)


# Tests for BFGS, n=4,8,11.
def testChebyBFGSn4():
    f = Optimize.Function(chebyquadn4, gradchebyquadn4)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-8)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

    
def testChebyBFGSn8():
    f = Optimize.Function(chebyquadn8, gradchebyquadn8)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-8)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBFGSn11():
    f = Optimize.Function(chebyquadn11, gradchebyquadn11)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-6)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))     
  
# Tests for BroydenGood, n=4,8,11.
def testChebyBroydenGoodn4():
    f = Optimize.Function(chebyquadn4, gradchebyquadn4)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-8)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyBroydenGoodn8():
    f = Optimize.Function(chebyquadn8, gradchebyquadn8)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-8)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBroydenGoodn11():
    f = Optimize.Function(chebyquadn11, gradchebyquadn11)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-6)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))    

# Tests for BroydenBad, n=4,8,11.
def testChebyBroydenBadn4():
    f = Optimize.Function(chebyquadn4, gradchebyquadn4)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-4)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyBroydenBadn8():
    f = Optimize.Function(chebyquadn8, gradchebyquadn8)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-2)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBroydenBadn11():
    f = Optimize.Function(chebyquadn11, gradchebyquadn11)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-2)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))  
    
# Tests for Newton, n=4,8,11.
def testChebyNewtonn4():
    f = Optimize.Function(chebyquadn4, gradchebyquadn4)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyNewtonn8():
    f = Optimize.Function(chebyquadn8, gradchebyquadn8)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyNewtonn11():
    f = Optimize.Function(chebyquadn11, gradchebyquadn11)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))
    
# Tests for DFP, n=4,8,11.
def testChebyDFPn4():
    f = Optimize.Function(chebyquadn4, gradchebyquadn4)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyDFPn8():
    f = Optimize.Function(chebyquadn8, gradchebyquadn8)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyDFPn11():
    f = Optimize.Function(chebyquadn11, gradchebyquadn11)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))
    
#### Same as above but without the gradient.
# Tests for BFGS, n=4,8,11.
def testChebyBFGSn4NoG():
    f = Optimize.Function(chebyquadn4)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-8)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

    
def testChebyBFGSn8NoG():
    f = Optimize.Function(chebyquadn8)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-8)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBFGSn11NoG():
    f = Optimize.Function(chebyquadn11)
    optimizer  = Optimize.OptimizeBFGS(tol=1e-6)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))     
  
# Tests for BroydenGood, n=4,8,11.
def testChebyBroydenGoodn4NoG():
    f = Optimize.Function(chebyquadn4)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-8)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyBroydenGoodn8NoG():
    f = Optimize.Function(chebyquadn8)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-8)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBroydenGoodn11NoG():
    f = Optimize.Function(chebyquadn11)
    optimizer  = Optimize.OptimizeBroydenGood(tol=1e-6)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))    

# Tests for BroydenBad, n=4,8,11.
def testChebyBroydenBadn4NoG():
    f = Optimize.Function(chebyquadn4)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-4)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyBroydenBadn8NoG():
    f = Optimize.Function(chebyquadn8)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-2)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyBroydenBadn11NoG():
    f = Optimize.Function(chebyquadn11)
    optimizer  = Optimize.OptimizeBroydenBad(tol=1e-2)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))  
    
# Tests for Newton, n=4,8,11.
def testChebyNewtonn4NoG():
    f = Optimize.Function(chebyquadn4)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyNewtonn8NoG():
    f = Optimize.Function(chebyquadn8)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyNewtonn11NoG():
    f = Optimize.Function(chebyquadn11)
    optimizer  = Optimize.OptimizeNewton(tol=1e-5)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))
    
# Tests for DFP, n=4,8,11.
def testChebyDFPn4NoG():
    f = Optimize.Function(chebyquadn4)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln4 = optimizer(f, x0n4)
    assert np.allclose(np.sort(soln4), np.sort(xmin4))

def testChebyDFPn8NoG():
    f = Optimize.Function(chebyquadn8)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln8 = optimizer(f, x0n8)
    assert np.allclose(np.sort(soln8), np.sort(xmin8))
    
def testChebyDFPn11NoG():
    f = Optimize.Function(chebyquadn11)
    optimizer  = Optimize.OptimizeDFP(tol=1e-5)
    soln11 = optimizer(f, x0n11)
    assert np.allclose(np.sort(soln11), np.sort(xmin11))  