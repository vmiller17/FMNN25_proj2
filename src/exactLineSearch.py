from scipy.optimize import minimize_scalar
def exactLineSearch(self, f, xk, sk):
    """
    :param Function f: An object of the Function class. F is called with numpy array of shape (m,).
    :param array xk: A numpy array of shape (m,), containing floats. xk = _currentValues in OptimizeBase.
    :param array sk: A numpy array of shape (m,), containing floats. sk is the newton direction.
    :returns alpha s.t. fi(alpha)=f(xk+alpha*sk) is minimized.
    """
    def fi(alpha):
        return f(xk+alpha*sk)
    return minimize_scalar(fi).x