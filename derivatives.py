import numpy as np
import numba as nb

@nb.jit(nopython=True)
def f(x, y, p):
    der = np.zeros(3)
    der[0] = 1
    der[1] = x
    der[2] = 1/(x+1)
    
    return der