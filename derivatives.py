import numpy as np
import numba as nb

#@nb.jit(nopython=True)
#def f(x, y, p):
    #der = np.zeros(3)
    #der[0] = 1
    #der[1] = x
    #der[2] = 1/(x+1)
    
    #return der
    
    
#@nb.jit(nopython=True)
#def f(x, y, p):
    #derivatives = np.zeros(2)
    #derivatives[0] = np.exp(-x)
    #derivatives[1] = p*x
    
    #return derivatives



@nb.jit(nopython=True)
def (x, y, p):
    derivatives = np.zeros(4)
    derivatives[0] = y[1]
    derivatives[1] = y[2]
    derivatives[2] = y[3]
    derivatives[3] = p**4*y[0]
    
    return derivatives