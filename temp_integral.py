import numpy as np
import numba as nb

@nb.jit(nopython=True)
def F_temp(p1, p2, p3):
    return np.array([1,0,.5,-1])

@nb.jit(nopython=True)
def create_p(p1, delta_p, e_max, B):
    num = int((B*e_max - p1)/delta_p + 1)
    return np.linspace(p1, B*e_max, num)

@nb.jit(nopython=True)
def J_1(p1, p2, p3):
    return 16/15 * p3**3 * (10*(p1+p2)**2 - 15*(p1+p2)*p3 + 6*p3**2)

@nb.jit(nopython=True)
def A_4(p1, p2, delta_p):
    x = np.linspace(0,p1, int(p1/delta_p+1))
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F_temp(p1, p2, x[i])* J_1(p1, p2, x[i])
    A = np.zeros(4)
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I4(p1, p, e_max, B):
    delta_p = p[1] - p[0]
    x = create_p(p1, delta_p, e_max, B)
    y = np.zeros((len(x), 4))
    output = np.zeros(4)

    for i in range(len(x)):
        y[i,:] = A_4(p1,x[i],delta_p)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output