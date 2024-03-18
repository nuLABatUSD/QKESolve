import Fvvsc
import numpy as np
import numba as nb

rho_zero = np.zeros(4)

@nb.jit(nopython=True)
def den_mat(p1, y_matrix, e_max, dp):
    if p1 > e_max or p1 <= 0:
        return rho_zero
    else:
        return y_matrix[p_index(p1, dp)-1, :]

@nb.jit(nopython=True)
def F(p1, p2, p3, y_matrix, e_max, dp):
    return Fvvsc.Fvvsc_components(den_mat(p1, y_matrix, e_max, dp), den_mat(p2, y_matrix, e_max, dp), den_mat(p3, y_matrix, e_max, dp), den_mat(p1+p2-p3, y_matrix, e_max, dp))
    
    
    
div_small_integer_fix = 1e-12
@nb.jit(nopython=True)
def p_length(p1, dp):
    return int(p1/dp + 1 + div_small_integer_fix)

@nb.jit(nopython=True)
def p_index(p1, dp):
    return int(p1/dp + div_small_integer_fix)

@nb.jit(nopython=True)
def create_p(p1, e_max, B, dp):
    num = p_length((B*e_max - p1), dp)
    return np.linspace(p1, B*e_max, num)

@nb.jit(nopython=True)
def J_1(p1, p2, p3):
    return 16/15 * p3**3 * (10*(p1+p2)**2 - 15*(p1+p2)*p3 + 6*p3**2)

@nb.jit(nopython=True)
def J_2(p1,p2):
    return 16/15 * p2**3 * (10*p1**2 + 5*p1*p2 + p2**2)

@nb.jit(nopython=True)
def J_3(p1,p2,p3):
    return 16/15 * ( (p1+p2)**5 - 10*(p1+p2)**2*p3**3 + 15*(p1+p2)*p3**4 - 6*p3**5)


@nb.jit(nopython=True)
def A_1(p1, p2, y_matrix, e_max, dp):
    x = create_p(0, p2, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_1(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I1(p1, y_matrix, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_1(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output

@nb.jit(nopython=True)
def A_2(p1, p2, y_matrix, e_max, dp):
    x = create_p(p2, p1, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_2(p1, p2)
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I2(p1, y_matrix, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_2(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output

@nb.jit(nopython=True)
def A_3(p1, p2, y_matrix, e_max, dp):
    x = create_p(p1,p1+p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_3(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I3(p1, y_matrix, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_3(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output

@nb.jit(nopython=True)
def A_4(p1, p2, y_matrix, e_max, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_1(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I4(p1, y_matrix, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_4(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output

@nb.jit(nopython=True)
def A_5(p1, p2, y_matrix, e_max, dp):
    x = create_p(p1,p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_2(p2, p1)
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A

@nb.jit(nopython=True)
def I5(p1, y_matrix, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_5(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output

@nb.jit(nopython=True)
def A_6(p1, p2, y_matrix, e_max, dp):
    x = create_p(p2,p1+p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i], y_matrix, e_max, dp)* J_3(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


@nb.jit(nopython=True)
def I6(p1, y_matrix, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_6(p1, x[i], y_matrix, e_max, dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output
@nb.jit(nopython=True)
def Ivvsc(p1, y_matrix, p, e_max, B, dp):
    return I1(p1, y_matrix, p, e_max, B, dp) + I2(p1, y_matrix, p, e_max, B, dp) + I3(p1, y_matrix, p, e_max, B, dp) + I4(p1, y_matrix, p, e_max, B, dp) + I5(p1, y_matrix, p, e_max, B, dp) + I6(p1, y_matrix, p, e_max, B, dp)

GF = 1.166e-11

c_pre = GF**2 / (2*np.pi)**3

@nb.jit(nopython=True)
def C_Ivvsc(y_mat, p, e_max, B, dp):
    C = np.zeros((len(p)-1,4))
    
    for i in range(len(C)):
        C[i,:] = Ivvsc(p[i+1], y_mat, p, e_max, B, dp) / p[i+1]**2
    return C * c_pre

@nb.jit(nopython=True, parallel=True)
def C_Ivvsc_p(y_mat, p, e_max, B, dp):
    C = np.zeros((len(p)-1,4))
    
    for i in nb.prange(len(C)):
        C[i,:] = Ivvsc(p[i+1], y_mat, p, e_max, B, dp) / p[i+1]**2
    return C * c_pre
