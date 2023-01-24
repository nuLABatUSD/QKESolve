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



#@nb.jit(nopython=True)
#def f(x, y, p):
    #derivatives = np.zeros(4)
    #derivatives[0] = y[1]
    #derivatives[1] = y[2]
    #derivatives[2] = y[3]
    #derivatives[3] = p**4*y[0]
    
    #return derivatives


#@nb.jit(nopython=True)
#def f(x, y, p):
    #derivatives = np.zeros(4)
    #derivatives[0] = 0
    #derivatives[1] = p[-1]/(2*p[-3])*(np.cos(2*p[-2])*y[2])
    #derivatives[2] = p[-1]/(2*p[-3])*((-np.cos(2*p[-2])*y[1]) - (np.sin(2*p[-2])*y[3]))
    #derivatives[3] = p[-1]/(2*p[-3])*(np.sin(2*p[-2])*y[2])
                                           
                                           
    #return derivatives


#@nb.jit(nopython=True)
#def f(x, y, p):
    #der= np.zeros(4)
    #der[:]= vacuum(y, p[0], p[-1], p[-2])
    #return der  

    
@nb.jit(nopython=True)                                           
def vacuum(y, E, dm2, th):
    der= np.zeros(4)
    der[0] = 0
    der[1] = dm2/(2*E)*(np.cos(2*th)*y[2])
    der[2] = dm2/(2*E)*((-np.cos(2*th)*y[1]) - (np.sin(2*th)*y[3]))
    der[3] = dm2/(2*E)*(np.sin(2*th)*y[2])
    return der



@nb.jit(nopython=True)
def f(x,y,p):     
    ym= matrix_maker(y)
    
    derm= np.zeros(ym.shape)
    for i in range(derm.shape[0]):
        derm[i,:]= vacuum(ym[i,:], p[0], p[-1], p[-2])
    
    return array_maker(derm)



@nb.jit(nopython=True)                                         
def matrix_maker(y):
    length= len(y)
    matrix = np.zeros((length//4,4))
    
    for i in range(matrix.shape[0]):
        for j in range(4):
            matrix[i,j]= y[4*i+j]
    return matrix



@nb.jit(nopython=True)
def array_maker(M):
    length= M.shape[0]
    array = np.zeros(length*4)
    
    for i in range(M.shape[0]):
        for j in range(4):
            array[4*i+j] = M[i,j]
    return array  
                                         
                                           