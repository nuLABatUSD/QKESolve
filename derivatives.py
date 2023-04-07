import numpy as np
import numba as nb
import numpy.polynomial.laguerre as geek

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

    
    
Gf= 1.1663787*10**-5*(1/(1000**2))
x, w= geek.laggauss(40)
    
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
    if (p[-5] != -1):
        T= p[-3]
        cT= p[-4]
        ym= matrix_maker(y)
        N= ym.shape[0]
        energy= p[:N]*T
        derm= np.zeros(ym.shape)
        Vvv= Vvv_function(ym, energy)
        VT= VT_function(ym, energy, T)
        if (p[-6] == -1):
            if (p[-7] == -1):
                C = scattering(ym, p, p[-8], p[-9])
            else:
                C= scattering(ym, p)
            Cs= C_scat(C, ym)
        else:
            C= np.zeros((N,4))
            Cs = np.zeros((N,4))
        for i in range(derm.shape[0]):
               derm[i,:]= vacuum(ym[i,:], energy[i], p[-1], p[-2]) + cross_product(ym[i,:], Vvv) + cT*cross_product(ym[i,:], energy[i]*VT) + (1/ym[i, 0])*C[i,:] - Cs[i, :]*(1/ym[i,0])
    
        return array_maker(derm)
    
    else:
        T= p[-3]
        cT= p[-4]
        ym, ym_bar= newmatrix_maker(y)
        N= ym.shape[0]
        energy= p[:N]*T
        derm= np.zeros(ym.shape)
        derm_bar= np.zeros(ym_bar.shape)
        Vvv_Vvvbar= Vvv_function(ym, energy) - Vvv_function(ym_bar, energy)
        VT= VT_barfunction(ym, ym_bar, energy, T)
        if (p[-6] == -1):
            if p[-7] == -1:
                C = scattering(ym, p, p[-8], p[-9])
                D = scattering(ym_bar, p, -p[-8], -p[-9])
            else:
                C= scattering(ym, p)
                D= scattering(ym_bar, p)

            Cs= C_scat(C, ym)
            Cs_bar= C_scat(D, ym_bar)
        else:
            C= np.zeros((N,4))
            D = np.zeros((N,4))

            Cs = np.zeros((N,4))
            Cs_bar = np.zeros((N,4))
        for i in range(derm.shape[0]):
            derm[i,:]= vacuum(ym[i,:], energy[i], p[-1], p[-2]) + cross_product(ym[i,:], Vvv_Vvvbar) + cT*cross_product(ym[i,:], energy[i]*VT) + (1/ym[i, 0])*C[i,:] - Cs[i, :]*(1/ym[i,0])
            derm_bar[i,:]= -vacuum(ym_bar[i,:], energy[i], p[-1], p[-2]) + cross_product(ym_bar[i,:], Vvv_Vvvbar) - cT*cross_product(ym_bar[i,:], energy[i]*VT) + (1/ym_bar[i, 0])*D[i,:] - Cs_bar[i, :]*(1/ym_bar[i,0])
        return newarray_maker(derm, derm_bar)



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
                           
    
@nb.jit(nopython=True)    
def probability(ym0, Eval, t, y):
    resultsMatrix= threeD(y)
    array= dndE(ym0, Eval)
    initial= n_ve(resultsMatrix[0,:,:], Eval)
    later= np.zeros(len(t))
    prob_ve= np.zeros(len(t))
    
    for i in range(len(t)):
        later[i]= n_ve(resultsMatrix[i,:,:], Eval)
        prob_ve[i]= later[i]/ initial
    return prob_ve



@nb.jit(nopython=True) 
def n_ve(ym0, Eval):
    array= dndE(ym0, Eval)
    nve= np.trapz(array, Eval)
    
    return nve



@nb.jit(nopython=True)
def threeD(y):
    d= matrix_maker(y[0,:])
    matrix= np.zeros((y.shape[0], d.shape[0], 4))
                    
    for i in range(y.shape[0]):
        matrix[i,:,:]= matrix_maker(y[i,:])
    
    return matrix




@nb.jit(nopython=True)
def dndE(ym0, Eval):
    array= np.zeros(len(ym0))
    for i in range(len(array)):
        p0= ym0[i,0]
        E= Eval[i]
        pz= ym0[i, 3]
        array[i]= .5* p0*(1+pz)*(E**2/2*np.pi**2)
        
    return array

@nb.jit(nopython=True)
def Vvv_function(ym, Eval):
    v= np.zeros(3)
    yx= Eval[:]**2*ym[:,0]*ym[:,1]
    yy= Eval[:]**2*ym[:,0]*ym[:,2]
    yz= Eval[:]**2*ym[:,0]*ym[:,3]
    x_integral= np.trapz(yx , Eval)
    y_integral= np.trapz(yy , Eval)
    z_integral= np.trapz(yz , Eval)
    v[0]= ((np.sqrt(2)*Gf)/(2*np.pi**2))*x_integral
    v[1]= ((np.sqrt(2)*Gf)/(2*np.pi**2))*y_integral
    v[2]= ((np.sqrt(2)*Gf)/(2*np.pi**2))*z_integral
    return v



@nb.jit(nopython=True)
def weight(x, T):
    u= .510998/T
    g= ((np.sqrt(x**2+u**2)*x**2*np.exp(x))/(np.exp(np.sqrt(x**2+u**2))+1))
    return g

    
    
    
@nb.jit(nopython=True)   
def g(w,x, T):
    summ= 0
    for i in range(40):
        summ= summ+ w[i]*weight(x[i], T)
    return summ






@nb.jit(nopython=True)
def VT_function(ym, Eval, T):
    u= .510998/T

    pepe= (4*T**4)/ (2*np.pi**2)* g(w,x, T)
    v= np.zeros(3)
    mw= 80433
    mz= 91187.6
    constant1= (16*np.sqrt(2)*Gf)/(3*mw**2)
    constant2= (8*np.sqrt(2)*Gf)/(3*mz**2)
    yx= Eval[:]**3*ym[:,0]*ym[:,1]
    yy= Eval[:]**3*ym[:,0]*ym[:,2]
    yz= Eval[:]**3*ym[:,0]*ym[:,3]
    x_integral= np.trapz(yx , Eval)
    y_integral= np.trapz(yy , Eval)
    z_integral= np.trapz(yz , Eval)
    v[0]= constant2*x_integral
    v[1]= constant2*y_integral
    v[2]= constant2*z_integral + constant1*pepe
    
    return -v



@nb.jit(nopython=True)
def VT_barfunction(ym, ym_bar, Eval, T):
    u= .510998/T

    pepe= (4*T**4)/ (2*np.pi**2)* g(w,x, T)
    v= np.zeros(3)
    mw= 80433
    mz= 91187.6
    constant1= (16*np.sqrt(2)*Gf)/(3*mw**2)
    constant2= (8*np.sqrt(2)*Gf)/(3*mz**2)
    yx= Eval[:]**3*(ym[:,0]*ym[:,1] + ym_bar[:,0]*ym_bar[:,1])
    yy= Eval[:]**3*(ym[:,0]*ym[:,2] + ym_bar[:,0]*ym_bar[:,2])
    yz= Eval[:]**3*(ym[:,0]*ym[:,3] + ym_bar[:,0]*ym_bar[:,2])
    x_integral= np.trapz(yx , Eval)
    y_integral= np.trapz(yy , Eval)
    z_integral= np.trapz(yz , Eval)
    v[0]= constant2*x_integral
    v[1]= constant2*y_integral
    v[2]= constant2*z_integral + constant1*pepe
    
    return -v


@nb.jit(nopython=True)
def cross_product(ym, v):
    der= np.zeros(4)
    der[0]= 0
    der[1]= v[1]*ym[3]- v[2]*ym[2]
    der[2]= v[2]*ym[1]- v[0]*ym[3]
    der[3]= v[0]*ym[2]- v[1]*ym[1]
    
    return der



@nb.jit(nopython=True)
def newmatrix_maker(y):
    N= len(y)// 2
    Mv= matrix_maker(y[:N])
    Mv_= matrix_maker(y[N:])
    return Mv, Mv_




@nb.jit(nopython=True)
def newarray_maker(Mv, Mvbar):
    N= Mv.shape[0]
    array= np.zeros(8*N)
    array[:4*N]= array_maker(Mv)
    array[4*N:]= array_maker(Mvbar)
    return array


@nb.jit(nopython=True)
def feq(Energy, Temperature, eta = 0):
    return 1/(np.exp(Energy/Temperature-eta)+1)

@nb.jit(nopython=True)
def scattering(ym ,p, eta_e = 0, eta_mu = 0):
    T= p[-3]
    N= ym.shape[0]
    Eval= p[:N]*T
    C= np.zeros((N,4))
    
    fe_eq = feq(Eval, T, eta_e)
    fm_eq = feq(Eval, T, eta_mu)

    for i in range(len(C)):
        c0= (-1.27*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1+ym[i,3])- fe_eq[i]))+(-.92*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1-ym[i,3])- fm_eq[i]))
#        c0= (-1.27*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1+ym[i,3])- 1/(np.exp(Eval[i])+1)))+(-.92*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1-ym[i,3])- 1/(np.exp(Eval[i])+1)))
        cx= (-1/2)*(1/3.15)*Gf**2*T**4*Eval[i]*.54598*ym[i,0]*ym[i,1]
        cy= (-1/2)*(1/3.15)*Gf**2*T**4*Eval[i]*.54598*ym[i,0]*ym[i,2]
        cz= (-1.27*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1+ym[i,3])- fe_eq[i]))-(-.92*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1-ym[i,3])- fm_eq[i]))
#        cz= (-1.27*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1+ym[i,3])- 1/(np.exp(Eval[i])+1)))-(-.92*Gf**2*T**4*Eval[i]*(.5*ym[i,0]*(1-ym[i,3])- 1/(np.exp(Eval[i])+1)))
    
        C[i,:]= [c0*ym[i,0],cx,cy,cz]
    return C





@nb.jit(nopython=True)
def C_scat(C, ym):
    N= ym.shape[0]
    Cscat= np.zeros((N,4))
    for i in range(len(C)):
        Cscat[i,0]= 0
        Cscat[i,1]= C[i, 0]*ym[i, 1]
        Cscat[i,2]= C[i, 0]*ym[i, 2]
        Cscat[i,3]= C[i, 0]*ym[i, 3]
        
    return Cscat
