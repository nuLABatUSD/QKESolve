#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import numba as nb
import random
import matplotlib.pyplot as plt

identity_matrix = np.zeros((2,2), dtype = 'complex128')
identity_matrix[0,0]= 1
identity_matrix[1,1] = 1

@nb.jit(nopython=True)
def random_complex_numbers():
    real_parts = np.random.randn(4)
    imaginary_parts = np.random.randn(4)
    complex_numbers = real_parts + 1j*imaginary_parts
    return complex_numbers

# In[4]:
######### KEY:
######### takes two, 4 component vectors each corresponding to a matrix
######### multiplies the "matrices" but keeps them as vector/array representations the whole time
######### outputs the components for the resulting matrix
@nb.jit(nopython=True)
def multiply_matrices(M1components, M2components): 
    M10, M1x, M1y, M1z = M1components
    M20, M2x, M2y, M2z = M2components
    m1xyz = np.array([M1x, M1y, M1z])
    m2xyz = np.array([M2x, M2y, M2z])
    m1_dot_m2 = M1x*M2x + M1y*M2y + M1z*M2z
    M30 = M10*M20 + m1_dot_m2
    #M30 = M10*M20 + np.tensordot(m1xyz, m2xyz, axes=1) -- numba didnt like this
    m3xyz = M10*m2xyz + M20*m1xyz +1j*(np.cross(m1xyz, m2xyz))
    M3components = np.array([M30, m3xyz[0], m3xyz[1], m3xyz[2]])
    return M3components

#functions to take p arrays and make general a and b, so that we can do multiplication noramllly

######### KEY:
######### this function generalizes the specific array that is (identity minus rho)...
######### it takes the 4 component vector/array that would create a matrix that is in the form (1-rho)
######### makes it into more generalized components "a0xyz"
@nb.jit(nopython=True)
def generalize_IDrho(IDrho_array):
    P0, Px, Py, Pz = IDrho_array
    pxyz = np.array([Px, Py, Pz])
    A0 = 1 - 0.5*P0
    axyz = -0.5*P0*pxyz
    a0xyz = np.array([A0, axyz[0], axyz[1], axyz[2]])
    return a0xyz

######### KEY:
######### this function generalizes the specific array that is just rho...
######### it takes the 4 component vector/array that would create a matrix that is in the form rho
######### makes it into more generalized components "b0xyz"
@nb.jit(nopython=True)
def generalize_rho(rho_array):
    P0, Px, Py, Pz = rho_array
    pxyz = np.array([Px, Py, Pz])
    B0 = 0.5*P0
    bxyz = 0.5*P0*pxyz
    b0xyz = np.array([B0, bxyz[0], bxyz[1], bxyz[2]])
    return b0xyz

# In[5]:

######### KEY:
######### this function takes four vectors: each a 4 component array to represent a matrix rho
######### it calculates the first term in Bennett et al. A.11: (1-rho1)rho3[(1-rho2)rho4 + tr(-)]
######### it outputs the resulting matrix for this term, but as its ARRAY/VECTOR representation
@nb.jit(nopython=True)
def F_components_term1(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4):
    a0xyz = generalize_IDrho(p0xyz_2)
    b0xyz = generalize_rho(p0xyz_4)
    c0xyz = multiply_matrices(a0xyz, b0xyz)
    d0xyz = np.array([c0xyz[0] + 2*c0xyz[0], c0xyz[1], c0xyz[2], c0xyz[3]])
    g0xyz = generalize_IDrho(p0xyz_1)
    h0xyz = generalize_rho(p0xyz_3)
    e0xyz = multiply_matrices(h0xyz, d0xyz)
    f0xyz = multiply_matrices(g0xyz, e0xyz)
    return (f0xyz)

######### KEY:
######### this function takes four vectors: each a 4 component array to represent a matrix rho
######### it calculates the second term in Bennett et al. A.11: rho1(1-rho3)[rho2(1-rho4) + tr(-)]
######### it outputs the resulting matrix for this term, but as its ARRAY/VECTOR representation
@nb.jit(nopython=True)
def F_components_term2(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4):
    a0xyz = generalize_IDrho(p0xyz_4)
    b0xyz = generalize_rho(p0xyz_2)
    c0xyz = multiply_matrices(b0xyz, a0xyz)
    d0xyz = np.array([c0xyz[0] + 2*c0xyz[0], c0xyz[1], c0xyz[2], c0xyz[3]])
    g0xyz = generalize_IDrho(p0xyz_3)
    h0xyz = generalize_rho(p0xyz_1)
    e0xyz = multiply_matrices(g0xyz, d0xyz)
    f0xyz = multiply_matrices(h0xyz, e0xyz)
    return (f0xyz)

# In[6]:

######### KEY:
######### this function takes four vectors: each a 4 component array to represent a matrix rho
######### it calculates the arrays for the FIRST TERM in Bennett et al. A.11: (1-rho1)rho3[(1-rho2)rho4 + tr(-)] using F_components_term1
######### then it adds the conjugate of this array, adds it to the original array, and outputs the real array
@nb.jit(nopython=True)
def Fvvsc_components_term1(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4):
    f0xyz = F_components_term1(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4)
    return 2 * np.real(f0xyz)
#### Return a real array, instead of a complex one
    f0xyz_conj = np.conjugate(f0xyz)
    Re_f0xyz = f0xyz + f0xyz_conj
    return Re_f0xyz

######### KEY:
######### this function takes four vectors: each a 4 component array to represent a matrix rho
######### it calculates the arrays for the SECOND TERM in Bennett et al. A.11: rho1(1-rho3)[rho2(1-rho4) + tr(-)] using F_components_term2
######### then it adds the harmonic conjugate of this array, adds it to the original array, and outputs the real array
@nb.jit(nopython=True)
def Fvvsc_components_term2(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4):
    f0xyz = F_components_term2(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4)
    return 2 * np.real(f0xyz)
#### Return a real array, instead of a complex one
    f0xyz_conj = np.conjugate(f0xyz)
    Re_f0xyz = f0xyz + f0xyz_conj
    return Re_f0xyz

# In[7]:

######## KEY:
######## this creates the full Bennet A.11 equation in ARRAY/VECTOR/COMPONENT form:
######## it does this by taking in those 4 arrays corresponding to 4 matrices
######## and then it uses Fvvsc_components_term1 and Fvvsc_components_term2 
######## to output the full array/vector representatation of the full Fvvsc term
####### the outputs should all be real
@nb.jit(nopython=True)
def Fvvsc_components(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4):
    term1 = Fvvsc_components_term1(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4)
    term2 = Fvvsc_components_term2(p0xyz_1, p0xyz_2, p0xyz_3, p0xyz_4)
    return term1 - term2