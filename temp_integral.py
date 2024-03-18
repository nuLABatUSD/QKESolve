#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy
import numba as nb
import matplotlib.pyplot as plt


# In[2]:

@nb.jit(nopython=True)
def F(p1, p2, p3):
    return np.array([1,0,.5,-1])

div_small_integer_fix = 1e-12
@nb.jit(nopython=True)
def p_index(p1, dp):
    return int(p1/dp + 1 + div_small_integer_fix)

@nb.jit(nopython=True)
def create_p(p1, e_max, B, dp):
    num = p_index((B*e_max - p1), dp)
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


# ## Integral 1

# In[4]:


@nb.jit(nopython=True)
def A_1(p1, p2, dp):
    x = create_p(0, p2, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_1(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[5]:


@nb.jit(nopython=True)
def I1(p1, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_1(p1, x[i], dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output





@nb.jit(nopython=True)
def A_2(p1, p2, dp):
    x = create_p(p2, p1, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_2(p1, p2)
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[9]:


@nb.jit(nopython=True)
def I2(p1, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_2(p1, x[i], dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output




@nb.jit(nopython=True)
def A_3(p1, p2, dp):
    x = create_p(p1,p1+p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_3(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[13]:


@nb.jit(nopython=True)
def I3(p1, p, e_max, B, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_3(p1,x[i],dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output


@nb.jit(nopython=True)
def A_4(p1, p2, dp):
    x = create_p(0, p1, 1, dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_1(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[17]:


@nb.jit(nopython=True)
def I4(p1, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_4(p1,x[i], dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output



# In[20]:


@nb.jit(nopython=True)
def A_5(p1, p2, dp):
    x = create_p(p1,p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_2(p2, p1)
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[21]:


@nb.jit(nopython=True)
def I5(p1, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_5(p1,x[i], dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output



# In[24]:


@nb.jit(nopython=True)
def A_6(p1, p2, dp):
    x = create_p(p2,p1+p2,1,dp)
    y = np.zeros((len(x), 4))
    for i in range(len(x)):
        y[i,:] = F(p1, p2, x[i])* J_3(p1, p2, x[i])
    A = np.zeros((4))
    for i in range(4):
        A[i] = np.trapz(y[:,i], x)
    return A


# In[25]:


@nb.jit(nopython=True)
def I6(p1, p, e_max, B, dp):
    x = create_p(p1, e_max, B, dp)
    y = np.zeros((len(x), 4))
    output = np.zeros((4))

    for i in range(len(x)):
        y[i,:] = A_6(p1,x[i],dp)

    for k in range(4):
        output[k] = np.trapz(y[:,k],x)
    
    return output


