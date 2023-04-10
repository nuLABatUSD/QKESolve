#!/usr/bin/env python
# coding: utf-8

# In[75]:


import SOLVING_QKE_FUNCTION as solve
import numpy as np
import derivatives as der
import matplotlib.pyplot as plt
import time

def solve_contd(foldername, filename, tau_final, newfilename, make_plot = True, print_info=False):
    if newfilename == filename:
        newfilename += "-1"
    old_data = np.load(foldername + "/" + filename + ".npz", allow_pickle=True)
    sd = old_data['settings'].item()
    rd = old_data['raw'].item()
    pd = old_data['prob'].item()

    time = rd['time']
    dt = rd['dt']
    nu3D = rd['nu3D']
    nubar3D = rd['nubar3D']
    y1 = der.newarray_maker(nu3D[-1], nubar3D[-1])
    
    solve.solve_QKE(sd['T'], y1, sd['incl_thermal_term'], sd['incl_anti_neutrinos'], foldername, newfilename, incl_collisions = sd['incl_collisions'], incl_eta = sd['incl_eta'], eta_e = sd['eta_e'], eta_mu = sd['eta_mu'], Emax = sd['eps_max'], dm2 = sd['delta m-squared'], sin22th = sd['sin^2 (2theta)'], tau_final = tau_final, t0 = time[-1], dt_init=dt[-1], make_plot = make_plot, print_info = print_info)


def solve_series(foldername, first_filename, tau_final, steps):
    first_data = np.load(foldername + "/" + first_filename + ".npz", allow_pickle=True)
    pd = first_data['prob'].item()
    
    tau_array = np.linspace(pd['tau'][-1], tau_final, steps)
    
    if tau_array[-1] <= tau_array[0]:
        print("tau_final must be greater than {}".format(tau_array[0]))
        return
    
    taus = []
    probs_ve = []
    probs_vebar = []
    
    taus.append(pd['tau'])
    probs_ve.append(pd['prob_ve'])
    probs_vebar.append(pd['prob_vebar'])
    
    print(r"Solve from $\tau$ = {} to $\tau$ = {}".format(tau_array[0], tau_array[1]))
    st = time.time()
    solve_contd(foldername, first_filename, tau_array[1], "{}-{}".format(first_filename,1), make_plot = False)
    print("Time elapse, {:.1f} seconds".format(time.time()-st))
    
    data = np.load(foldername + "/" + "{}-{}.npz".format(first_filename,1), allow_pickle = True)
    pd = data['prob'].item()
    
    taus.append(pd['tau'])
    probs_ve.append(pd['prob_ve'] * probs_ve[0][-1])
    probs_vebar.append(pd['prob_vebar'] * probs_vebar[0][-1])

    plt.figure()
    for j in range(len(taus)):
        plt.plot(taus[j], probs_ve[j])
        
    plt.figure()
    for j in range(len(taus)):
        plt.plot(taus[j], probs_vebar[j])
    plt.show()
    
    for i in range(2, steps):
        print(r"Solve from $\tau$ = {} to $\tau$ = {}".format(tau_array[i-1], tau_array[i]))

        st = time.time()
        solve_contd(foldername, "{}-{}".format(first_filename, i-1), tau_array[i], "{}-{}".format(first_filename, i), make_plot=False)
        print("Time elapse, {:.1f} seconds".format(time.time()-st))
        
        data = np.load(foldername + "/" + "{}-{}.npz".format(first_filename,i), allow_pickle = True)
        pd = data['prob'].item()

        taus.append(pd['tau'])
        probs_ve.append(pd['prob_ve'] * probs_ve[i-1][-1])
        probs_vebar.append(pd['prob_vebar'] * probs_vebar[i-1][-1])

        plt.figure()
        for j in range(len(taus)):
            plt.plot(taus[j], probs_ve[j])

        plt.figure()
        for j in range(len(taus)):
            plt.plot(taus[j], probs_vebar[j])
        plt.show()
        
    np.savez(foldername + "/{}-total_prob".format(first_filename), tau = np.array(taus, dtype=object), prob_ve = np.array(probs_ve, dtype=object), prob_vebar = np.array(probs_vebar, dtype=object))
    


# In[3]:


def make_ics_eqm(N, Emax, eta_e, eta_mu):
    eps= np.linspace(Emax/N, Emax, N)
    ym0 = np.zeros((N,4))
    
    rho_ee = 1/(np.exp(eps-eta_e)+1)
    rho_mm = 1/(np.exp(eps-eta_mu)+1)

    for i in range(len(ym0)):
        p0 = rho_ee[i] + rho_mm[i]
        pz = (rho_ee[i] - rho_mm[i]) / p0
        ym0[i,:] = [p0,0,0,pz]

    
    rhobar_ee = 1/(np.exp(eps+eta_e)+1)
    rhobar_mm = 1/(np.exp(eps+eta_mu)+1)
    
    
    ym0_bar= np.zeros((N,4))
    for i in range(len(ym0_bar)):
        p0= rhobar_ee[i] + rhobar_mm[i]
        pz = (rhobar_ee[i] - rhobar_mm[i]) / p0
        ym0_bar[i,:]= [p0,0,0,pz]
        
    return der.newarray_maker(ym0,ym0_bar)


