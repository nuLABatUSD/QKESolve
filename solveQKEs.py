
import numpy as np
import matplotlib.pyplot as plt
import ODESolve as ODE
import derivatives as der

import os

dm2_atm = 2.5e-15
sin22th_default = 0.8


def prob_plot(tau,prob_ve):
    plt.figure()
    plt.plot(tau, prob_ve)
    plt.xlabel(r'$\tau$')
    plt.ylabel("Probability")
    plt.show()
    
    return
    
def solve_all_electron_ic(T, incl_thermal_term, incl_anti, foldername, filename_head, overwrite_file = False, make_plot = True, N=100, Emax=10, dm2=dm2_atm, sin22th=sin22th_default, print_info=True):
    fn = foldername + '/' + filename_head + '.npz'
    
    if os.path.exists(fn):
        if not overwrite_file:
            print("File : {} already exists.  Abort".format(fn))
            
            return
            
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    th = 0.5 * np.arcsin(sin22th)
    
    Eval = np.linspace(Emax/N, Emax, N)
    
    ym0 = np.zeros((N,4))
    for i in range(len(ym0)):
        p0 = 1/(np.exp(Eval[i])+1)
        ym0[i,:] = [p0,0,0,1]
        
    p= np.zeros(N+5)
    p[-1]= dm2
    p[-2]= th
    p[:N]= np.linspace(Emax/N, Emax, N)
    p[-3]= T

    if incl_thermal_term:
        p[-4] = 1
        
    if incl_anti:
        p[-5] = -1
        
        ym0_bar= np.zeros((N,4))
        for i in range(len(ym0_bar)):
            p0= 1/(np.exp(Eval[i])+1)
            ym0_bar[i,:]= [p0,0,0,1]
            
        y0 = der.newarray_maker(ym0, ym0_bar)
    else:
        p[-5] = 0
        y0 = der.array_maker(ym0)
        
    settings_dict = {
        'incl_thermal_term':incl_thermal_term,
        'incl_anti_neutrinos':incl_anti,
        'N': N,
        'eps_max': Emax,
        'delta m-squared': dm2,
        'sin^2 (2theta)': sin22th,
        'T':T
    }    
    
    t0= 0
    dt0=  0.01/np.max(np.abs(der.f(t0,y0,p)))

    N_step = 1000
    dN = 5
    tau_final=10
    t_final = tau_final*2*2.2*T/dm2

    t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)
    
    
    if not end:
        dN *= np.ceil(t_final/t[-1]) + 1
        dN = int(dN)
        
        t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)

    raw_data_dict = {
        'time': t,
        'dt': dx,
        'dN' : dN
    }
    
    tau = t * (dm2 / (2 * 2.2 * T))
    probability_data = {
        'tau': tau
    }
    
    if incl_anti:
        nu_3d_matrix = der.threeD(y[:,:4*N])
        nubar_3d_matrix = der.threeD(y[:,4*N:])

        raw_data_dict['nu3D'] = nu_3d_matrix
        raw_data_dict['nubar3D'] = nubar_3d_matrix
        
        try:
            prob_ve = der.probability(ym0, Eval, t, y[:,:4*N])
            
            if make_plot:
                prob_plot(tau,prob_ve)
                
            probability_data['prob_ve'] = prob_ve
        except:
            print("No initial neutrinos; probability ill defined")
            
        try:
            prob_vebar = der.probability(ym0_bar, Eval, t, y[:,4*N:])
            if make_plot:
                prob_plot(tau,prob_vebar)
                
            probability_data['prob_vebar'] = prob_vebar
        except:
            print("No initial anti-neutrinos; probability ill defined")
        
    else:
        nu_3d_matrix = der.threeD(y)
        
        raw_data_dict['nu3D'] = nu_3d_matrix
        
        prob_ve = der.probability(ym0, Eval, t, y)
        probability_data['prob_ve'] = prob_ve
        
        if make_plot:
            prob_plot(tau,prob_ve)
    
    np.savez(fn, settings=settings_dict, raw=raw_data_dict, prob = probability_data)

    if print_info:
        print("Data saved to file " + fn)
        print("{} time steps saved with dN = {}".format(len(t), dN))
        for i in settings_dict:
            print(i, settings_dict[i])
