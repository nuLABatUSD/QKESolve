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
    
def solve_QKE(T, y0, incl_thermal_term, incl_anti, foldername, filename_head, incl_collisions = True, incl_eta = False, eta_e = 0, eta_mu = 0, overwrite_file = False, make_plot = True, Emax=10, dm2=dm2_atm, sin22th=sin22th_default, tau_final = 10, print_info=True, use_fixed_dN = False, dN_fixed = 5, N_fixed = 1000, dt_init = -1, t0 = 0):
    fn = foldername + '/' + filename_head + '.npz'
    
    if os.path.exists(fn):
        if not overwrite_file:
            print("File : {} already exists.  Abort".format(fn))
            
            return
        
    if incl_anti:
        N = len(y0) // 8
    else:
        N = len(y0) // 4
            
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    th = 0.5 * np.arcsin(sin22th)
    
    Eval = np.linspace(Emax/N, Emax, N)

        
    p= np.zeros(N+9)
    p[-1]= dm2
    p[-2]= th
    p[:N]= np.linspace(Emax/N, Emax, N)
    p[-3]= T

    if incl_thermal_term:
        p[-4] = 1
        
    if incl_anti:
        p[-5] = -1
    else:
        p[-5] = 0

    if incl_collisions:
        p[-6] = -1
        
    if incl_eta:
        p[-7] = -1
        p[-8] = eta_e
        p[-9] = eta_mu
    else:
        p[-7] = 0
        
    settings_dict = {
        'incl_thermal_term':incl_thermal_term,
        'incl_anti_neutrinos':incl_anti,
        'incl_collisions':incl_collisions,
        'incl_eta':incl_eta,
        'eta_e':eta_e,
        'eta_mu':eta_mu,
        'N': N,
        'eps_max': Emax,
        'delta m-squared': dm2,
        'sin^2 (2theta)': sin22th,
        'T':T
    }    
    
    #t0= 0
    if dt_init > 0:
        dt0 = dt_init
    else:
        dt0=  0.01/np.max(np.abs(der.f(t0,y0,p)))

    N_step = 1000
    dN = 5
    
    if use_fixed_dN:
        dN = dN_fixed
        N_step = N_fixed
    #tau_final=10
    t_final = tau_final*2*2.2*T/dm2

    t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)
    
    
    if not end and not use_fixed_dN:
        dN *= np.ceil((t_final-t0)/(t[-1]-t0)) + 1
        dN = int(dN)
        
        t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)

    raw_data_dict = {
        'initial array': y0,
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
        
        ym0, ym0_bar= der.newmatrix_maker(y0)
        
        raw_data_dict['nu_init matrix'] = ym0
        raw_data_dict['nu_bar_init matrix'] = ym0_bar

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
        
        ym0= der.matrix_maker(y0)
        
        raw_data_dict['nu_init matrix'] = ym0

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
            
    if return_final_state:
        return t[-1], dx[-1], y[-1,:]

