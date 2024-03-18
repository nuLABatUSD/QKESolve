import numpy as np
import matplotlib.pyplot as plt
import ODESolve as ODE
import derivatives as der
import time

import os


dm2_atm = 2.5e-15
sin22th_default = 0.8
mpl = 1.221e22

gss = np.loadtxt("SMgstar.dat", usecols=(0,1,2), unpack=True)

def Hubble(T):
    index = np.where(gss[0] < T)[0][-1]
    rho = np.pi**2 * T**4 / 30 * gss[1][index]
    H2 = 8 * np.pi / 3 / mpl**2 * rho
    return np.sqrt(H2)

def prob_plot(tau,prob_ve):
    plt.figure()
    plt.plot(tau, prob_ve)
    plt.xlabel(r'$\tau$')
    plt.ylabel("Probability")
    plt.show()
    
    return
    
def make_dictionaries(t, y, dx, N, dN, y0, Eval, T, dm2, incl_anti, make_plot=False):
    raw_data_dict = {
        'initial array': y0,
        'time': t,
        'Ht': Hubble(T) * t,
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

    return raw_data_dict, probability_data

    
def solve_QKE_time(T, y0, incl_thermal_term, incl_anti, foldername, filename_head, incl_collisions = True, incl_eta = False, eta_e = 0, eta_mu = 0, overwrite_file = False, make_plot = True, Emax=10, dm2=dm2_atm, sin22th=sin22th_default, tau_final = 10, print_info=True, use_fixed_dN = False, dN_fixed = 5, N_fixed = 1000, dt_init = -1, t0 = 0, return_final_state = False, use_max_run_time = False, max_hours = 12, use_Ht = False, Ht_max = 0.01, save_verbose = True, progress_saves = True, num_save_files = 10):
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
    
    if use_max_run_time or use_Ht:
        H = Hubble(T)
        t_final = max(t_final, Ht_max / H)
        

    begin_time = time.time()
    t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)
    while_run = 1
    while use_max_run_time and (time.time() - begin_time) < 120:
        t, y, dx, end = ODE.ODEOneRun(t[-1], y[-1,:], dx[-1], p, N_step, dN, t_final)
        while_run += 1
    total_time = time.time() - begin_time
    
    time_array = np.zeros(num_save_files)
    if not end and not use_fixed_dN:
        mf = (np.ceil(while_run * (t_final-t0)/(t[-1]-t0)) + 1)
        if use_max_run_time:
            mf = min(mf, while_run * max_hours * 3600 / total_time + 1)
        dN *= mf
        dN = int(dN)
        
        
        if progress_saves:
            dN = dN // num_save_files
            begin_time = time.time()
            
        t, y, dx, end = ODE.ODEOneRun(t0, y0, dt0, p, N_step, dN, t_final)
                
        if progress_saves:
            time_array[0] = time.time() - begin_time
            t1 = t[::num_save_files]
            y1 = y[::num_save_files,:]
            dx1 = dx[::num_save_files]

            raw_data_dict, probability_data = make_dictionaries(t, y, dx, N, dN, y0, Eval, T, dm2, incl_anti, make_plot=False)
            
            if t1[-1] != t[-1]:
                t = np.append(t1, t[-1])
                y = np.append(y1, y[-1,:])
                dx = np.append(dx1, dx[-1])
            else:
                t = np.copy(t1)
                y = np.copy(y1)
                dx = np.copy(dx1)
            
            np.savez(fn[:-4] + "-0.npz",  settings=settings_dict, raw=raw_data_dict, prob = probability_data)
            
            if save_verbose:
                print("Run 1 of {}, elapsed time {:.2f} hrs (time for run {:.1f} min)".format(num_save_files, np.sum(time_array)/3600, time_array[0]/60))
            for i in range(1, num_save_files):
                begin_time = time.time()
                t1, y1, dx1, end = ODE.ODEOneRun(t[-1], y[-1,:], dx[-1], p, N_step, dN, t_final)
                
                time_array[i] = time.time() - begin_time

                t_old = np.copy(t)
                y_old = np.copy(y)
                dx_old = np.copy(dx)
                
                t_new = t1[::num_save_files]
                y_new = y1[::num_save_files,:]
                dx_new = dx1[::num_save_files]
                
                if t_new[-1] != t1[-1]:
                    t_new = np.append(t_new, t1[-1])
                    y_new = np.append(y_new, [y1[-1,:]])
                    dx_new = np.append(dx_new, dx1[-1])

                t = np.zeros(len(t_old) + len(t_new))
                y = np.zeros((len(t), len(y[0])))
                dx = np.zeros(len(t))

                t[:len(t_old)] = t_old
                y[:len(t_old), :] = y_old
                dx[:len(t_old)] = dx_old

                t[len(t_old):] = t_new
                y[len(t_old):, :] = y_new
                dx[len(t_old):] = dx_new
                
                raw_data_dict, probability_data = make_dictionaries(t1, y1, dx1, N, dN, y0, Eval, T, dm2, incl_anti, make_plot=False)
                
                np.savez(fn[:-4] + "-{}.npz".format(i), settings=settings_dict, raw=raw_data_dict, prob = probability_data)
            
                if save_verbose:
                    print("Run {} of {}, elapsed time {:.2f} hrs (time for run {:.1f} min)".format(i+1,num_save_files, np.sum(time_array)/3600, time_array[i]/60))

    if progress_saves:
        dN *= num_save_files

    raw_data_dict, probability_data = make_dictionaries(t, y, dx, N, dN, y0, Eval, T, dm2, incl_anti, make_plot=make_plot)
    
    np.savez(fn, settings=settings_dict, raw=raw_data_dict, prob = probability_data)
    
    if progress_saves:
        np.save(fn[:-4] + "-times", time_array)

    if print_info:
        print("Data saved to file " + fn)
        print("{} time steps saved with dN = {}".format(len(t), dN))
        for i in settings_dict:
            print(i, settings_dict[i])
            
    if return_final_state:
        return t[-1], dx[-1], y[-1,:]

