import numpy as np
import numba as nb
import Ivvsc
import importlib
import time
import sys
import os
import multiprocessing as mp


filename = sys.argv[1]
N_processors = int(sys.argv[2])
if len(sys.argv) == 4:
    N_files = int(sys.argv[3])
else:
    N_files = 2
    
try:
    data_file = np.load(sys.argv[1], allow_pickle=True)
    

    run_ok = True
except:
    print("Data file {} not read. Abort.".format(sys.argv[1]))
    run_ok = False

settings = data_file['settings'].item()
raw = data_file['raw'].item()
prob = data_file['prob'].item()

nu3D = raw['nu3D']
nubar3D = raw['nubar3D']

N = settings['N'] + 1
e_max = settings['eps_max'] * settings['T']
B = 2

p = np.linspace(0, e_max, N)
delta_p = p[1] - p[0]

C_full = np.zeros_like(raw['nu3D'])
Cbar_full = np.zeros_like(raw['nubar3D'])

N_total = len(C_full)


output_filename = sys.argv[1][:-4]

if os.path.exists("{}-Collision.npz".format(output_filename)):
    print("Output file already exists. Abort.")
    run_ok = False

def calc_c_i(i):
    return Ivvsc.C_Ivvsc(nu3D[i,:,:], p, e_max, B, delta_p), Ivvsc.C_Ivvsc(nubar3D[i,:,:], p, e_max, B, delta_p)
#    CC[i, :, :] = Ivvsc.C_Ivvsc(nu3D[i,:,:], p, e_max, B, delta_p)
#    return

def calc_dummy(i):
    return i * np.ones((C_full.shape[1], C_full.shape[2])), -i * np.ones((C_full.shape[1], C_full.shape[2]))
#    CC[i, :, :] = i * np.ones((CC.shape[1], CC.shape[2]))
#    return

if __name__ == '__main__':

    if run_ok:
        pool = mp.Pool(N_processors)
        
        time_array = np.zeros(N_files)

        dN = len(C_full) // N_files

        for k in range(N_files-1):
            run_list = []
            for i in range(k*dN, (k+1)*dN):
                run_list.append((i))

            beg = time.time()
            res = pool.map(calc_c_i, run_list)
            time_array[k] = time.time() - beg
            
            for j, i in enumerate(run_list):
                C_full[i, :, :] = res[j][0]
                Cbar_full[i, :, :] = res[j][1]

            np.savez("{}-C{}.npz".format(output_filename, k), coll_nu = C_full[:(k+1)*dN], coll_nubar = Cbar_full[:(k+1)*dN])
            print("File {} of {}. Elapsed time {:.2f} hrs (this file {:.1f} min)".format(k+1, N_files, np.sum(time_array[:(k+1)])/60/60, time_array[k]/60))
        run_list = []
        for i in range((k+1)*dN, len(C_full)):
            run_list.append((i))

        beg = time.time()
        res = pool.map(calc_c_i, run_list)
        time_array[-1] = time.time() - beg
        for j, i in enumerate(run_list):
            C_full[i, :, :] = res[j][0]
            Cbar_full[i, :, :] = res[j][1]

        np.savez("{}-Collision.npz".format(output_filename), coll_nu = C_full, coll_nubar = Cbar_full)
        print("Total time {:.2f} hrs.".format(np.sum(time_array)/60/60))

        pool.close()
        pool.join()

