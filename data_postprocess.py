import numpy as np
import glob
import os
from MPCFigurePlotter import MPCFigurePlotter
import matplotlib.pyplot as plt

def find_padding_index(xsim, u):
    # Check each timestep (i.e., column) for all zeros in both xsim and u
    for t in range(xsim.shape[1] - 1):
        if np.all(xsim[:, t] == 0.0) and np.all(u[:, t] == 0.0):
            return t  # padding starts at this index
    return u.shape[1]  # no padding found, use full length

def main(fp, model_name):
    l_x_sim  = [np.load(x) for x in sorted(glob.glob(os.path.join(fp, "numpy/x_sim_track*.npy")))]
    l_u_mpc     = [np.load(x) for x in sorted(glob.glob(os.path.join(fp, "numpy/u_mpc_track*.npy")))]
    l_track     = [np.load(x) for x in sorted(glob.glob(os.path.join(fp, "track*.npy")))]
    
    # plt.ion()
    
    for i in range(len(l_x_sim)):
        xsim = l_x_sim[i]
        u = l_u_mpc[i]
        track = l_track[i]
        plotter = MPCFigurePlotter(track, 200, model_name, i)
        
        T = find_padding_index(xsim, u)
        
        for i_timestep in range(T):
            plotter.plot_new_data(xsim[:, i_timestep], u[:, i_timestep], i_timestep)
        plt.pause(0.00001)
        plotter.save_plot_and_close(f"more_figures/track_{i}")
        

if __name__ == "__main__":
    # main("results_lstm", "CRN with CSLQ-MPC") # crn
    # main("results_no_lstm", "DSKBM with SLQ-MPC") # dskbm
    # main("results_sysid", "CRN with Sys-ID Mismatch on CSLQ-MPC") #crn-sysid
    # main("results_sysid_no_lstm", "DSKBM with Sys-ID Mismatch on SLQ-MPC") #dskbm mismatch
    main("results_pdn", "PDN with CSLQ-MPC") #dskbm mismatch