import numpy as np
import glob
import os
from plotting.MPCFigureComparisonPlotter import MPCFigureComparisonPlotter
import matplotlib.pyplot as plt

def find_padding_index(xsim, u):
    # Check each timestep (i.e., column) for all zeros in both xsim and u
    for t in range(xsim.shape[1] - 1):
        if np.all(xsim[:, t] == 0.0) and np.all(u[:, t] == 0.0):
            return t  # padding starts at this index
    return u.shape[1]  # no padding found, use full length

def main(fp_1, fp_2, model_1_name, model_2_name):
    l_x_sim_1  = [np.load(x) for x in sorted(glob.glob(os.path.join(fp_1, "numpy/x_sim_track*.npy")))]
    l_u_mpc_1  = [np.load(x) for x in sorted(glob.glob(os.path.join(fp_1, "numpy/u_mpc_track*.npy")))]
    l_track    = [np.load(x) for x in sorted(glob.glob(os.path.join(fp_1, "track*.npy")))]

    l_x_sim_2  = [np.load(x) for x in sorted(glob.glob(os.path.join(fp_2, "numpy/x_sim_track*.npy")))]
    l_u_mpc_2  = [np.load(x) for x in sorted(glob.glob(os.path.join(fp_2, "numpy/u_mpc_track*.npy")))]
    
    for i in range(len(l_x_sim_1)):
        x_sim_1 = l_x_sim_1[i]
        u_1     = l_u_mpc_1[i]
        x_sim_2 = l_x_sim_2[i]
        u_2     = l_u_mpc_2[i]

        track = l_track[i]
        plotter = MPCFigureComparisonPlotter(track, 200, model_1_name, model_2_name, i)
        
        T_1 = find_padding_index(x_sim_1, u_1)
        T_2 = find_padding_index(x_sim_1, u_1)
        
        plotter.plot_full_data(x_sim_1[:, :T_1], u_1[:, :T_1], x_sim_2[:, :T_2], u_2[:, :T_2])
        # for i_timestep in range(T):
            # plotter.plot_new_data(xsim[:, i_timestep], u[:, i_timestep], i_timestep)
        
        plt.pause(0.00001)
        plotter.save_plot_and_close(f"final_figures/track_{i}")
        

if __name__ == "__main__":
    # main("results_lstm", "CRN with CSLQ-MPC") # crn
    # main("results_no_lstm", "DSKBM with SLQ-MPC") # dskbm
    # main("results_sysid", "CRN with Sys-ID Mismatch on CSLQ-MPC") #crn-sysid
    # main("results_sysid_no_lstm", "DSKBM with Sys-ID Mismatch on SLQ-MPC") #dskbm mismatch
    # main("results_lstm", "results_pdn", "CRN with CSLQ-MPC", "PDN with CSLQ-MPC") #dskbm mismatch
    # main("results_lstm", "results_no_lstm", "CRN with CSLQ-MPC", "DSKBM with SLQ-MPC") #dskbm mismatch
    main("results_sysid", "results_sysid_no_lstm", "Mismatched CRN with CSLQ-MPC", "Mismatched DSKBM with SLQ-MPC")