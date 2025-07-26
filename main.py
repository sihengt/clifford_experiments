import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, \
    compare_state_dot

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC

from MPCPlotter import MPCPlotter

import multiprocessing as mp

# TODO: YAMLify this.
SIM_DURATION = 200  # time steps ## NOT IN YAML

def main(data_dir):    
    plt.ion()
    
    # PyBullet + robot setup
    params, mpc_params, sim = init_pybullet_sim(data_dir)
    
    sim.resetRobot(pose=((0,0),(0, 0, 0, np.sqrt(2)/2)))
    
    tp = TrajProc()
    cs_kbm = csDSKBM(mpc_params)

    # Creating a sample trajectory to track
    track = create_debug_track(tp, params['terrain'])
    # track = create_random_track(tp, params['terrain'], 5)

    mpc_plotter = MPCPlotter(track, SIM_DURATION)

    # VARIABLES FOR TRACKING
    x_sim = np.zeros((mpc_params['model']['nStates'], SIM_DURATION))
    u_sim = np.zeros((mpc_params['model']['nActions'], SIM_DURATION - 1))

    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, 0.0, 0.0]).T

    # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
    u_bar_start = init_u_bar(mpc_params)

    mpc = MPC(mpc_params, cs_kbm)
    
    state_logger = init_logger()
    
    l_pybullet_xdot = []
    l_kbm_xdot = []
    ## MULTIPROCESSING (try)
    breakpoint()
    for sim_time in range(SIM_DURATION - 1):
        # Solves QP and returns optimal action in u_sim_opt (converted from MPC to sim action)
        u_sim_opt, u_mpc, l_ref_idx= plan_mpc_step(
            x_sim[:, sim_time],
            u_bar_start,
            track,
            tp,
            mpc,
            mpc_params,
            return_mpc_action=True,
            return_all_mpc_actions=True
        )

        # Store action to be taken this sim_time into u_sim
        u_sim[:, sim_time] = u_sim_opt.numpy()

        # Update plots
        mpc_plotter.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)
        plt.pause(0.0001)
                
        x_dot_km1 = step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger, mpc_params['dt'])

        # We want to check if x_dot makes sense. The best way to compare it is to collect x_dot AND collect the 
        # output of the bicycle model into two arrays and plot them right after.
        l_pybullet_xdot.append(x_dot_km1)
        kbm_xdot = cs_kbm.f_x_dot(x_sim[:, sim_time], u_mpc[:, 0]).full()
        l_kbm_xdot.append(kbm_xdot) # results from running kbm with previous action and previous

        # [DEBUG] Warm-starting
        # u_bar_start[:, :-1] = u_mpc[:, 1:]
        # u_bar_start[:, -1] = u_mpc[:, -1]

    import torch
    l_kbm_xdot = np.hstack(l_kbm_xdot)
    l_pybullet_xdot = torch.stack(l_pybullet_xdot)
    compare_state_dot(l_pybullet_xdot, l_kbm_xdot)

if __name__ == "__main__":
    main("nominal_dec_8")
