import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, mpc_worker

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
    
    ## NEW CODE
    import torch
    from learning.architecture.dynamicsModel import AdaptiveDynamicsModel
    from TrajProc.models.ResDSKBM import ResDSKBM

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    lstm = AdaptiveDynamicsModel(params['network'], params['controls'], mpc_params).to(device)
    lstm.load_state_dict(torch.load(os.path.join(data_dir,'adm.pt'), map_location=torch.device(device)))
    lstm.eval()
    
    params['mpc'] = mpc_params
    model = ResDSKBM(params)

    breakpoint()
    
    ## NEW CODE DONE

    sim.resetRobot(pose=((0,0),(0, 0, 0, np.sqrt(2)/2)))
    
    tp = TrajProc()
    cs_kbm = csDSKBM(mpc_params)

    # Creating a sample trajectory to track
    track = create_debug_track(tp, params['terrain'])

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
    
    ## MULTIPROCESSING (try)
    for sim_time in range(SIM_DURATION - 1):
        # Solves QP and returns optimal action in u_sim_opt (converted from MPC to sim action)
        u_sim_opt, l_ref_idx = plan_mpc_step(x_sim[:, sim_time], u_bar_start, track, tp, mpc, mpc_params)
                
        # Store action to be taken this sim_time into u_sim
        u_sim[:, sim_time] = u_sim_opt.numpy()

        # Update plots
        mp.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)
        plt.pause(0.0001)
                
        step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger)

if __name__ == "__main__":
    main("debug_model")
