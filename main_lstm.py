import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import copy

from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, mpc_worker

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC
from TrajProc.models.ResDSKBM import ResDSKBM

from MPCPlotter import MPCPlotter

import multiprocessing as mp

# TODO: YAMLify this.
SIM_DURATION = 200  # time steps ## NOT IN YAML

def main(data_dir):    
    plt.ion()
    
    # PyBullet + robot setup
    params, mpc_params, sim = init_pybullet_sim(data_dir)    
    params['mpc'] = mpc_params
    
    

    sim.resetRobot(pose=((0,0),(0, 0, 0, np.sqrt(2)/2)))
    
    tp = TrajProc()
    # cs_kbm = csDSKBM(mpc_params)
    model = ResDSKBM(params)

    # Creating a sample trajectory to track
    track = create_debug_track(tp, params['terrain'])

    mpc_plotter = MPCPlotter(track, SIM_DURATION)

    # VARIABLES FOR TRACKING
    nX = mpc_params['model']['nStates']
    mU = mpc_params['model']['nActions']
    L = params['train']['trainPredSeqLen']
    x_sim = np.zeros((nX, SIM_DURATION))
    u_sim = np.zeros((mU, SIM_DURATION - 1))

    # Initialize as numpy for storage, convert to tensor whenever we need
    from collections import deque
    xdot_q  = deque([np.zeros(nX)] * L, L)
    u_q     = deque([np.zeros(mU)] * L, L)
    
    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, 0.0, 0.0]).T
    xdot_q.appendleft(x_sim[:, 0])
    
    # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
    u_bar_start = init_u_bar(mpc_params)

    mpc = MPC(mpc_params, model)
    
    state_logger = init_logger()
    
    ## MULTIPROCESSING (try)
    for sim_time in range(SIM_DURATION - 1):
        # Solves QP and returns optimal action in u_sim_opt (converted from MPC to sim action)
        current_model = copy.deepcopy(mpc.model)
        u_sim_opt, u_mpc_opt, l_ref_idx = plan_mpc_step(x_sim[:, sim_time], u_bar_start, track, tp, mpc, mpc_params, return_mpc_action=True)
        
        # Store action to be taken this sim_time into u_sim
        u_sim[:, sim_time] = u_sim_opt.numpy()

        # Update plots
        mp.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)
        plt.pause(0.0001)
                
        x_dot_k = step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger)

        # TODO: check current u_mpc_opt datastructure
        # TODO: check all of the rest too (i.e. updating queue, updating hidden layer)
        breakpoint()
        current_model.u_q.appendleft(u_mpc_opt)
        current_model.xdot_q.appendleft(x_dot_k)
        _, _, current_model.hidden = current_model.lstm(model.window_from_queue(model.xdot_q), model.window_from_queue(model.u_q))
        mpc.model = current_model

if __name__ == "__main__":
    main("debug_model")
