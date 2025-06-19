import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import copy
import torch

from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, mpc_worker, \
    compare_state_dot_with_res

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC
from TrajProc.models.ResDSKBM import ResDSKBM

from MPCPlotter import MPCPlotter

# TODO: YAMLify this.
SIM_DURATION = 200  # time steps ## NOT IN YAML

def main(data_dir):    
    
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
    plt.ion()

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
    l_kbm = []
    l_sim = []
    l_res = []
    for sim_time in range(SIM_DURATION - 1):
        # Solves QP and returns optimal action in u_sim_opt (converted from MPC to sim action)
        current_model = copy.deepcopy(mpc.model)
        u_sim_opt, u_mpc_opt, l_ref_idx = plan_mpc_step(
            x_sim[:, sim_time],
            u_bar_start,
            track,
            tp,
            mpc,
            mpc_params,
            return_mpc_action=True
        )
        

        # Store action to be taken this sim_time into u_sim
        u_sim[:, sim_time] = u_sim_opt.numpy()

        # Update plots
        mpc_plotter.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)
        plt.pause(0.0001)
                
        # From pybullet
        x_dot_k = step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger, params['mpc']['dt']) # (4,)
        l_sim.append(x_dot_k) # (4,)
        
        # Update current model's queue
        current_model.update_queue(x_dot_k, u_mpc_opt.numpy())

        # [DEBUG] KBM on x_sim[:, sim_time] and u_mpc_opt
        x_dot_kbm = current_model.f_x_dot(x_sim[:, sim_time], u_mpc_opt.numpy())
        x_dot_kbm = x_dot_kbm.full().T # (1, 4)
        x_dot_kbm = x_dot_kbm.squeeze(0)
        l_kbm.append(x_dot_kbm)
        
        # [DEBUG] LSTM on current window
        res = current_model.query_lstm(useHidden=True, updateHidden=True)
        l_res.append(res.squeeze(0))

        mpc.model = copy.deepcopy(current_model)
    
    kbm_xdot = np.array(l_kbm)
    res_xdot = torch.cat(l_res).reshape(-1, 4).to("cpu").numpy()
    sim_xdot = torch.cat(l_sim).reshape(-1, 4).to("cpu").numpy()
    compare_state_dot_with_res(sim_xdot, kbm_xdot, res_xdot)
    
if __name__ == "__main__":
    main("models/velocity_model")
