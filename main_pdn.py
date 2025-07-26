import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import copy
import torch

from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, mpc_worker, \
    compare_state_dot, compare_state_dot_with_full, compare_state_dot_with_res, analyze_state_dot_with_res, save_numpy_array_to_file, analyze_state_dot_with_full, draw_debug_track, \
    make_track_segment

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC
from TrajProc.models.ResDSKBM import ResDSKBM

from MPCPlotter import MPCPlotter

import pandas as pd

# TODO: YAMLify this.
SIM_DURATION = 200  # time steps ## NOT IN YAML

def check_lap_completion(last_idx, nn_idx, path_length, threshold=50, debug=False):
    """
    Determines if we've completed a lap.
    
    Returns True if lastIndex was near the end, and nn_idx wrapped around to near the start.
    
    Parameters:
        nn_idx: current nearest index on path
        path_length: total number of points on path
        threshold: number of indices from either end to count as "near"
    """    
    near_end_before = last_idx > path_length - threshold
    near_start_now = nn_idx < threshold
    
    if debug:
        print(f"Near End: {last_idx} > {path_length} - {threshold} = {near_end_before}")
        print(f"Near Start: {nn_idx} < {threshold} = {near_start_now}")

    if near_end_before and near_start_now:
        return True
    return False

def main(data_dir):
    # Evaluation tracks:
    eval_tracks = np.load("poisson_generated_tracks.npy")

    # Setup structure to keep data you need from the different runs.
    l_df = []

    for i_track in range(eval_tracks.shape[0]):
        # PyBullet + robot setup
        params, mpc_params, sim = init_pybullet_sim(data_dir)    
        params['mpc'] = mpc_params

        tp = TrajProc()
        # cs_kbm = csDSKBM(mpc_params)
        model = ResDSKBM(params)

        # Creating a sample trajectory to track
        # track = create_debug_track(tp, params['terrain'])

        # TODO list.
        track = eval_tracks[i_track]
        starting_angle = np.arctan2(track[1, 1], track[1, 0])
        starting_q = p.getQuaternionFromEuler([0, 0, starting_angle])

        sim.resetRobot(pose=((0,0), starting_q))

        xs = track[:, 0] * params['terrain']['mapScale'] 
        ys = track[:, 1] * params['terrain']['mapScale']
        track = tp.generate_path_from_wp(
            xs,
            ys,
            0.05
        )

        cmap = plt.get_cmap("plasma")
        n_seg   = xs.shape[0] - 1
        colour_table = cmap(np.linspace(0, 1, n_seg))[:, :3]   # keep only RGB

        # draw_debug_track(xs, ys)
        z = 0.10                     # lift the track a little above ground
        for i in range(len(xs) - 1):
            make_track_segment([xs[i],   ys[i],   z],
                            [xs[i+1], ys[i+1], z],
                            radius = 0.015,
                            rgba   = colour_table[i].tolist() + [1,])
        # make_track_segment(xs, ys)
        
        save_numpy_array_to_file(track, f"results/track{i_track}.npy")

        mpc_plotter = MPCPlotter(track, SIM_DURATION)
        plt.ion()

        # VARIABLES FOR TRACKING
        nX = mpc_params['model']['nStates']
        mU = mpc_params['model']['nActions']
        L = params['train']['trainPredSeqLen']

        x_sim = np.zeros((nX, SIM_DURATION))
        u_sim = np.zeros((mU, SIM_DURATION - 1))
        u_mpc = np.zeros((mU, SIM_DURATION - 1))
            
        # Step 2: Create starting conditions x0
        x_sim[:, 0] = np.array([0.0, 0.0, 0.0, starting_angle]).T
        
        # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
        u_bar_start = init_u_bar(mpc_params)

        mpc = MPC(mpc_params, model)
        
        state_logger = init_logger()
        
        ## MULTIPROCESSING (try)
        l_kbm = []
        l_sim = []
        l_res = []
        prev_idx = 0
        l_cte = []
        for sim_time in range(SIM_DURATION - 1):
            # Solves QP and returns optimal action in u_sim_opt (converted from MPC to sim action)
            current_model = copy.deepcopy(mpc.model)

            # Calls mpc.predict
            u_sim_opt, l_u_mpc_opt, l_ref_idx = plan_mpc_step(
                x_sim[:, sim_time],
                u_bar_start,
                track,
                tp,
                mpc,
                mpc_params,
                return_mpc_action=True,
                return_all_mpc_actions=True,
                n_iters=10
            )

            # Check if we've lapped
            if check_lap_completion(prev_idx, l_ref_idx[0], track.shape[1]):
                break

            prev_idx = l_ref_idx[0]

            # Update current_model's queue (current model has not been touched at this point)
            u_bar_start = l_u_mpc_opt
            u_mpc_opt = l_u_mpc_opt[:, -1]
            current_model.update_queue(x_sim[:, sim_time], u_mpc_opt)
            
            # Store action to be taken this sim_time into u_sim
            u_sim[:, sim_time] = u_sim_opt.numpy()

            # Store mpc action into u_mpc_opt
            u_mpc[:, sim_time] = u_mpc_opt

            # Update plots
            mpc_plotter.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)
            plt.pause(0.0001)
                    
            # From pybullet
            x_dot_k = step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger, params['mpc']['dt']) # (4,)
            l_sim.append(x_dot_k) # (4,)

            l_cte.append(np.linalg.norm(x_sim[:2, sim_time] - track[:2, l_ref_idx[0]]))

            # [DEBUG] KBM on x_sim[:, sim_time] and u_mpc_opt
            x_dot_kbm = current_model.f_x_dot(x_sim[:, sim_time], u_mpc_opt).full().T.squeeze(0) 
            l_kbm.append(x_dot_kbm)
            
            # [DEBUG] LSTM on current window
            res = current_model.query_lstm(useHidden=False, updateHidden=False)
            l_res.append(res.squeeze(0))

            mpc.model = copy.deepcopy(current_model)

        # Save figure of the run
        # TODO: can we save an animation of the figure?
        # TODO: can we use mpc_plotter to handle this instead of calling this directly from here?
        mpc_plotter.convert_frames_into_video(f"results/MPCResultsTrack{i_track}.mp4")
        mpc_plotter.save_plot_and_close(f"results/MPCResultsTrack{i_track}.png")
        
        # Comparing kbm, residual and simulator.
        kbm_xdot = np.array(l_kbm)
        res_xdot = torch.cat(l_res).reshape(-1, 4).to("cpu").numpy()
        sim_xdot = torch.cat(l_sim).reshape(-1, 4).to("cpu").numpy()
        
        # compare_state_dot_with_res(sim_xdot, kbm_xdot, res_xdot, fig_fp=f"results/DynamicsComparisonTrack{i_track}.png")
        compare_state_dot_with_full(sim_xdot, kbm_xdot, res_xdot, fig_fp=f"results/DynamicsComparisonTrack{i_track}.png")
        # compare_state_dot(sim_xdot, kbm_xdot)
        # rmse_sim_kbm, rmse_sim_res_kbm = analyze_state_dot_with_res(sim_xdot, kbm_xdot, res_xdot)
        rmse_sim_kbm, rmse_sim_res_kbm = analyze_state_dot_with_full(sim_xdot, kbm_xdot, res_xdot)

        # Just in case.
        save_numpy_array_to_file(x_sim, f"results/numpy/x_sim_track{i_track}.npy")
        save_numpy_array_to_file(u_mpc, f"results/numpy/u_mpc_track{i_track}.npy")
        save_numpy_array_to_file(sim_xdot, f"results/numpy/xdot_sim_track{i_track}.npy")
        save_numpy_array_to_file(kbm_xdot, f"results/numpy/xdot_kbm_track{i_track}.npy")
        save_numpy_array_to_file(res_xdot, f"results/numpy/xdot_res_track{i_track}.npy")
        save_numpy_array_to_file(np.array(l_cte), f"results/cte_track{i_track}.npy")

        l_df.append(
            {"rmse_sim_kbm":np.mean(rmse_sim_kbm),
             "rmse_sim_res_kbm":np.mean(rmse_sim_res_kbm),
             "cte": np.mean(np.array(l_cte)),
             "time_taken": len(l_cte)}
        )
        p.disconnect()
    df = pd.DataFrame(l_df)
    df.to_csv("results/results.csv")
if __name__ == "__main__":

    # Read in the debug tracks
    # For each track, run main with the track.
    main("models/full")
