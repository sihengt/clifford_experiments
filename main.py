import yaml
import os
import logging
import pybullet as p

from utils.planarRobotState import convert_planar_world_frame
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.Terrain import Terrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam

import numpy as np
import matplotlib.pyplot as plt

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC

from MPCPlotter import MPCPlotter
from MPCConfigLoader import MPCConfigLoader

import casadi as cs
import time

from simple_pid import PID


# TODO: YAMLify this.
SIM_DURATION = 200  # time steps ## NOT IN YAML

def main(data_dir):    
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }
    
    # Handling parameters for MPC
    mpc_config_loader = MPCConfigLoader(os.path.join(data_dir,'config/mpc.yaml'))
    # Converting parameters from list form to casadi parameters usable by MPC class
    mpc_params = mpc_config_loader.construct_casadi_params()
    
    plt.ion()
    tp = TrajProc()

    # PyBullet + robot setup
    # physicsClientID = p.connect(p.GUI)
    physicsClientID = p.connect(p.DIRECT)
    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    robot = CliffordRobot(robotParams)
    
    # Terrain
    terrain = Terrain(params['terrain'], physicsClientId=physicsClientID)
    # terrain = RandomRockyTerrain(params['terrain'], physicsClientId=physicsClientID)
    # terrain.generate()
    
    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=True,
        camFollowBot=True,
        stateProcessor=convert_planar_world_frame
    )
    
    sim.resetRobot(pose=((0,0),(0, 0, 0, np.sqrt(2)/2)))

    cs_kbm = csDSKBM(mpc_params)

    # Creating a sample trajectory to track
    xs = np.array([0, 125, 125, -125, -180, -75, 0]) * params['terrain']['mapScale']
    ys = np.array([0, 125, -125, -125, 75, 100, 0]) * params['terrain']['mapScale']
    track = tp.generate_path_from_wp(
        xs,
        ys,
        0.05
    )

    mp = MPCPlotter(track, SIM_DURATION)

    # [DEBUG] adds track into PyBullet
    # Colors start black and eventually turn white.
    colors = np.linspace(0, 1, xs.shape[0]-1)
    for i in range(xs.shape[0] - 1):
        p.addUserDebugLine(
            np.array([xs[i], ys[i], 0.1]),
            np.array([xs[i+1], ys[i+1], 0.1]),
            [colors[i], colors[i], colors[i]],
            5.0
        )
    
    opt_time = []

    # VARIABLES FOR TRACKING
    x_sim = np.zeros((mpc_params['model']['nStates'], SIM_DURATION))
    u_sim = np.zeros((mpc_params['model']['nActions'], SIM_DURATION - 1))

    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, 0.0, 0.0]).T

    # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
    u_bar_start = np.zeros((mpc_params['model']['nActions'], mpc_params['T']))
    u_bar_start[0, :] = mpc_params['maxAcc'] / 2
    u_bar_start[1, :] = 0.0
    u_bar_start[2, :] = 0.0

    l_a = []
    l_df = []
    l_dr = []
    l_state = []

    mpc = MPC(mpc_params, cs_kbm)
    
    # Create and configure logger
    logging.basicConfig(
        filename="logs/state.log",
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO)
    state_logger = logging.getLogger('States')
    
    for sim_time in range(SIM_DURATION - 1):
        iter_start = time.time()
    
        x_ref, nn_idx, l_ref_idx = tp.get_reference_trajectory(
            x_sim[:, sim_time],
            track,
            mpc_params['refVel'],
            0.05,
            mpc_params['T'],
            mpc_params['dt']
        )
        
        # TO DEBUG, plot points of nn_idx to T
        # [DEBUG] Trying to see if we reach the end of path tracking
        if track.shape[1] - nn_idx < 0.01 * 1115:
            print("we're close")
            print(nn_idx)

        if nn_idx == track.shape[1]:
            print("Reached.")
            break
        
        # You can add a warm-start to u_bar through u_bar_start.
        # In this loop we use the previous solved controls as a warmstart.
        X_mpc, U_mpc, x_ref = mpc.predict(x_sim[:, sim_time], x_ref, u_bar_start)
            
        a_mpc   = np.array(U_mpc[0, :]).flatten()
        d_f_mpc = np.array(U_mpc[1, :]).flatten()
        d_r_mpc = np.array(U_mpc[2, :]).flatten()
        u_bar = np.vstack(((X_mpc[2, 0] + a_mpc[0] * mpc_params['dt']) / 0.1, d_f_mpc[0], d_r_mpc[0]))

        current_state = X_mpc[:, 0]
        l_state.append(current_state)

        curr_a = U_mpc[0, 0]
        curr_df = U_mpc[1, 0]
        curr_dr = U_mpc[2, 0]
        
        # Wheel velocity = omega = v_com / wheel_radius.
        # TODO: I've hard-coded wheel radius to be 0.1 here.
        l_a.append([(X_mpc[2, 0] + curr_a * mpc_params['dt']) / 0.1])
        l_df.append(curr_df)
        l_dr.append(curr_dr)
        
        # Take first action
        u_sim[:, sim_time] = u_bar[:, 0]

        current_state = x_sim[:, sim_time]
        mp.plot_new_data(current_state, u_sim[:, sim_time], l_ref_idx, sim_time)
        
        # print("curr_v:{}\t dt:{}\t curr_a:{}".format(X_mpc[2,0], DT, curr_a))
        # print("Action to take:{}".format(torch.tensor(u_sim[:, sim_time])))
        print("nn_idx={}/{}".format(nn_idx, track.shape[1]))
        time_taken = time.time() - iter_start
        print("Time taken for solver:{}".format(time_taken))
        
        # Save actions solved by MPC as warm-start for next iteration.
        # u_bar_start = u_bar
        
        with open("log_actions.txt", "a") as f:
            f.write("Sim_time={}\t u_sim={}\n".format(sim_time, u_sim[:, sim_time]))
        
        # TODO: add check for termFlag.
        # Sends action to simulator to drive the robot by "simTimeStep" "numStepsPerControl" times.
        lastState, action, current_state, termFlag = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)

        # current_state: [x, y, heading, vel_x, vel_y, vel_theta]
        current_state = current_state.numpy()
        current_state[[2, 3]] = current_state[[3, 2]]
        print("curr_state:{}".format(current_state))
        x_sim[:, sim_time + 1] = current_state[:4]

        state_logger.info(x_sim[:, sim_time + 1])

        # Measure elapsed time for MPC
        opt_time.append(time_taken)
        
        plt.pause(0.0001)
    
    import pickle
    with open("opt_time.pkl", "wb") as pkl:
        pickle.dump(opt_time, pkl)
    with open("l_state.pkl", "wb") as pkl:
        pickle.dump(l_state, pkl)
    with open("l_a.pkl", "wb") as pkl:
        pickle.dump(l_a, pkl)
    with open("l_df.pkl", "wb") as pkl:
        pickle.dump(l_df, pkl)
    with open("l_dr.pkl", "wb") as pkl:
        pickle.dump(l_dr, pkl)


if __name__ == "__main__":
    main("nominal_dec_8")
