import yaml
import os
import pybullet as p
from utils.planarRobotState import convert_planar
from utils.planarRobotState import convert_planar_world_frame
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.RandomRockyTerrain import RandomRockyTerrain
from clifford_pybullet.Terrain import Terrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam

import torch
import numpy as np
from control.PurePursuit import PurePursuit
from plotting.PurePursuitPlot import PurePursuitPlot
import matplotlib.pyplot as plt

from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC
from TrajProc.scripts import *
import casadi as cs
import time

from simple_pid import PID

############
## FOR MPC #
############
N_STATES = 4
N_ACTIONS = 3
L = 0.684
l_f = 0.342
l_r = 0.342
T = 10
DT = 0.1 # "simTimeStep": 0.004166666666666 * number of steps per control: 24

MAX_SPEED = 10.0
MAX_STEER = np.radians(30)
MAX_D_ACC = 2.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 5.0
REF_VEL = 7.0

def main(data_dir):    
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }

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

    ## MPC 
    cs_kbm = csDSKBM(L, l_f, l_r, T, DT)

    # Creating a sample trajectory to track
    xs = np.array([0, 125, 125, -125, -180, -75, 0]) * params['terrain']['mapScale']
    ys = np.array([0, 125, -125, -125, 75, 100, 0]) * params['terrain']['mapScale']
    track = generate_path_from_wp(
        xs,
        ys,
        0.05
    )

    # [DEBUG] adds track into PyBullet
    # Colors start black and eventually turn white.
    colors = np.linspace(0, 1, xs.shape[0]-1)
    for i in range(xs.shape[0]-1):
        p.addUserDebugLine(np.array([xs[i], ys[i], 0.1]), np.array([xs[i+1], ys[i+1], 0.1]), [colors[i], colors[i], colors[i]], 5.0)
    
    sim_duration = 2000  # time steps
    opt_time = []

    # VARIABLES FOR TRACKING
    x_sim = np.zeros((N_STATES, sim_duration))
    u_sim = np.zeros((N_ACTIONS, sim_duration - 1))

    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, 0.0, np.pi/2]).T

    # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
    u_bar_start = np.zeros((N_ACTIONS, T))
    u_bar_start[0, :] = MAX_ACC / 2
    u_bar_start[1, :] = 0.0
    u_bar_start[2, :] = 0.0

    l_a = []
    l_df = []
    l_dr = []
    l_state = []

    MPC_PARAMS = {
        "n_x": N_STATES,
        "m_u": N_ACTIONS,
        "T": T,
        "dt": DT,
        "X_lb": cs.DM([-cs.inf, -cs.inf, -cs.inf, -cs.inf],),
        "X_ub": cs.DM([cs.inf, cs.inf, cs.inf, cs.inf],),
        "U_lb": cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER]), 
        "U_ub": cs.DM([MAX_ACC, MAX_STEER, MAX_STEER]),
        "dU_b": cs.DM([MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]),
        "Q": cs.DM(np.diag([20, 20, 10, 0])),
        "Qf": cs.DM(np.diag([30, 30, 30, 0])),
        "R": cs.DM(np.diag([10, 10, 10])),
        "R_": cs.DM(np.diag([10, 10, 10]))
    }

    mpc = MPC(MPC_PARAMS, cs_kbm)
    # TODO: add logging.
    
    for sim_time in range(sim_duration - 1):
        iter_start = time.time()
    
        # u_bar_start is either a random action OR warm-started with previous iteration's solution.
        X_mpc, U_mpc, x_ref = mpc.predict(x_sim[:, sim_time], u_bar_start, REF_VEL, track)
            
        a_mpc   = np.array(U_mpc[0, :]).flatten()
        d_f_mpc = np.array(U_mpc[1, :]).flatten()
        d_r_mpc = np.array(U_mpc[2, :]).flatten()
        u_bar = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

        # [DEBUG] adding lines to see what reference trajectory we're tracking
        for i in range(x_ref.shape[1] - 1):
            p.addUserDebugLine(np.array([x_ref[0, i], x_ref[0, i], 0.2]), np.array([x_ref[0, i+1], x_ref[1, i+1], 0.2]), [1, 0, 0], 3.0, 1.0)

        current_state = X_mpc[:, 0]
        l_state.append(current_state)

        a_mpc   = np.array(U_mpc[0, :]).flatten()
        df_mpc  = np.array(U_mpc[1, :]).flatten()
        dr_mpc  = np.array(U_mpc[2, :]).flatten()

        l_a.append(a_mpc[0])
        l_df.append(df_mpc[0])
        l_dr.append(dr_mpc[0])

        u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
        
        # Take first action
        u_sim[:, sim_time] = u_bar[:, 0]
        print("Action to take:{}".format(torch.tensor(u_sim[:, sim_time])))

        # Save second action as warmstart for next iteration.
        u_bar_start = u_bar[:, 1]
        
        with open("log_actions.txt", "a") as f:
            f.write("Sim_time={}\t u_sim={}\n".format(sim_time, u_sim[:, sim_time]))
        
        # Sends action to simulator to drive the robot by "simTimeStep" "numStepsPerControl" times.
        lastState, action, current_state, termFlag = sim.controlLoopStep(torch.tensor(u_sim[:, sim_time]))
        
        # current_state: [x, y, heading, vel_x, vel_y, vel_theta]
        current_state = current_state.numpy()
        current_state[[2, 3]] = current_state[[3, 2]]
        x_sim[:, sim_time + 1] = current_state[:4]

        # Measure elapsed time for MPC
        opt_time.append(time.time() - iter_start)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(track[0, :], track[1, :], "b")
    ax1.scatter(x_sim[0, :], x_sim[1, :], s=0.5, color='red', zorder=1)

    ax1.plot(x_sim[0, :], x_sim[1, :], color='green', zorder=-1)
    # plt.plot(x_traj[0, :], x_traj[1, :])
    ax1.axis("equal")
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")

    ax2 = fig.add_subplot(gs[1, 0]) 
    x = np.arange(len(l_a))
    ax2.plot(x, l_a)
    ax2.set_ylabel("acceleration command")
    ax2.set_xlabel("time")

    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(l_df))
    ax3.plot(x, l_df)
    ax3.set_ylabel("Front steering")
    ax3.set_xlabel("time")

    ax4 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(l_dr))
    ax4.plot(x, l_dr)
    ax4.set_ylabel("Rear steering")
    ax4.set_xlabel("time")
    plt.show()
    
    fig.savefig("output.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main("nominal_dec_8")
