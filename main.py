import yaml
import os
import pybullet as p
from utils.planarRobotState import convert_planar
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

# from TrajProc.models.DKBM_casadi import csDSKBM
from TrajProc.models.DKBM import csDSKBM
from TrajProc.controls.MPC import MPC
from TrajProc.scripts import *
import casadi as cs
import time

from simple_pid import PID

############
## FOR MPC #
############
N_STATES = 3
N_ACTIONS = 3
L = 0.684
l_f = 0.342
l_r = 0.342
T = 20
N = 50
DT = 0.01 # TODO: think about this in simulator context - we might have to change it?

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_D_ACC = 0.1
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 1.5
REF_VEL = 1.0

# TODO: are we able to compute everything in quaternions, and only use Euler angles for plotting?

# def body_total_com(body_uid):
#     """
#     Returns (com_world, total_mass) for a multibody, where com_world is an (x,y,z)
#     tuple in world coordinates.
#     """
#     n_links = p.getNumJoints(body_uid)
#     total_mass   = 0.0
#     com_sum      = np.zeros(3)

#     # ---------------- base (-1) and every link ----------------
#     for link in range(-1, n_links):
#         mass, _, _, local_inertial_pos, local_inertial_orn, *_ = \
#             p.getDynamicsInfo(body_uid, link)      # has mass & local CoM :contentReference[oaicite:0]{index=0}
#         if mass == 0:                 # static or visual‑only part
#             continue

#         if link == -1:                # base link
#             link_world_pos, link_world_orn = p.getBasePositionAndOrientation(body_uid)
#         else:
#             link_state                = p.getLinkState(body_uid, link, computeForwardKinematics=1)
#             link_world_pos, link_world_orn = link_state[0], link_state[1]

#         # transform the local inertial offset into world frame
#         com_i, _ = p.multiplyTransforms(link_world_pos, link_world_orn,
#                                          local_inertial_pos, local_inertial_orn)
#         com_sum      += mass * np.array(com_i)
#         total_mass   += mass

#     return tuple(com_sum / total_mass), total_mass

# def drop_debug_marker(world_pos, colour=[0,1,0], life=0):
#     sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=colour+[1])
#     p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere,
#                       basePosition=world_pos)

def main(data_dir):
    plt.ion()
    MAX_TIME_STEPS = 200
    N_TRAJECTORIES_TO_COLLECT = 10
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }

    # PyBullet + robot setup
    physicsClientID = p.connect(p.GUI)
    #physicsClientID = p.connect(p.DIRECT)
    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    robot = CliffordRobot(robotParams)
    
    # Terrain
    # terrain = RandomRockyTerrain(params['terrain'], physicsClientId=physicsClientID)
    # terrain.generate()
    terrain = Terrain(params['terrain'], physicsClientId=physicsClientID)


    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=True,
        camFollowBot=True,
        stateProcessor=convert_planar
    )
    
    sim.resetRobot(pose=((0,0),(0, 0, np.sqrt(2)/2, np.sqrt(2)/2)))

    ###############
    ## MPC THINGS #
    ###############
    cs_kbm = csDSKBM(N_STATES, N_ACTIONS, L, l_f, l_r, T, N)

    ## Creating a sample trajectory to track
    ## Eventually we convex hull some random points we sample within the map.
    xs = np.array([0, 125, 125, -125, -180, -75, 0]) * params['terrain']['mapScale']
    ys = np.array([0, 125, -125, -125, 75, 100, 0]) * params['terrain']['mapScale']
    track = generate_path_from_wp(
        xs,
        ys,
        0.05
    )

    for i in range(xs.shape[0]-1):
        p.addUserDebugLine(np.array([xs[i], ys[i], 0.1]), np.array([xs[i+1], ys[i+1], 0.1]), [0, 1, 0], 5.0)
    
    sim_duration = 2000  # time steps
    opt_time = []

    # VARIABLES FOR TRACKING
    x_sim = np.zeros((N_STATES, sim_duration))
    u_sim = np.zeros((N_ACTIONS, sim_duration - 1))

    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, np.radians(0)]).T

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
        "X_lb": cs.DM([-cs.inf, -cs.inf, -cs.inf],),
        "X_ub": cs.DM([cs.inf, cs.inf, cs.inf],),
        "U_lb": cs.DM([0, -MAX_STEER, -MAX_STEER]), 
        "U_ub": cs.DM([MAX_SPEED, MAX_STEER, MAX_STEER]),
        "dU_b": cs.DM([MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]),
        "Q": cs.DM(np.diag([20, 20, 0])),
        "Qf": cs.DM(np.diag([30, 30, 0])),
        "R": cs.DM(np.diag([10, 10, 10])),
        "R_": cs.DM(np.diag([10, 10, 10]))
    }

    mpc = MPC(MPC_PARAMS, cs_kbm)

    for sim_time in range(sim_duration - 1):
        iter_start = time.time()
        
        # Re-optimizing up to five times with the previous solved controls as a warm-start to try to get a better solution.
        for i_iter in range(1):
            # For the first iteration we use dummy controls as u_bar
            if i_iter == 0:
                u_bar = u_bar_start
            
            X_mpc, U_mpc = mpc.predict(x_sim[:, sim_time], u_bar, track)
            
            a_mpc   = np.array(U_mpc[0, :]).flatten()
            d_f_mpc = np.array(U_mpc[1, :]).flatten()
            d_r_mpc = np.array(U_mpc[2, :]).flatten()
            u_bar_new = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

            delta_u = np.sum(np.sum(np.abs(u_bar_new - u_bar), axis=0), axis=0)
            if delta_u < 0.05:
                break
                
            u_bar = u_bar_new
        
        current_state = X_mpc[:, 0]
        l_state.append(current_state)
        x_mpc       = np.array(X_mpc[0, :]).flatten()
        y_mpc       = np.array(X_mpc[1, :]).flatten()
        # v_mpc       = np.array(X_mpc[2, :]).flatten()
        theta_mpc   = np.array(X_mpc[2, :]).flatten()

        a_mpc   = np.array(U_mpc[0, :]).flatten()
        df_mpc  = np.array(U_mpc[1, :]).flatten()
        dr_mpc  = np.array(U_mpc[2, :]).flatten()

        l_a.append(a_mpc[0])
        l_df.append(df_mpc[0])
        l_dr.append(dr_mpc[0])

        u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
        
        # Take first action
        u_sim[:, sim_time] = u_bar[:, 0]
        print("Action to take:{}".format(torch.tensor(u_bar[:, 0])))
        # current_state = (x, y, theta, vel_x, vel_y, vel_theta)
        lastState, action, current_state, termFlag = sim.controlLoopStep(torch.tensor(u_bar[:, 0]))
        current_state = current_state.numpy()

        x_sim[:, sim_time + 1] = np.array([current_state[0], current_state[1], current_state[2]])

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

    breakpoint()


    # for i_traj in range(N_TRAJECTORIES_TO_COLLECT):
    #     # Reset robot pose
    #     sim.resetRobot(pose=((0,0),(0, 0, np.sqrt(2)/2, np.sqrt(2)/2)))

    #     # TODO: move into data collection.
    #     for t in range(MAX_TIME_STEPS):
    #         # [throttle, front_angle, rear_angle]
    #         action = pure_pursuit.track_traj(current_state, ref_state)
    #         lastState, action, current_state, termFlag = sim.controlLoopStep(action.cpu().squeeze())

if __name__ == "__main__":
    main("nominal_dec_8")
