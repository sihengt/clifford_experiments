import numpy as np
import logging
import pybullet as p
from TrajProc.controls.translator import MpcAction, MpcState, SimState, MpcToSimAction
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.Terrain import Terrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam
from utils.planarRobotState import convert_planar_world_frame, convert_planar_world_frame_with_vel
from MPCConfigLoader import MPCConfigLoader

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC

import matplotlib.pyplot as plt

import yaml
import os

import torch

def init_pybullet_sim(data_dir):
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        'network': yaml.safe_load(open(os.path.join(data_dir,'config/network.yaml'),'r')),
        }

    # Converting parameters from list form to casadi parameters usable by MPC class
    mpc_config_loader = MPCConfigLoader(os.path.join(data_dir,'config/mpc.yaml'))
    mpc_params = mpc_config_loader.construct_casadi_params()

    # YOU'RE LOOKING FOR THIS!
    physicsClientID = p.connect(p.DIRECT)
    # physicsClientID = p.connect(p.GUI)

    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    robot = CliffordRobot(robotParams)

    # Terrain
    terrain = Terrain(params['terrain'], physicsClientId=physicsClientID)

    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=True, # TODO: try this with False
        camFollowBot=False,
        stateProcessor=convert_planar_world_frame_with_vel
    )

    return params, mpc_params, sim

def create_debug_track(tp, terrain_params, debug=True):
    # Creating a sample trajectory to track
    xs = np.array([0, 125, 125, -125, -180, -75, 0]) * terrain_params['mapScale']
    ys = np.array([0, 125, -125, -125, 75, 100, 0])  * terrain_params['mapScale']

    track = tp.generate_path_from_wp(
        xs,
        ys,
        0.05
    )

    if debug:
        draw_debug_track(xs, ys)

    return track

def create_random_track(tp, terrain_params, n_vertices, debug=True):
    x_max = terrain_params['mapWidth'] / 2
    x_min = -x_max
    y_max = terrain_params['mapLength'] / 2
    y_min = -y_max
    
    xs = np.concatenate( ([0], np.random.randint(x_min, x_max, n_vertices - 1)) ) * terrain_params['mapScale']
    ys = np.concatenate( ([0], np.random.randint(y_min, y_max, n_vertices - 1)) ) * terrain_params['mapScale']

    track = tp.generate_path_from_wp(
        xs,
        ys,
        0.05
    )

    if debug:
        draw_debug_track(xs, ys)    

    return track

def draw_debug_track(xs, ys):
    # [DEBUG] adds track into PyBullet
    # Colors start black and eventually turn white.
    cmap = plt.get_cmap("plasma")
    n_seg   = xs.shape[0] - 1
    colour_table = cmap(np.linspace(0, 1, n_seg))[:, :3]   # keep only RGB

    colors = np.linspace(0, 1, xs.shape[0]-1)
    for i in range(xs.shape[0] - 1):
        p.addUserDebugLine(
            np.array([xs[i], ys[i], 0.1]),
            np.array([xs[i+1], ys[i+1], 0.1]),
            colour_table[i].tolist(),
            5.0
        )    
    return

def make_track_segment(p0, p1, radius=0.02, rgba=(1, 0, 0, 1)):
    """
    Draw one capsule‑segment that runs from p0 to p1 and shows up
    in both the GUI and getCameraImage() renders.

    p0, p1 : 3‑element array‑likes (x, y, z)
    radius : capsule radius, metres
    rgba   : colour including alpha
    """
    p0, p1   = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    vec      = p1 - p0
    length   = np.linalg.norm(vec)
    if length < 1e-6:           # ignore zero‑length segments
        return

    centre   = (p0 + p1) / 2.0
    axis     = vec / length     # unit vector of the segment

    # --- quaternion that rotates [0,0,1] onto 'axis' -------------------------
    z_axis   = np.array([0.0, 0.0, 1.0])
    dot      = np.clip(np.dot(z_axis, axis), -1.0, 1.0)

    if dot > 0.999999:                          # already parallel (+z)
        quat = [0, 0, 0, 1]
    elif dot < -0.999999:                       # opposite direction (–z)
        quat = p.getQuaternionFromEuler([np.pi, 0, 0])
    else:
        rot_axis = np.cross(z_axis, axis)
        rot_axis /= np.linalg.norm(rot_axis)
        angle    = np.arccos(dot)
        quat     = p.getQuaternionFromAxisAngle(rot_axis, angle)
    # -------------------------------------------------------------------------

    vis_id = p.createVisualShape(p.GEOM_CAPSULE,
                                 radius      = radius,
                                 length      = length,
                                 rgbaColor   = rgba)

    p.createMultiBody(baseMass                  = 0,         # static
                      baseCollisionShapeIndex   = -1,        # ***no collisions***
                      baseVisualShapeIndex      = vis_id,
                      basePosition              = centre.tolist(),
                      baseOrientation           = quat)

def init_logger():
    # Create and configure logger
    logging.basicConfig(
        filename="logs/state.log",
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO)
    state_logger = logging.getLogger('States')
    return state_logger

def init_u_bar(mpc_params):
    u_bar_start = np.zeros((mpc_params['model']['nActions'], mpc_params['T']))
    u_bar_start[0, :] = mpc_params['maxAcc'] / 2
    u_bar_start[1, :] = 0.0
    u_bar_start[2, :] = 0.0

    return u_bar_start

def plan_mpc_step(
        current_state,
        u_bar_start,
        ref_track,
        tp,
        mpc,
        mpc_params,
        n_iters=1,
        return_mpc_action=False,
        return_all_mpc_actions=False,
    ):
    """
    Params:
        current_state:  state of the following form [x, y, velocity, yaw]
        u_bar_state:    reference actions that repeat T times [[acc, left_steering, right_steering], ...]
        tp:             Trajectory Processor class that contains functions for processing trajectories
        mpc:            MPC class
        mpc_params:     parameters used in optimization

    Returns:
        u_sim_opt: optimized actions for the simulator [wheel_velocity, front steering, rear steering]
        l_ref_idx: a list of reference indices used during MPC.
    """
    x_ref, nn_idx, l_ref_idx = tp.get_reference_trajectory(
        current_state,
        ref_track,
        mpc_params['refVel'],
        0.05,
        mpc_params['T'],
        mpc_params['dt']
    )

    # You can add a warm-start to u_bar through u_bar_start.
    # In this loop we use the previous solved controls as a warmstart.
    X_mpc, U_mpc, x_ref = mpc.predict(current_state, x_ref, u_bar_start, n_iters=n_iters)

    # Converting action from MPC-space to sim space
    x_mpc = MpcState(*X_mpc[:, 0])
    u_mpc = MpcAction(*U_mpc[:, 0])
    u_sim_opt = MpcToSimAction(0.1, mpc_params['dt'], x_mpc.com_v, u_mpc)
    
    if return_mpc_action:
        if return_all_mpc_actions:
            return u_sim_opt, U_mpc, l_ref_idx
        else:
            return u_sim_opt, u_mpc, l_ref_idx
    else:
        return u_sim_opt, l_ref_idx

def mpc_worker(state_q, result_q, stop_evt, mpc_params, ref_track):
    """
    Params:
        current_state:  state of the following form [x, y, velocity, yaw]
        u_bar_state:    reference actions that repeat T times [[acc, left_steering, right_steering], ...]
        tp:             Trajectory Processor class that contains functions for processing trajectories
        mpc:            MPC class
        mpc_params:     parameters used in optimization

    Returns:

    """
    tp = TrajProc()
    cs_kbm = csDSKBM(mpc_params)
    mpc = MPC(mpc_params, cs_kbm)

    u_bar_start = init_u_bar(mpc_params)

    while not stop_evt.is_set():
        current_state = state_q.get()
        if current_state is None:
            break

        u_sim_opt, l_ref_idx = plan_mpc_step(
            current_state, u_bar_start, ref_track, tp, mpc, mpc_params
        )

        result_q.put((u_sim_opt, l_ref_idx))


def step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger, dt):
    # previous_state, _, current_state, termFlag, previous_state_body = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)
    previous_state, _, current_state, termFlag = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)

    x_sim[:, sim_time + 1] = current_state[:4].numpy()

    # Handle velocity
    state_logger.info(x_sim[:, sim_time + 1])
    
    xy_dot      = previous_state[4:6] # 0 1 2 3 (x, y, v, theta) # 4 5 (xdot, ydot) 6 (yaw)
    v_dot       = torch.tensor([(current_state[2] - previous_state[2]) / dt], dtype=torch.float32)
    theta_dot   = previous_state[6:]
    # theta_dot   = previous_state_body[6:]
    x_dot = torch.cat((xy_dot, v_dot, theta_dot))

    assert(x_dot.dtype == torch.float32)

    return x_dot

def compare_state_dot(l_pybullet_xdot, l_kbm_xdot):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].set_title("Comparison of $\dot{x}$: PyBullet, KBM")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    axs[0, 0].plot(np.arange(l_pybullet_xdot.shape[0]), l_pybullet_xdot[:, 0], label='PyBullet $\dot{x}$', color='r')
    axs[0, 0].plot(np.arange(l_pybullet_xdot.shape[0]), l_kbm_xdot[:, 0], label='KBM $\dot{x}$', color='c')
    axs[0, 0].legend()

    axs[0, 1].set_title("Comparison of $\dot{y}$: PyBullet, KBM")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    axs[0, 1].plot(np.arange(l_pybullet_xdot.shape[0]), l_pybullet_xdot[:, 1], label='PyBullet $\dot{y}$', color='r')
    axs[0, 1].plot(np.arange(l_pybullet_xdot.shape[0]), l_kbm_xdot[:, 1], label='KBM $\dot{y}$', color='c')
    axs[0, 1].legend()
    
    axs[1, 0].set_title("Comparison of $a$: PyBullet, KBM")
    axs[1, 0].set_ylabel(r"Acceleration ($m/s^2$)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].plot(np.arange(l_pybullet_xdot.shape[0]), l_pybullet_xdot[:, 2], label='PyBullet $a$', color='r')
    axs[1, 0].plot(np.arange(l_pybullet_xdot.shape[0]), l_kbm_xdot[:, 2], label='KBM $a$', color='c')
    axs[1, 0].legend()

    axs[1, 1].set_title(r"Comparison of $\dot{\theta}$: PyBullet, KBM")
    axs[1, 1].set_ylabel("Angular Velocity (rad/s)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].plot(np.arange(l_pybullet_xdot.shape[0]), l_pybullet_xdot[:, 3], label=r'PyBullet $\dot{\theta}$', color='r')
    axs[1, 1].plot(np.arange(l_pybullet_xdot.shape[0]), l_kbm_xdot[:, 3], label=r'KBM $\dot{\theta}$', color='c')
    axs[1, 1].legend()
    
    plt.savefig("DEBUG_DATA_COLLECTION.png")
    # plt.show(block=True)

def compare_state_dot_with_res(l_pybullet_xdot, l_kbm_xdot, l_res, fig_fp="results/DEBUGResNetwork.png"):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout()  # Automatically adjusts spacing between subplots
    
    assert(l_pybullet_xdot.shape[0] == l_kbm_xdot.shape[0])
    assert(l_kbm_xdot.shape[0] == l_res.shape[0])
    axs[0, 0].set_title("Comparison of $\dot{x}$: PyBullet, KBM, Corrected KBM")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    x_ts = np.arange(l_pybullet_xdot.shape[0]) * 0.1
    axs[0, 0].plot(x_ts, l_pybullet_xdot[:, 0], label='PyBullet $\dot{x}$', color='r', linestyle='dashed')
    axs[0, 0].plot(x_ts, l_kbm_xdot[:, 0], label='KBM $\dot{x}$', color='c', linestyle='dashdot')
    axs[0, 0].plot(x_ts, l_kbm_xdot[:, 0] + l_res[:, 0], label='Corrected KBM $\dot{x}$', color='m', linestyle='dotted')
    axs[0, 0].legend()

    axs[0, 1].set_title("Comparison of $\dot{y}$: PyBullet, KBM, Corrected KBM")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].plot(x_ts, l_pybullet_xdot[:, 1], label='PyBullet $\dot{y}$', color='r', linestyle='dashed')
    axs[0, 1].plot(x_ts, l_kbm_xdot[:, 1], label='KBM $\dot{y}$', color='c', linestyle='dashdot')
    axs[0, 1].plot(x_ts, l_kbm_xdot[:, 1] + l_res[:, 1], label='Corrected KBM $\dot{y}$', color='m', linestyle='dotted')
    axs[0, 1].legend()
    
    axs[1, 0].set_title("Comparison of $a$: PyBullet, KBM, Corrected KBM")
    axs[1, 0].set_ylabel(r"Acceleration ($m/s^2$)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].plot(x_ts, l_pybullet_xdot[:, 2], label='PyBullet $a$', color='r', linestyle='dashed')
    axs[1, 0].plot(x_ts, l_kbm_xdot[:, 2], label='KBM $a$', color='c', linestyle='dashdot')
    axs[1, 0].plot(x_ts, l_kbm_xdot[:, 2] + l_res[:, 2], label='Corrected KBM $a$', color='m', linestyle='dotted')
    axs[1, 0].legend()

    axs[1, 1].set_title(r"Comparison of $\dot{\theta}$: PyBullet, KBM, Corrected KBM")
    axs[1, 1].set_ylabel("Angular Velocity (rad/s)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].plot(x_ts, l_pybullet_xdot[:, 3], label=r'PyBullet $\dot{\theta}$', color='r', linestyle='dashed')
    axs[1, 1].plot(x_ts, l_kbm_xdot[:, 3], label=r'KBM $\dot{\theta}$', color='c', linestyle='dashdot')
    axs[1, 1].plot(x_ts, l_kbm_xdot[:, 3] + l_res[:, 3], label=r'Corrected KBM $\dot{\theta}$', color='m', linestyle='dotted')
    axs[1, 1].legend()

    plt.savefig(fig_fp, dpi=300)
    plt.close(fig)

def compare_state_dot_with_full(l_pybullet_xdot, l_kbm_xdot, l_res, fig_fp="results/DEBUGResNetwork.png"):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout()  # Automatically adjusts spacing between subplots
    
    assert(l_pybullet_xdot.shape[0] == l_kbm_xdot.shape[0])
    assert(l_kbm_xdot.shape[0] == l_res.shape[0])
    axs[0, 0].set_title("Comparison of $\dot{x}$: PyBullet, KBM, Predicted")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    x_ts = np.arange(l_pybullet_xdot.shape[0]) * 0.1
    axs[0, 0].plot(x_ts, l_pybullet_xdot[:, 0], label='PyBullet $\dot{x}$', color='r', linestyle='dashed')
    axs[0, 0].plot(x_ts, l_kbm_xdot[:, 0], label='KBM $\dot{x}$', color='c', linestyle='dashdot')
    axs[0, 0].plot(x_ts, l_res[:, 0], label='Predicted $\dot{x}$', color='m', linestyle='dotted')
    axs[0, 0].legend()

    axs[0, 1].set_title("Comparison of $\dot{y}$: PyBullet, KBM, Predicted")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].plot(x_ts, l_pybullet_xdot[:, 1], label='PyBullet $\dot{y}$', color='r', linestyle='dashed')
    axs[0, 1].plot(x_ts, l_kbm_xdot[:, 1], label='KBM $\dot{y}$', color='c', linestyle='dashdot')
    axs[0, 1].plot(x_ts, l_res[:, 1], label='Predicted $\dot{y}$', color='m', linestyle='dotted')
    axs[0, 1].legend()
    
    axs[1, 0].set_title("Comparison of $a$: PyBullet, KBM, Predicted")
    axs[1, 0].set_ylabel(r"Acceleration ($m/s^2$)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].plot(x_ts, l_pybullet_xdot[:, 2], label='PyBullet $a$', color='r', linestyle='dashed')
    axs[1, 0].plot(x_ts, l_kbm_xdot[:, 2], label='KBM $a$', color='c', linestyle='dashdot')
    axs[1, 0].plot(x_ts, l_res[:, 2], label='Predicted $a$', color='m', linestyle='dotted')
    axs[1, 0].legend()

    axs[1, 1].set_title(r"Comparison of $\dot{\theta}$: PyBullet, KBM, Predicted")
    axs[1, 1].set_ylabel("Angular Velocity (rad/s)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].plot(x_ts, l_pybullet_xdot[:, 3], label=r'PyBullet $\dot{\theta}$', color='r', linestyle='dashed')
    axs[1, 1].plot(x_ts, l_kbm_xdot[:, 3], label=r'KBM $\dot{\theta}$', color='c', linestyle='dashdot')
    axs[1, 1].plot(x_ts, l_res[:, 3], label=r'Predicted $\dot{\theta}$', color='m', linestyle='dotted')
    axs[1, 1].legend()

    plt.savefig(fig_fp, dpi=300)
    plt.close(fig)


# TODO: analyze the differences between pybullet's xdot (ground truth), KBM and KBM + residual.
# Save the difference for each time step into a csv?

def rmse(arr_1, arr_2):
    assert arr_1.shape == arr_2.shape
    return np.sqrt(np.mean((arr_1 - arr_2)**2, axis=0))

def analyze_state_dot_with_res(l_sim_xdot, l_kbm_xdot, l_res):
    rmse_sim_kbm = rmse(l_sim_xdot, l_kbm_xdot)
    rmse_sim_res_kbm = rmse(l_sim_xdot, l_kbm_xdot + l_res)
    return rmse_sim_kbm, rmse_sim_res_kbm

def analyze_state_dot_with_full(l_sim_xdot, l_kbm_xdot, l_res):
    rmse_sim_kbm = rmse(l_sim_xdot, l_kbm_xdot)
    rmse_sim_res_kbm = rmse(l_sim_xdot, l_res)
    return rmse_sim_kbm, rmse_sim_res_kbm

def save_numpy_array_to_file(arr, fn):
    with open(fn, "wb") as f:
        np.save(f, arr)