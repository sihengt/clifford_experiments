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
        camFollowBot=True,
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
    colors = np.linspace(0, 1, xs.shape[0]-1)
    for i in range(xs.shape[0] - 1):
        p.addUserDebugLine(
            np.array([xs[i], ys[i], 0.1]),
            np.array([xs[i+1], ys[i+1], 0.1]),
            [colors[i], colors[i], colors[i]],
            5.0
        )    
    return

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

def plan_mpc_step(current_state, u_bar_start, ref_track, tp, mpc, mpc_params, return_mpc_action=False):
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
    # TODO: add a check for whether we are close to the end. If so, either amend get_reference_trajectory OR the 
    # states or do something.
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
    X_mpc, U_mpc, x_ref = mpc.predict(current_state, x_ref, u_bar_start)

    # Converting action from MPC-space to sim space
    x_mpc = MpcState(*X_mpc[:, 0])
    u_mpc = MpcAction(*U_mpc[:, 0])
    u_sim_opt = MpcToSimAction(0.1, mpc_params['dt'], x_mpc.com_v, u_mpc)
    
    if return_mpc_action:
        return u_sim_opt, U_mpc[:, 0], l_ref_idx
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
    previous_state, _, current_state, termFlag = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)
    
    x_sim[:, sim_time + 1] = current_state[:4].numpy()

    # Handle velocity
    state_logger.info(x_sim[:, sim_time + 1])
    
    xy_dot      = current_state[4:6]
    a_dot       = torch.tensor([(current_state[3] - previous_state[3]) / dt], dtype=torch.float32)
    theta_dot   = current_state[6:]
    x_dot = torch.cat((xy_dot, a_dot, theta_dot))
    assert(x_dot.dtype == torch.float32)

    return x_dot

def do_stuff(x_sim):
    return