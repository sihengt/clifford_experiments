"""Shared helpers for the Clifford sim, the MPC loop and dynamics diagnostics."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
import yaml

from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.SimController import SimController
from clifford_pybullet.Terrain import Terrain
from clifford_pybullet.utils.genParam import genParam
from TrajProc.controls.translator import MpcAction, MpcState, MpcToSimAction
from utils.MPCConfigLoader import MPCConfigLoader
from utils.planarRobotState import convert_planar_world_frame_with_vel

PATH_STEP = 0.05     # Spacing (m) between points of a reference path
WHEEL_RADIUS = 0.1   # Clifford wheel radius (m), matches the SDF. Used to turn speed into wheel speed.

CONFIG_NAMES = ('controls', 'robotRange', 'sim', 'terrain', 'train', 'network')


def load_params(data_dir):
    """
    Loads <data_dir>/config/*.yaml.

    Returns:
        params: dict with one entry per config file, plus 'dataDir'
        mpc_params: mpc.yaml converted to CasADi types (see MPCConfigLoader)
    """
    params = {'dataDir': data_dir}
    for name in CONFIG_NAMES:
        with open(os.path.join(data_dir, 'config', f'{name}.yaml')) as f:
            params[name] = yaml.safe_load(f)
    mpc_params = MPCConfigLoader(os.path.join(data_dir, 'config', 'mpc.yaml')).construct_casadi_params()
    return params, mpc_params


def init_pybullet_sim(data_dir, gui=False, recorder=None, realtime=None):
    """
    Loads all configs from <data_dir>/config and builds the robot, terrain and SimController.

    Params:
        data_dir: model directory containing config/*.yaml (and adm.pt for learned models).
        gui: if True, opens the PyBullet GUI; otherwise runs headless (DIRECT).
        recorder: optional object with grab(), called after every physics step (see CameraRecorder).
        realtime: if True, the sim sleeps so it runs at wall-clock speed. Only changes speed, not
            results. Defaults to the value of gui.
    """
    params, mpc_params = load_params(data_dir)
    if realtime is None:
        realtime = gui

    physicsClientID = p.connect(p.GUI if gui else p.DIRECT)

    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    robot = CliffordRobot(robotParams)
    terrain = Terrain(params['terrain'], physicsClientId=physicsClientID)

    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=realtime,
        camFollowBot=False,
        stateProcessor=convert_planar_world_frame_with_vel,
        recorder=recorder,
    )

    return params, mpc_params, sim


def create_random_track(tp, terrain_params, n_vertices, debug=True):
    x_max = terrain_params['mapWidth'] / 2
    x_min = -x_max
    y_max = terrain_params['mapLength'] / 2
    y_min = -y_max

    xs = np.concatenate(([0], np.random.randint(x_min, x_max, n_vertices - 1))) * terrain_params['mapScale']
    ys = np.concatenate(([0], np.random.randint(y_min, y_max, n_vertices - 1))) * terrain_params['mapScale']

    track = tp.generate_path_from_wp(xs, ys, PATH_STEP)

    if debug:
        draw_debug_track(xs, ys)

    return track


def draw_debug_track(xs, ys):
    """Draws the waypoint polygon as debug lines in the PyBullet GUI (colour goes dark to light)."""
    colour_table = plt.get_cmap("plasma")(np.linspace(0, 1, xs.shape[0] - 1))[:, :3]
    for i in range(xs.shape[0] - 1):
        p.addUserDebugLine(
            np.array([xs[i], ys[i], 0.1]),
            np.array([xs[i + 1], ys[i + 1], 0.1]),
            colour_table[i].tolist(),
            5.0
        )


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


def draw_track(xs, ys, z=0.10, radius=0.015):
    """Draws the waypoint polygon as coloured capsules (visible in the GUI and in camera renders)."""
    colour_table = plt.get_cmap("plasma")(np.linspace(0, 1, xs.shape[0] - 1))[:, :3]
    for i in range(len(xs) - 1):
        make_track_segment([xs[i], ys[i], z], [xs[i + 1], ys[i + 1], z],
                           radius=radius, rgba=colour_table[i].tolist() + [1, ])


def init_u_bar(mpc_params):
    u_bar_start = np.zeros((mpc_params['model']['nActions'], mpc_params['T']))
    u_bar_start[0, :] = mpc_params['maxAcc'] / 2
    u_bar_start[1, :] = 0.0
    u_bar_start[2, :] = 0.0

    return u_bar_start


def plan_mpc_step(current_state, u_bar_start, ref_track, tp, mpc, mpc_params, n_iters=1):
    """
    Builds the reference from the nearest point on the track and solves the MPC.

    Params:
        current_state:  state of the following form [x, y, velocity, yaw]
        u_bar_start:    (3, T) warm-start actions [acc, front steering, rear steering]
        ref_track:      (3, N) path [x; y; yaw] with PATH_STEP spacing
        tp:             TrajProc instance (keeps the last nearest index between calls)
        mpc:            MPC instance
        mpc_params:     parameters used in optimization
        n_iters:        linearize-and-solve iterations inside the MPC

    Returns:
        u_sim_opt: first MPC action converted for the simulator [wheel speed, front steering, rear steering]
        U_mpc: (3, T) all MPC actions [acc, front steering, rear steering]; U_mpc[:, 0] is the one applied
        l_ref_idx: path indices used for the reference; l_ref_idx[0] is the nearest index
    """
    x_ref, _, l_ref_idx = tp.get_reference_trajectory(
        current_state,
        ref_track,
        mpc_params['refVel'],
        PATH_STEP,
        mpc_params['T'],
        mpc_params['dt']
    )

    X_mpc, U_mpc, x_ref = mpc.predict(current_state, x_ref, u_bar_start, n_iters=n_iters)

    # Converting action from MPC-space to sim space
    x_mpc = MpcState(*X_mpc[:, 0])
    u_mpc = MpcAction(*U_mpc[:, 0])
    u_sim_opt = MpcToSimAction(WHEEL_RADIUS, mpc_params['dt'], x_mpc.com_v, u_mpc)

    return u_sim_opt, U_mpc, l_ref_idx


def state_derivative(previous_state, current_state, dt):
    """
    Target x_dot for one control step, as used for training data and evaluation.

    Params:
        previous_state, current_state: planar states [x, y, v, yaw, x_dot, y_dot, yaw_dot]
            (from convert_planar_world_frame_with_vel) at the start and end of the step.

    Returns:
        [x_dot, y_dot, v_dot, yaw_dot]: x_dot, y_dot and yaw_dot are the sim velocities at the start of
        the step; v_dot is the finite difference of speed over the step.
    """
    xy_dot = previous_state[4:6]
    v_dot = torch.tensor([(current_state[2] - previous_state[2]) / dt], dtype=previous_state.dtype)
    theta_dot = previous_state[6:]
    return torch.cat((xy_dot, v_dot, theta_dot))


def step_sim_and_log(sim, sim_time, x_sim, u_sim, state_logger, dt):
    """
    Applies u_sim[:, sim_time] for one control step, writes the new state into x_sim[:, sim_time + 1],
    and returns the x_dot for the step (see state_derivative). state_logger may be None.
    """
    previous_state, _, current_state, termFlag = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)

    x_sim[:, sim_time + 1] = current_state[:4].numpy()
    if state_logger is not None:
        state_logger.info(x_sim[:, sim_time + 1])

    x_dot = state_derivative(previous_state, current_state, dt)
    assert(x_dot.dtype == torch.float32)
    return x_dot


def model_xdot(kbm_xdot, net_xdot, target):
    """
    x_dot predicted by the learned model.

    target: 'residual' if the network predicts x_dot_sim - x_dot_kbm (CRN), 'full' if it predicts x_dot (PDN).
    """
    if target == 'residual':
        return kbm_xdot + net_xdot
    if target == 'full':
        return net_xdot
    raise ValueError(f"target must be 'residual' or 'full', got {target!r}")


def plot_dynamics_comparison(sim_xdot, kbm_xdot, net_xdot, target, fig_fp, dt=0.1):
    """
    Plots x_dot from the sim, the KBM and the learned model (see model_xdot) for each state.
    """
    assert sim_xdot.shape[0] == kbm_xdot.shape[0] == net_xdot.shape[0]
    learned_xdot = model_xdot(kbm_xdot, net_xdot, target)
    learned_name = 'Corrected KBM' if target == 'residual' else 'Predicted'

    panels = [
        (r"$\dot{x}$", "Velocity (m/s)"),
        (r"$\dot{y}$", "Velocity (m/s)"),
        (r"$a$", r"Acceleration ($m/s^2$)"),
        (r"$\dot{\theta}$", "Angular Velocity (rad/s)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout()
    ts = np.arange(sim_xdot.shape[0]) * dt
    for i, (ax, (symbol, ylabel)) in enumerate(zip(axs.flat, panels)):
        ax.set_title(f"Comparison of {symbol}: PyBullet, KBM, {learned_name}")
        ax.set_ylabel(ylabel)
        if i >= 2:
            ax.set_xlabel("Time (s)")
        ax.plot(ts, sim_xdot[:, i], label=f'PyBullet {symbol}', color='r', linestyle='dashed')
        ax.plot(ts, kbm_xdot[:, i], label=f'KBM {symbol}', color='c', linestyle='dashdot')
        ax.plot(ts, learned_xdot[:, i], label=f'{learned_name} {symbol}', color='m', linestyle='dotted')
        ax.legend()

    plt.savefig(fig_fp, dpi=300)
    plt.close(fig)


def rmse(arr_1, arr_2):
    assert arr_1.shape == arr_2.shape
    return np.sqrt(np.mean((arr_1 - arr_2)**2, axis=0))


def dynamics_rmse(sim_xdot, kbm_xdot, net_xdot, target):
    """
    Returns per-state RMSE of the KBM and of the learned model (see model_xdot) against the sim.
    """
    return rmse(sim_xdot, kbm_xdot), rmse(sim_xdot, model_xdot(kbm_xdot, net_xdot, target))


def save_numpy_array_to_file(arr, fn):
    with open(fn, "wb") as f:
        np.save(f, arr)
