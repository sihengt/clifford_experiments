"""
Runs the MPC with a given model on the evaluation tracks and saves trajectories, dynamics diagnostics
and metrics.

Thesis rows:
    python -m scripts.evaluate --model dskbm --model-dir models/random_track --out results/dskbm
    python -m scripts.evaluate --model crn   --model-dir models/random_track --out results/crn
    python -m scripts.evaluate --model pdn   --model-dir models/full         --out results/pdn
    python -m scripts.evaluate --model dskbm --model-dir models/sysid        --out results/dskbm_mismatch
    python -m scripts.evaluate --model crn   --model-dir models/sysid        --out results/crn_mismatch

Models:
    dskbm   nominal KBM in the MPC. The LSTM in --model-dir is only used for diagnostics.
    crn     KBM + LSTM residual in the MPC (network trained on x_dot_sim - x_dot_kbm).
    pdn     LSTM output in the MPC (network trained on x_dot_sim).

Output layout (same as the thesis results dirs, so the plotting scripts read both):
    run.yaml                                arguments, git commits, adm.pt checksum, package versions
    config/*.yaml                           copy of the model configs used
    results.csv                             one row per track
    track{i}.npy                            reference path (3, N): x, y, yaw
    cte_track{i}.npy                        cross-track error per step
    numpy/x_sim_track{i}.npy                (4, max_steps) states, zero after the lap ends
    numpy/u_mpc_track{i}.npy                (3, max_steps - 1) MPC actions, zero after the lap ends
    numpy/xdot_{sim,kbm,res}_track{i}.npy   (n_steps, 4) x_dot from sim, KBM, network output
    DynamicsComparisonTrack{i}.png
    MPCResultsTrack{i}.{png,mp4}            only with --plot
"""
import argparse
import copy
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

import casadi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet as p
import torch
import yaml

from TrajProc.TrajProc import TrajProc
from TrajProc.controls.MPC import MPC
from TrajProc.models.ResDSKBM import ResDSKBM
from utils.StatusPrint import StatusPrint
from utils.simulation.simulation_utils import (PATH_STEP, draw_track, dynamics_rmse, init_pybullet_sim,
                                               init_u_bar, plan_mpc_step, plot_dynamics_comparison,
                                               save_numpy_array_to_file, step_sim_and_log)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class ModelSpec:
    use_lstm: bool  # add the network output to the model used by the MPC
    target: str     # what the network predicts: 'residual' or 'full'


MODELS = {
    'dskbm': ModelSpec(use_lstm=False, target='residual'),
    'crn':   ModelSpec(use_lstm=True,  target='residual'),
    'pdn':   ModelSpec(use_lstm=True,  target='full'),
}


@dataclass
class TrackResult:
    track: np.ndarray     # (3, N) reference path
    x_sim: np.ndarray     # (4, max_steps) states, zero after the last step
    u_mpc: np.ndarray     # (3, max_steps - 1) MPC actions, zero after the last step
    xdot_sim: np.ndarray  # (n_steps, 4)
    xdot_kbm: np.ndarray  # (n_steps, 4)
    xdot_net: np.ndarray  # (n_steps, 4) raw network output
    cte: np.ndarray       # (n_steps,)

    @property
    def n_steps(self):
        return len(self.cte)


def check_lap_completion(last_idx, nn_idx, path_length, threshold=50):
    """
    Returns True if last_idx was near the end of the path and nn_idx wrapped around to near the start.

    Parameters:
        nn_idx: current nearest index on path
        path_length: total number of points on path
        threshold: number of indices from either end to count as "near"
    """
    return last_idx > path_length - threshold and nn_idx < threshold


def run_track(waypoints, model_dir, spec, max_steps=200, n_iters=10, gui=False, recorder=None, plotter_dir=None):
    """
    Drives one lap of the track given by waypoints (n, 2) in map units. Stops when the lap is complete
    or after max_steps - 1 control steps.

    Params:
        plotter_dir: if given, saves a live plot frame per step there and makes a video at the end.
    """
    params, mpc_params, sim = init_pybullet_sim(model_dir, gui=gui, recorder=recorder)
    params['mpc'] = mpc_params
    try:
        tp = TrajProc()
        model = ResDSKBM(params, use_lstm=spec.use_lstm)

        starting_angle = np.arctan2(waypoints[1, 1], waypoints[1, 0])
        sim.resetRobot(pose=((0, 0), p.getQuaternionFromEuler([0, 0, starting_angle])))

        xs = waypoints[:, 0] * params['terrain']['mapScale']
        ys = waypoints[:, 1] * params['terrain']['mapScale']
        track = tp.generate_path_from_wp(xs, ys, PATH_STEP)
        if gui:
            draw_track(xs, ys)

        mpc_plotter = None
        if plotter_dir is not None:
            from plotting.MPCPlotter import MPCPlotter
            mpc_plotter = MPCPlotter(track, max_steps, frame_dir=os.path.join(plotter_dir, "frames"))

        nX = mpc_params['model']['nStates']
        mU = mpc_params['model']['nActions']
        x_sim = np.zeros((nX, max_steps))
        u_sim = np.zeros((mU, max_steps - 1))
        u_mpc = np.zeros((mU, max_steps - 1))
        x_sim[:, 0] = np.array([0.0, 0.0, 0.0, starting_angle])

        u_bar_start = init_u_bar(mpc_params)
        mpc = MPC(mpc_params, model)

        l_kbm, l_sim, l_res, l_cte = [], [], [], []
        prev_idx = 0
        for sim_time in range(max_steps - 1):
            # mpc.predict changes the model's history during rollouts; keep a clean copy.
            current_model = copy.deepcopy(mpc.model)

            u_sim_opt, U_mpc, l_ref_idx = plan_mpc_step(
                x_sim[:, sim_time], u_bar_start, track, tp, mpc, mpc_params, n_iters=n_iters)

            if check_lap_completion(prev_idx, l_ref_idx[0], track.shape[1]):
                break
            prev_idx = l_ref_idx[0]

            u_bar_start = U_mpc
            u_mpc_opt = U_mpc[:, -1]
            current_model.update_queue(x_sim[:, sim_time], u_mpc_opt)

            u_sim[:, sim_time] = u_sim_opt.numpy()
            u_mpc[:, sim_time] = u_mpc_opt

            if mpc_plotter is not None:
                mpc_plotter.plot_new_data(x_sim[:, sim_time], u_sim[:, sim_time], l_ref_idx, sim_time)

            l_sim.append(step_sim_and_log(sim, sim_time, x_sim, u_sim, None, mpc_params['dt']))
            l_cte.append(np.linalg.norm(x_sim[:2, sim_time] - track[:2, l_ref_idx[0]]))

            # Diagnostics: KBM and network output for the step just taken.
            l_kbm.append(current_model.f_x_dot(x_sim[:, sim_time], u_mpc_opt).full().T.squeeze(0))
            l_res.append(current_model.query_lstm().squeeze(0))

            mpc.model = copy.deepcopy(current_model)

        if mpc_plotter is not None:
            mpc_plotter.convert_frames_into_video(os.path.join(plotter_dir, "video.mp4"))
            mpc_plotter.save_plot_and_close(os.path.join(plotter_dir, "plot.png"))

        return TrackResult(
            track=track,
            x_sim=x_sim,
            u_mpc=u_mpc,
            xdot_sim=torch.cat(l_sim).reshape(-1, 4).cpu().numpy(),
            xdot_kbm=np.array(l_kbm),
            xdot_net=torch.cat(l_res).reshape(-1, 4).cpu().numpy(),
            cte=np.array(l_cte),
        )
    finally:
        p.disconnect(sim.physicsClientId)


def save_track_result(out_dir, i_track, result, target):
    save_numpy_array_to_file(result.track, os.path.join(out_dir, f"track{i_track}.npy"))
    save_numpy_array_to_file(result.cte, os.path.join(out_dir, f"cte_track{i_track}.npy"))

    numpy_dir = os.path.join(out_dir, "numpy")
    save_numpy_array_to_file(result.x_sim, os.path.join(numpy_dir, f"x_sim_track{i_track}.npy"))
    save_numpy_array_to_file(result.u_mpc, os.path.join(numpy_dir, f"u_mpc_track{i_track}.npy"))
    save_numpy_array_to_file(result.xdot_sim, os.path.join(numpy_dir, f"xdot_sim_track{i_track}.npy"))
    save_numpy_array_to_file(result.xdot_kbm, os.path.join(numpy_dir, f"xdot_kbm_track{i_track}.npy"))
    save_numpy_array_to_file(result.xdot_net, os.path.join(numpy_dir, f"xdot_res_track{i_track}.npy"))

    plot_dynamics_comparison(result.xdot_sim, result.xdot_kbm, result.xdot_net, target,
                             os.path.join(out_dir, f"DynamicsComparisonTrack{i_track}.png"))

    rmse_kbm, rmse_model = dynamics_rmse(result.xdot_sim, result.xdot_kbm, result.xdot_net, target)
    return {"rmse_sim_kbm": np.mean(rmse_kbm),
            "rmse_sim_res_kbm": np.mean(rmse_model),
            "cte": np.mean(result.cte),
            "time_taken": result.n_steps}


def git_state(path):
    def git(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_run_info(out_dir, args):
    adm_path = os.path.join(args.model_dir, "adm.pt")
    info = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git": {name: git_state(os.path.join(REPO_ROOT, sub))
                for name, sub in [("clifford_experiments", ""), ("TrajProc", "TrajProc"),
                                  ("clifford_pybullet", "clifford_pybullet")]},
        "adm_pt_sha256": sha256(adm_path) if os.path.exists(adm_path) else None,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "casadi": str(casadi.__version__), "numpy": str(np.__version__)},
    }
    with open(os.path.join(out_dir, "run.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    shutil.copytree(os.path.join(args.model_dir, "config"), os.path.join(out_dir, "config"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="model used by the MPC")
    parser.add_argument("--model-dir", required=True, help="dir with config/*.yaml and adm.pt")
    parser.add_argument("--out", required=True, help="output dir (must not exist unless --overwrite)")
    parser.add_argument("--tracks", default="poisson_generated_tracks.npy", help="(n_tracks, n_waypoints, 2) array")
    parser.add_argument("--track-ids", type=int, nargs="+", help="subset of tracks to run (default: all)")
    parser.add_argument("--max-steps", type=int, default=200, help="length of the state log; at most max-steps - 1 control steps")
    parser.add_argument("--n-iters", type=int, default=10, help="MPC linearize-and-solve iterations per step")
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI (runs in real time)")
    parser.add_argument("--record", action="store_true", help="save a camera video per track (needs --gui)")
    parser.add_argument("--plot", action="store_true", help="save a live-plot video and figure per track (slow)")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing --out dir")
    args = parser.parse_args(argv)
    if args.record and not args.gui:
        parser.error("--record needs --gui")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.gui:
        matplotlib.use("Agg")

    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"{args.out} exists. Use --overwrite or choose another --out.")
    if args.overwrite and os.path.exists(os.path.join(args.out, "config")):
        shutil.rmtree(os.path.join(args.out, "config"))
    os.makedirs(os.path.join(args.out, "numpy"), exist_ok=True)
    write_run_info(args.out, args)

    spec = MODELS[args.model]
    eval_tracks = np.load(args.tracks)
    track_ids = args.track_ids if args.track_ids is not None else range(eval_tracks.shape[0])

    rows = []
    for i_track in track_ids:
        StatusPrint(f"[evaluate] {args.model} track {i_track}")
        recorder = None
        if args.record:
            from clifford_pybullet.CameraRecorder import CameraRecorder
            recorder = CameraRecorder(os.path.join(args.out, f"SimTrack{i_track}.mp4"))
        plotter_dir = os.path.join(args.out, f"MPCResultsTrack{i_track}") if args.plot else None

        result = run_track(eval_tracks[i_track], args.model_dir, spec, max_steps=args.max_steps,
                           n_iters=args.n_iters, gui=args.gui, recorder=recorder, plotter_dir=plotter_dir)
        if recorder is not None:
            recorder.close()

        row = save_track_result(args.out, i_track, result, spec.target)
        rows.append({"track": i_track, **row})
        StatusPrint(f"[evaluate] track {i_track}: {result.n_steps} steps, mean CTE {row['cte']:.3f} m")
        pd.DataFrame(rows).set_index("track").to_csv(os.path.join(args.out, "results.csv"))


if __name__ == "__main__":
    main()
