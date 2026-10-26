"""Reads evaluation results dirs written by scripts/evaluate.py (and the older thesis results dirs)."""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Run:
    """One evaluated track, trimmed to the steps that were actually run."""
    track_id: int
    track: np.ndarray     # (3, N) reference path: x, y, yaw
    x_sim: np.ndarray     # (4, n_steps + 1) states, including the state after the last step
    u_mpc: np.ndarray     # (3, n_steps) saved MPC actions
    xdot_sim: np.ndarray  # (n_steps, 4)
    xdot_kbm: np.ndarray  # (n_steps, 4)
    xdot_net: np.ndarray  # (n_steps, 4) raw network output
    cte: np.ndarray       # (n_steps,) cross-track error

    @property
    def n_steps(self):
        return len(self.cte)


def track_ids(results_dir):
    """Track ids present in results_dir, in numeric order (track10 after track9)."""
    ids = [int(re.search(r"cte_track(\d+)\.npy$", f).group(1))
           for f in glob.glob(os.path.join(results_dir, "cte_track*.npy"))]
    if not ids:
        raise FileNotFoundError(f"no cte_track*.npy in {results_dir}")
    return sorted(ids)


def load_run(results_dir, track_id):
    """
    Loads one track. x_sim and u_mpc are saved zero-padded to the maximum run length; they are cut to
    the number of steps of this run, which is the length of the cross-track error log.
    """
    def load(rel):
        return np.load(os.path.join(results_dir, rel))

    cte = load(f"cte_track{track_id}.npy")
    n = len(cte)
    return Run(
        track_id=track_id,
        track=load(f"track{track_id}.npy"),
        x_sim=load(f"numpy/x_sim_track{track_id}.npy")[:, :n + 1],
        u_mpc=load(f"numpy/u_mpc_track{track_id}.npy")[:, :n],
        xdot_sim=load(f"numpy/xdot_sim_track{track_id}.npy"),
        xdot_kbm=load(f"numpy/xdot_kbm_track{track_id}.npy"),
        xdot_net=load(f"numpy/xdot_res_track{track_id}.npy"),
        cte=cte,
    )


def load_runs(results_dir):
    """All tracks in results_dir as {track_id: Run}, in numeric order."""
    return {i: load_run(results_dir, i) for i in track_ids(results_dir)}


def run_model(results_dir):
    """The --model of the run (dskbm, crn, pdn) from run.yaml, or None for results dirs made before run.yaml."""
    path = os.path.join(results_dir, "run.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)["args"]["model"]
