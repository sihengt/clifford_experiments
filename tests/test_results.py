import os

import numpy as np
import pytest

from utils.results import load_run, load_runs, track_ids

MAX_STEPS = 20


def write_fake_run(results_dir, track_id, n_steps, seed=0, xdot_net=None, xdot_sim=None, xdot_kbm=None):
    """Writes one track in the evaluate.py layout: x_sim / u_mpc zero-padded to MAX_STEPS."""
    rng = np.random.default_rng(seed + track_id)
    os.makedirs(os.path.join(results_dir, "numpy"), exist_ok=True)
    x_sim = np.zeros((4, MAX_STEPS))
    x_sim[:, :n_steps + 1] = rng.uniform(0.5, 1.5, (4, n_steps + 1))
    u_mpc = np.zeros((3, MAX_STEPS - 1))
    u_mpc[:, :n_steps] = rng.uniform(0.5, 1.5, (3, n_steps))
    arrays = {
        f"track{track_id}.npy": rng.uniform(size=(3, 50)),
        f"cte_track{track_id}.npy": rng.uniform(size=n_steps),
        f"numpy/x_sim_track{track_id}.npy": x_sim,
        f"numpy/u_mpc_track{track_id}.npy": u_mpc,
        f"numpy/xdot_sim_track{track_id}.npy": rng.uniform(size=(n_steps, 4)) if xdot_sim is None else xdot_sim,
        f"numpy/xdot_kbm_track{track_id}.npy": rng.uniform(size=(n_steps, 4)) if xdot_kbm is None else xdot_kbm,
        f"numpy/xdot_res_track{track_id}.npy": rng.uniform(size=(n_steps, 4)) if xdot_net is None else xdot_net,
    }
    for rel, arr in arrays.items():
        np.save(os.path.join(results_dir, rel), arr)


def test_run_is_cut_to_its_own_length(tmp_path):
    write_fake_run(str(tmp_path), 0, n_steps=12)
    run = load_run(str(tmp_path), 0)
    assert run.n_steps == 12
    assert run.x_sim.shape == (4, 13) and run.u_mpc.shape == (3, 12)
    assert np.all(run.x_sim != 0) and np.all(run.u_mpc != 0), "no zero padding left"


def test_track_ids_sort_numerically(tmp_path):
    for i in [0, 2, 10, 1]:
        write_fake_run(str(tmp_path), i, n_steps=3)
    assert track_ids(str(tmp_path)) == [0, 1, 2, 10]
    assert list(load_runs(str(tmp_path))) == [0, 1, 2, 10]


def test_missing_results_dir_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        track_ids(str(tmp_path))
