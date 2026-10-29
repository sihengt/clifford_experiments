import os

import numpy as np
import pandas as pd
import pybullet as p
import pytest

from scripts.fit_speed_model import collect_step_data, fit_speed_params, speed_v_next, step_programs
from scripts.make_random_tracks import random_tracks
from scripts.precheck_speed_model import model_actions, next_speed_errors
from scripts.replay_check import replay_run, replay_verdict
from tests.conftest import CONFIG_DIR, load_config
from TrajProc.controls.MPC import MPC
from TrajProc.models.DSKBM import csDSKBM
from utils.nlp_reference import rollout
from utils.simulation.simulation_utils import WHEEL_RADIUS, init_pybullet_sim

MODEL_DIR = os.path.dirname(CONFIG_DIR)


def test_random_tracks_shape_start_and_seed():
    terrain = load_config("terrain")
    tracks = random_tracks(5, 7, terrain, seed=3)
    assert tracks.shape == (5, 7, 2)
    np.testing.assert_array_equal(tracks[:, 0], 0)
    assert np.all(tracks[:, 1:, 0] >= -terrain['mapWidth'] / 2) and np.all(tracks[:, 1:, 0] < terrain['mapWidth'] / 2)
    assert np.all(tracks[:, 1:, 1] >= -terrain['mapLength'] / 2) and np.all(tracks[:, 1:, 1] < terrain['mapLength'] / 2)
    np.testing.assert_array_equal(tracks, np.round(tracks))
    np.testing.assert_array_equal(tracks, random_tracks(5, 7, terrain, seed=3))
    assert not np.array_equal(tracks, random_tracks(5, 7, terrain, seed=4))


def test_speed_v_next_matches_the_model_step(speed_target_mpc_params):
    model = csDSKBM(speed_target_mpc_params)
    cfg = speed_target_mpc_params['model']
    rng = np.random.default_rng(0)
    for _ in range(50):
        v, v_cmd = rng.uniform(0, 6, 2)
        df, dr = rng.uniform(-0.4, 0.4, 2)
        x_next = np.array(model.step(np.array([1.0, 2.0, v, 0.3]), np.array([v_cmd, df, dr]))[0]).ravel()
        expected = speed_v_next(np.array([v]), np.array([v_cmd]), cfg['tau'], cfg['k'], cfg['aMax'],
                                speed_target_mpc_params['dt'], cfg['nSubsteps'])
        np.testing.assert_allclose(x_next[2], expected[0], atol=1e-9)


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    dt, n_substeps = 0.1, 10
    v, v_cmd = rng.uniform(0, 6, (2, 500))
    v_next = speed_v_next(v, v_cmd, 0.04, 0.97, 6.5, dt, n_substeps) + rng.normal(0, 1e-4, 500)
    fit = fit_speed_params(v, v_cmd, v_next, dt, n_substeps)
    assert fit['tau'] == pytest.approx(0.04, rel=0.02)
    assert fit['k'] == pytest.approx(0.97, rel=0.02)
    assert fit['aMax'] == pytest.approx(6.5, rel=0.02)


def test_step_programs_layout():
    fit, heldout = step_programs(seed=1)
    assert len(fit) == 6 * 3 * 6 and all(len(e) == 30 for e in fit)
    assert len(heldout) == 6 and all(len(e) == 50 for e in heldout)
    assert fit[0][0] == (0.0, 0.0, 0.0) and fit[0][-1][0] == 2.0
    assert all(0 <= c[0] <= 6 and abs(c[1]) <= 0.5 and abs(c[2]) <= 0.5 for e in heldout for c in e)
    assert step_programs(seed=1)[1] == heldout


@pytest.mark.sim
def test_collect_step_data_resets_each_episode():
    _, _, sim = init_pybullet_sim(MODEL_DIR, gui=False)
    try:
        episodes = [[(2.0, 0.0, 0.0)] * 5, [(2.0, 0.0, 0.0)] * 5]
        data = collect_step_data(sim, episodes)
        assert set(data) == {'v', 'v_cmd', 'steer_f', 'steer_r', 'v_next'}
        assert all(len(a) == 10 for a in data.values())
        # both episodes start from rest
        assert data['v'][0] < 0.1 and data['v'][5] < 0.1
        np.testing.assert_allclose(data['v'][:5], data['v'][5:], atol=1e-3)
        assert data['v_next'][4] > data['v'][0]
    finally:
        p.disconnect(sim.physicsClientId)


@pytest.mark.sim
def test_terrain_override_reaches_the_terrain():
    _, _, sim = init_pybullet_sim(MODEL_DIR, gui=False, terrain_overrides={'mapScale': 0.2})
    try:
        assert sim.terrain.terrainParams['mapScale'] == 0.2
        assert sim.terrain.mapBounds[:, 0].abs().max() == pytest.approx(50, abs=0.5)
    finally:
        p.disconnect(sim.physicsClientId)


def test_next_speed_errors_are_zero_on_the_models_own_rollout(speed_target_mpc_params):
    model = csDSKBM(speed_target_mpc_params)
    U = np.array([[1.0, 3.0, 5.0, 2.0], [0.1, -0.2, 0.3, 0.0], [0.0, 0.1, -0.3, 0.2]])
    x_sim = rollout(model, np.array([0.0, 0.0, 0.5, 0.1]), U)
    errors = next_speed_errors(model, x_sim, U)
    assert errors.shape == (4,)
    assert np.abs(errors).max() < 1e-9


def test_model_actions_round_trip():
    dt = 0.1
    rng = np.random.default_rng(0)
    x_sim = rng.uniform(0, 5, (4, 6))
    u_sim = rng.uniform(0, 50, (3, 5))
    acc = model_actions('acceleration', x_sim, u_sim, dt)
    np.testing.assert_allclose(x_sim[2, :5] + acc[0] * dt, u_sim[0] * WHEEL_RADIUS)
    np.testing.assert_array_equal(acc[1:], u_sim[1:])
    target = model_actions('speed_target', x_sim, u_sim, dt)
    np.testing.assert_allclose(target[0], u_sim[0] * WHEEL_RADIUS)


def replay_frame(**changes):
    row = dict(track=0, t=0, dist_u0=0.0, gap_rel=0.0, solve_time=0.01, status='converged')
    return pd.DataFrame([{**row, **changes}] * 3)


def test_replay_verdict_fails_on_each_criterion():
    assert replay_verdict(replay_frame())['passed']
    assert not replay_verdict(replay_frame(dist_u0=0.06))['passed']
    assert not replay_verdict(replay_frame(gap_rel=0.002))['passed']
    assert not replay_verdict(replay_frame(solve_time=0.12))['passed']


def test_replay_run_reports_one_row_per_step(tmp_path, speed_target_mpc_params):
    model = csDSKBM(speed_target_mpc_params)
    mpc = MPC(speed_target_mpc_params, model)
    n_steps = 3
    U = np.tile(np.array([[2.0], [0.0], [0.0]]), (1, n_steps))
    x_sim = rollout(model, np.array([0.0, 0.0, 2.0, 0.0]), U)
    os.makedirs(tmp_path / "numpy")
    s = np.arange(0, 30, 0.05)
    np.save(tmp_path / "track0.npy", np.vstack([s, np.zeros_like(s), np.zeros_like(s)]))
    np.save(tmp_path / "cte_track0.npy", np.zeros(n_steps))
    np.save(tmp_path / "numpy" / "x_sim_track0.npy", np.hstack([x_sim, np.zeros((4, 5))]))
    df = replay_run(mpc, model, speed_target_mpc_params, str(tmp_path), n_iters=50)
    assert len(df) == n_steps
    assert list(df.columns) == ['track', 't', 'dist_u0', 'gap_rel', 'solve_time', 'status']
    assert df.dist_u0.max() < 1e-3
