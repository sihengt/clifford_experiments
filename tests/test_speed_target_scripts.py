import os

import numpy as np
import pandas as pd
import pybullet as p
import pytest
import yaml

from scripts.compare_variants import load_variant, verdict
from scripts.fit_speed_model import collect_step_data, fit_speed_params, speed_v_next, step_programs
from scripts.make_random_tracks import random_tracks
from scripts.precheck_speed_model import model_actions, next_speed_errors
from scripts.replay_check import replay_run, replay_verdict
from scripts.tune_speed_settings import apply_overrides, evaluate_argv, pick_best
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


def write_variant_run(run_dir, cte, time_taken, max_steps=20):
    """results.csv and run.yaml of one run, one row per track."""
    from tests.test_results import write_fake_run
    os.makedirs(run_dir)
    for i, t in enumerate(time_taken):
        write_fake_run(str(run_dir), i, n_steps=int(min(t, max_steps - 1)))
    pd.DataFrame({'track': range(len(cte)), 'cte': cte, 'time_taken': time_taken}).to_csv(
        os.path.join(run_dir, 'results.csv'), index=False)
    with open(os.path.join(run_dir, 'run.yaml'), 'w') as f:
        yaml.safe_dump({'args': {'max_steps': max_steps}}, f)
    return str(run_dir)


def make_variant(tmp_path, name, runs):
    return load_variant([write_variant_run(tmp_path / f"{name}{i}", cte, tt) for i, (cte, tt) in enumerate(runs)])


def test_apply_overrides_sets_elements_and_values():
    cfg = {'R': [10, 10, 10], 'R_': [10, 10, 10], 'maxCmdAcc': 3.0}
    out = apply_overrides(cfg, {'R_[0]': 1, 'maxCmdAcc': 3.5})
    assert out == {'R': [10, 10, 10], 'R_': [1, 10, 10], 'maxCmdAcc': 3.5}
    assert cfg['R_'] == [10, 10, 10] and cfg['maxCmdAcc'] == 3.0


def test_each_grid_point_gets_its_own_build_dir():
    a = evaluate_argv('dskbm', '/out/point_0', 't.npy', 'acados')
    b = evaluate_argv('dskbm', '/out/point_1', 't.npy', 'acados', n_iters=10)
    assert a[a.index('--acados-build-dir') + 1] == '/out/point_0/build'
    assert b[b.index('--acados-build-dir') + 1] == '/out/point_1/build'
    assert a[a.index('--model-dir') + 1] == '/out/point_0/model'
    assert '--n-iters' not in a
    assert b[b.index('--n-iters') + 1] == '10'


def test_pick_best_prefers_laps_then_cte():
    summary = pd.DataFrame({'point': [0, 1, 2, 3], 'laps': [3, 5, 5, 4], 'mean_cte': [0.1, 0.4, 0.3, 0.05]})
    assert pick_best(summary) == 2


def test_verdict_identical_variants_are_not_worse_and_not_better(tmp_path):
    runs = [([0.5, 0.6, 0.7], [10, 12, 14]), ([0.52, 0.58, 0.7], [10, 12, 14])]
    result = verdict(make_variant(tmp_path, 'a', runs), make_variant(tmp_path, 'b', runs))
    assert result['not_worse'] and not result['better'] and not result['passed']


def test_verdict_lower_cte_beyond_spread_is_better(tmp_path):
    base = [([0.5, 0.6, 0.7], [10, 12, 14]), ([0.52, 0.58, 0.7], [10, 12, 14])]
    cand = [([0.3, 0.4, 0.5], [10, 12, 14]), ([0.32, 0.38, 0.5], [10, 12, 14])]
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert result['not_worse'] and result['better'] and result['passed']


def test_verdict_missed_lap_is_worse(tmp_path):
    base = [([0.5, 0.6, 0.7], [10, 12, 14])] * 2
    cand = [([0.1, 0.1, 0.1], [10, 12, 19])] * 2
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert not result['not_worse'] and not result['passed']


# Mean CTE per run is 0.5 and 0.7 for the baseline, so its spread over runs is 0.14
SAME_STEPS = [10, 12, 14]
BASE_CTE_RUNS = [([0.5] * 3, SAME_STEPS), ([0.7] * 3, SAME_STEPS)]


def cte_runs(offset):
    return [([c + offset for c in cte], tt) for cte, tt in BASE_CTE_RUNS]


def test_verdict_cte_higher_within_spread_is_still_not_worse(tmp_path):
    result = verdict(make_variant(tmp_path, 'a', BASE_CTE_RUNS), make_variant(tmp_path, 'b', cte_runs(0.05)))
    assert result['not_worse'] and not result['better']


def test_verdict_cte_higher_beyond_spread_is_worse(tmp_path):
    result = verdict(make_variant(tmp_path, 'a', BASE_CTE_RUNS), make_variant(tmp_path, 'b', cte_runs(0.3)))
    assert not result['not_worse'] and not result['passed']


def test_verdict_cte_lower_within_spread_is_not_better(tmp_path):
    result = verdict(make_variant(tmp_path, 'a', BASE_CTE_RUNS), make_variant(tmp_path, 'b', cte_runs(-0.05)))
    assert result['not_worse'] and not result['better'] and not result['passed']


def test_verdict_more_laps_is_better(tmp_path):
    base = [([0.5] * 3, [10, 12, 19])] * 2
    cand = [([0.5] * 3, [10, 12, 14])] * 2
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert result['not_worse'] and result['better'] and result['passed']


def test_verdict_fewer_steps_is_better(tmp_path):
    base = [([0.5] * 3, [10, 12, 14])] * 2
    cand = [([0.5] * 3, [8, 10, 12])] * 2
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert result['not_worse'] and result['better'] and result['passed']


def test_verdict_steps_ignore_tracks_missed_in_any_run(tmp_path):
    # Track 2 is missed once by the baseline, so only tracks 0 and 1 count for steps
    base = [([0.5] * 3, [10, 12, 14]), ([0.5] * 3, [10, 12, 19])]
    # The candidate is faster only on the excluded track 2
    cand = [([0.5] * 3, [10, 12, 12])] * 2
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert result['metrics']['steps_tracks'] == [0, 1]
    assert result['not_worse'] and not result['better']


def test_verdict_without_a_track_completed_in_all_runs_does_not_count_steps(tmp_path):
    # Every track is missed in some run. The candidate is faster where it completes, with the same misses.
    base = [([0.5] * 3, [19, 12, 19]), ([0.5] * 3, [10, 19, 19])]
    cand = [([0.5] * 3, [19, 10, 19]), ([0.5] * 3, [8, 19, 19])]
    result = verdict(make_variant(tmp_path, 'a', base), make_variant(tmp_path, 'b', cand))
    assert result['metrics']['steps_tracks'] == []
    assert result['not_worse'] and not result['better']
