import os

import numpy as np

from scripts import plot_comparison
from tests.test_results import write_fake_run


def test_comparison_uses_each_runs_own_length(tmp_path):
    """Fix 01: the second run used to be cut (or padded) to the first run's length."""
    dir_1, dir_2 = str(tmp_path / "a"), str(tmp_path / "b")
    write_fake_run(dir_1, 0, n_steps=8)
    write_fake_run(dir_2, 0, n_steps=15)

    [(track_id, run_1, run_2)] = plot_comparison.paired_runs(dir_1, dir_2)
    assert track_id == 0
    assert run_1.x_sim.shape[1] == 9 and run_2.x_sim.shape[1] == 16
    assert run_2.u_mpc.shape[1] == 15


def test_comparison_pairs_only_shared_tracks(tmp_path):
    dir_1, dir_2 = str(tmp_path / "a"), str(tmp_path / "b")
    for i in (0, 1, 2):
        write_fake_run(dir_1, i, n_steps=5)
    for i in (1, 2, 3):
        write_fake_run(dir_2, i, n_steps=5)
    assert [t for t, _, _ in plot_comparison.paired_runs(dir_1, dir_2)] == [1, 2]


def test_comparison_main_writes_one_figure_per_track(tmp_path):
    dir_1, dir_2, out = str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "figs")
    for i in (0, 1):
        write_fake_run(dir_1, i, n_steps=6)
        write_fake_run(dir_2, i, n_steps=9)
    plot_comparison.main([dir_1, dir_2, "--names", "CRN", "DSKBM", "--out", out])
    assert sorted(os.listdir(out)) == ["track_0.png", "track_1.png"]


def write_consistent_runs(results_dir, target, n_tracks=3, n_steps=10):
    """Runs where the learned model is exact: sim = kbm + net (residual) or sim = net (full)."""
    rng = np.random.default_rng(0)
    for i in range(n_tracks):
        kbm = rng.uniform(size=(n_steps, 4))
        net = rng.uniform(size=(n_steps, 4))
        sim = kbm + net if target == "residual" else net
        write_fake_run(results_dir, i, n_steps, xdot_sim=sim, xdot_kbm=kbm, xdot_net=net)


def test_dynamics_metrics_uses_the_given_target(tmp_path):
    """Fix 02: the target was guessed from the folder name, so a PDN dir not named results_full was
    scored as residual (kbm + net)."""
    from scripts import dynamics_metrics
    for target in ("residual", "full"):
        results_dir = str(tmp_path / f"results_pdn_{target}")
        write_consistent_runs(results_dir, target)
        rmse_kbm, rmse_model = dynamics_metrics.per_track_rmse(dynamics_metrics.load_runs(results_dir), target)
        assert np.allclose(rmse_model.values, 0.0), target
        assert np.all(rmse_kbm.values > 0)


def test_dynamics_metrics_has_one_row_per_track(tmp_path):
    from scripts import dynamics_metrics
    results_dir = str(tmp_path / "r")
    write_consistent_runs(results_dir, "residual", n_tracks=4)
    rmse_kbm, rmse_model = dynamics_metrics.per_track_rmse(dynamics_metrics.load_runs(results_dir), "residual")
    assert list(rmse_kbm.index) == [0, 1, 2, 3] and list(rmse_model.index) == [0, 1, 2, 3]


def test_dynamics_metrics_main_writes_into_the_results_dir(tmp_path):
    from scripts import dynamics_metrics
    results_dir = str(tmp_path / "r")
    write_consistent_runs(results_dir, "full")
    dynamics_metrics.main([results_dir, "--target", "full"])
    assert sorted(os.listdir(os.path.join(results_dir, "analysis"))) == [
        "AverageRMSEperState.pdf", "RMSEReduction.pdf", "rmse_kbm.csv", "rmse_model.csv"]


def test_dynamics_metrics_needs_a_target_for_old_results_dirs(tmp_path):
    import pytest
    from scripts import dynamics_metrics
    results_dir = str(tmp_path / "r")
    write_consistent_runs(results_dir, "full")
    with pytest.raises(SystemExit):
        dynamics_metrics.main([results_dir])  # no run.yaml and no --target


def test_plot_runs_writes_one_figure_per_track(tmp_path):
    from scripts import plot_runs
    results_dir = str(tmp_path / "r")
    for i in (0, 1, 2):
        write_fake_run(results_dir, i, n_steps=7)
    plot_runs.main([results_dir, "--name", "CRN"])
    assert sorted(os.listdir(os.path.join(results_dir, "figures"))) == ["track_0.png", "track_1.png", "track_2.png"]
