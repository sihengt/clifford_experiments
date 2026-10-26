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
