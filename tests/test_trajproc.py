import numpy as np
import pytest

from TrajProc.TrajProc import TrajProc

STEP = 0.05
T, DT, V = 6, 0.1, 2.0


def square(closed):
    xs = [0.0, 5.0, 5.0, 0.0] + ([0.0] if closed else [])
    ys = [0.0, 0.0, 5.0, 5.0] + ([0.0] if closed else [])
    return TrajProc().generate_path_from_wp(np.array(xs), np.array(ys), STEP)


def reference_at(path, idx):
    """Reference trajectory for a car standing on path point idx."""
    tp = TrajProc()
    tp.lastIndex = idx
    state = np.array([path[0, idx], path[1, idx], V, path[2, idx]])
    return tp.get_reference_trajectory(state, path, V, STEP, T, DT)


def test_path_points_are_evenly_spaced():
    path = square(closed=False)
    spacing = np.linalg.norm(np.diff(path[:2], axis=1), axis=0)
    assert np.allclose(spacing, STEP, atol=0.01)


def test_reference_mid_path_moves_ahead_at_target_speed():
    path = square(closed=False)
    xref, nn_idx, ref_idx = reference_at(path, 20)
    assert nn_idx in (20, 21)  # a car exactly on a point counts as having passed it
    assert np.all(xref[2] == V)
    # Point i of the reference is (i + 1) * V * DT ahead of the nearest point.
    expected = [nn_idx + int(round((i + 1) * V * DT / STEP)) for i in range(T + 1)]
    assert ref_idx[1:] == expected


def test_closed_path_keeps_target_speed_across_the_lap_end():
    """Fix 07: on a closed track the reference speed dropped to 0 for points past the end of the path."""
    path = square(closed=True)
    n = path.shape[1]
    xref, _, ref_idx = reference_at(path, n - 10)
    assert np.all(xref[2] == V)
    assert any(i < 50 for i in ref_idx[1:]), "reference should continue into the next lap"


def test_open_path_stops_at_the_end_instead_of_jumping_to_the_start():
    """Fix 07: on an open path the reference wrapped to the start of the path."""
    path = square(closed=False)
    n = path.shape[1]
    xref, _, ref_idx = reference_at(path, n - 10)
    assert all(i >= n - 10 for i in ref_idx[1:]), "reference must not jump back to the start"
    past_end = [k for k, i in enumerate(ref_idx[1:]) if i == n - 1]
    assert past_end, "reference should reach the last point"
    assert np.all(xref[2, past_end[1:]] == 0.0)
    np.testing.assert_allclose(xref[:2, -1], path[:2, -1])


@pytest.mark.parametrize("closed", [True, False])
def test_closed_is_detected_from_the_path(closed):
    assert TrajProc.is_closed(square(closed), STEP) == closed
