import casadi as cs
import numpy as np

from TrajProc.models.DSKBM import csDSKBM
from TrajProc.models.HistoryModel import HistoryModel

from mpc_problems import Integrator


def step(model, x, past, u):
    """One step of the model from robot state x and past window entries; returns (x_next, past_next)."""
    z_next = model.f(model.pack(np.asarray(x, dtype=float), np.asarray(past, dtype=float)), u)
    x_next, past_next = model.unpack(z_next)
    return np.asarray(x_next).ravel(), np.asarray(past_next)


def test_step_shifts_current_state_and_action_into_window():
    """Entries are [selected states, actions], oldest first; the oldest drops out and the current one goes in."""
    model = HistoryModel(Integrator(), n_past=2, state_idx=[1])

    x_next, past_next = step(model, x=[1.0, 2.0], past=[[10.0, 11.0], [20.0, 21.0]], u=3.0)

    np.testing.assert_allclose(x_next, [4.0, 2.0])
    np.testing.assert_allclose(past_next, [[20.0, 21.0], [2.0, 3.0]])


def test_residual_reads_window_oldest_first_with_current_step_last():
    """The residual sees [past..., current] and its output is added to x_dot of the robot."""
    window = cs.SX.sym('w', 3, 2)
    residual = cs.Function('mu', [window], [cs.vertcat(window[0, 0] - window[2, 0], window[1, 1])])
    model = HistoryModel(Integrator(), n_past=2, state_idx=[1], residual=residual)

    x_next, _ = step(model, x=[1.0, 2.0], past=[[10.0, 11.0], [20.0, 21.0]], u=3.0)

    # window = [[10, 11], [20, 21], [2, 3]] -> c = [10 - 2, 21]
    np.testing.assert_allclose(x_next, [1.0 + 3.0 + 8.0, 2.0 + 21.0])


def test_dskbm_without_history_matches_integrate(mpc_params):
    """The symbolic step of the DSKBM is the same step as the one integrate() takes."""
    kbm = csDSKBM(mpc_params)
    model = HistoryModel(kbm, n_past=1)
    x, u = np.array([1.0, -0.5, 3.0, 0.4]), np.array([1.0, 0.3, -0.2])

    x_next, past_next = step(model, x, np.zeros((1, 3)), u)

    np.testing.assert_allclose(x_next, kbm.integrate(x, u)[0].full().ravel(), atol=1e-12)
    np.testing.assert_allclose(past_next, [u])
