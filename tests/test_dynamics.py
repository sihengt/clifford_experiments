import casadi as cs
import numpy as np
import pytest
import torch

from learning.ModelTrainer import kbm_nominal
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.models.ResDSKBM import ResDSKBM

RNG = np.random.default_rng(0)


def random_state_action(n):
    states = np.column_stack([
        RNG.uniform(-5, 5, n),          # x
        RNG.uniform(-5, 5, n),          # y
        RNG.uniform(0, 4, n),           # v
        RNG.uniform(-np.pi, np.pi, n),  # yaw
    ])
    actions = np.column_stack([
        RNG.uniform(-4, 4, n),          # a
        RNG.uniform(-0.5, 0.5, n),      # front steer
        RNG.uniform(-0.5, 0.5, n),      # rear steer
    ])
    return states, actions


def discrete_offset_gain(A, dt):
    """Maps a constant term c in x_dot = A x + B u + c to its effect on x_{k+1} (4th-order series, as in the models)."""
    A = np.asarray(A)
    A2 = A @ A
    return dt * np.eye(4) + dt**2 / 2 * A + dt**3 / 6 * A2 + dt**4 / 24 * A2 @ A


def test_training_kbm_matches_mpc_kbm(mpc_params):
    """The torch KBM used to build residual targets must equal the CasADi KBM used inside the MPC."""
    kbm = csDSKBM(mpc_params)
    states, actions = random_state_action(50)

    torch_xdot = kbm_nominal(torch.tensor(states), torch.tensor(actions),
                             mpc_params["model"]["l_f"], mpc_params["model"]["l_r"]).numpy()
    casadi_xdot = np.array([kbm.f_x_dot(x, u).full().ravel() for x, u in zip(states, actions)])

    np.testing.assert_allclose(torch_xdot, casadi_xdot, rtol=1e-9, atol=1e-9)


def test_integrate_returns_consistent_affine_model(mpc_params):
    kbm = csDSKBM(mpc_params)
    for x, u in zip(*random_state_action(20)):
        x_next, A_d, B_d, C_d = kbm.integrate(x, u)
        affine = (A_d @ x + B_d @ u + C_d).full().ravel()
        np.testing.assert_allclose(x_next.full().ravel(), affine, atol=1e-6)


def test_integrate_is_close_to_nonlinear_step(mpc_params):
    """One linearized step should stay close to the true nonlinear ODE step for dt = 0.1 s."""
    kbm = csDSKBM(mpc_params)
    for x, u in zip(*random_state_action(10)):
        x_next = kbm.integrate(x, u)[0].full().ravel()
        x_true = kbm.forward_one_step(x, u)
        np.testing.assert_allclose(x_next, x_true, atol=0.05)


def test_resdskbm_without_lstm_matches_dskbm(model_params):
    kbm = csDSKBM(model_params["mpc"])
    res_kbm = ResDSKBM(model_params, use_lstm=False)
    for x, u in zip(*random_state_action(10)):
        for a, b in zip(kbm.integrate(x, u), res_kbm.integrate(x, u)):
            np.testing.assert_allclose(np.asarray(a.full()), np.asarray(b.full()), atol=1e-12)


def test_resdskbm_with_lstm_adds_network_output_to_kbm(model_params):
    """Pins how the network output enters the MPC model: x_dot = f_kbm + mu (residual form)."""
    dt = model_params["mpc"]["dt"]
    with_lstm = ResDSKBM(model_params, use_lstm=True)
    without_lstm = ResDSKBM(model_params, use_lstm=False)

    x, u = np.array([0.5, -0.2, 1.5, 0.3]), np.array([0.4, 0.1, -0.05])
    mu = with_lstm.query_lstm().cpu().numpy().ravel()

    _, A_d, _, C_d_with = with_lstm.integrate(x, u)
    _, _, _, C_d_without = without_lstm.integrate(x, u)

    A = without_lstm.get_A(x, u).full()
    expected_shift = discrete_offset_gain(A, dt) @ mu
    np.testing.assert_allclose((C_d_with - C_d_without).full().ravel(), expected_shift, atol=1e-6)
