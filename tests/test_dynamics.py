import copy

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


def mu_for_step(model, x, u):
    """Network output that integrate(x, u) should use: the history window ending with (x, u)."""
    probe = copy.deepcopy(model)
    probe.update_queue(x, u)
    return probe.query_lstm().cpu().numpy().ravel()


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


def finite_difference_jacobians(model, x, u, eps=1e-6):
    """Central differences of the next state from model.integrate() w.r.t. x and u."""
    def next_state(x, u):
        return copy.deepcopy(model).integrate(x, u)[0].full().ravel()
    A = np.column_stack([(next_state(x + e, u) - next_state(x - e, u)) / (2 * eps) for e in np.eye(4) * eps])
    B = np.column_stack([(next_state(x, u + e) - next_state(x, u - e)) / (2 * eps) for e in np.eye(3) * eps])
    return A, B


def test_integrate_returns_jacobians_of_its_next_state(mpc_params):
    """The QP linearizes the rollout step with A_d, B_d. SQP needs them to be the derivatives of that step."""
    kbm = csDSKBM(mpc_params)
    for x, u in zip(*random_state_action(10)):
        _, A_d, B_d, _ = kbm.integrate(x, u)
        A_fd, B_fd = finite_difference_jacobians(kbm, x, u)
        np.testing.assert_allclose(A_d.full(), A_fd, atol=1e-6)
        np.testing.assert_allclose(B_d.full(), B_fd, atol=1e-6)


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


def test_residual_model_jacobians_are_exact_with_network_output_held_fixed(model_params, monkeypatch):
    """CRN: with mu held fixed, A_d and B_d are the derivatives of the next state (dmu/dx, dmu/du are not modelled)."""
    model = ResDSKBM(model_params, use_lstm=True, target="residual")
    x, u = np.array([0.5, -0.2, 2.5, 0.3]), np.array([0.4, 0.3, -0.2])
    mu = torch.tensor(mu_for_step(model, x, u)[None, :], dtype=torch.float32)
    monkeypatch.setattr(model, "query_lstm", lambda *args, **kwargs: mu)

    _, A_d, B_d, _ = model.integrate(x, u)
    A_fd, B_fd = finite_difference_jacobians(model, x, u)
    np.testing.assert_allclose(A_d.full(), A_fd, atol=1e-6)
    np.testing.assert_allclose(B_d.full(), B_fd, atol=1e-6)


def one_step_change(model, x, u):
    return model.integrate(x, u)[0].full().ravel() - x


@pytest.mark.parametrize("target", ["residual", "full"])
def test_network_output_enters_the_mpc_model_according_to_target(model_params, target):
    """Fix 03: residual -> x_dot = f_kbm + mu; full -> x_dot = mu (KBM Jacobians kept for the QP).

    At the linearization point the affine model's x_dot is constant over the step, so
    x_{k+1} - x_k = gamma(A) @ x_dot, with gamma the same series the models use.
    """
    dt = model_params["mpc"]["dt"]
    model = ResDSKBM(model_params, use_lstm=True, target=target)
    x, u = np.array([0.5, -0.2, 1.5, 0.3]), np.array([0.4, 0.1, -0.05])

    mu = mu_for_step(model, x, u)
    f_kbm = model.f_x_dot(x, u).full().ravel()
    gamma = discrete_offset_gain(model.get_A(x, u).full(), dt)

    expected_x_dot = f_kbm + mu if target == "residual" else mu
    np.testing.assert_allclose(one_step_change(model, x, u), gamma @ expected_x_dot, atol=1e-6)


def test_full_target_keeps_kbm_jacobians(model_params):
    """The QP needs A and B to see how actions change the state; with no network Jacobians the KBM ones are used."""
    full = ResDSKBM(model_params, use_lstm=True, target="full")
    nominal = csDSKBM(model_params["mpc"])
    x, u = np.array([0.5, -0.2, 1.5, 0.3]), np.array([0.4, 0.1, -0.05])
    _, A_d, B_d, _ = full.integrate(x, u)
    _, A_d_kbm, B_d_kbm, _ = nominal.integrate(x, u)
    np.testing.assert_allclose(A_d.full(), A_d_kbm.full())
    np.testing.assert_allclose(B_d.full(), B_d_kbm.full())


def test_unknown_target_is_rejected(model_params):
    with pytest.raises(ValueError):
        ResDSKBM(model_params, target="resid")


def test_lstm_window_includes_the_step_being_predicted(model_params):
    """Fix 05: training targets step k from a window that ends with (x_k, u_k). integrate() queried the
    LSTM before adding (x_bar, u_bar), so the window ended one step early."""
    dt = model_params["mpc"]["dt"]
    model = ResDSKBM(model_params, use_lstm=True, target="residual")
    rng = np.random.default_rng(1)
    for _ in range(model.window_size):
        model.update_queue(rng.uniform(-1, 1, 4), rng.uniform(-0.3, 0.3, 3))
    x, u = np.array([0.5, -0.2, 1.5, 0.3]), np.array([0.4, 0.1, -0.05])

    mu_expected = mu_for_step(model, x, u)

    f_kbm = model.f_x_dot(x, u).full().ravel()
    gamma = discrete_offset_gain(model.get_A(x, u).full(), dt)
    np.testing.assert_allclose(one_step_change(model, x, u), gamma @ (f_kbm + mu_expected), atol=1e-6)
