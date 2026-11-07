import copy

import casadi as cs
import numpy as np
import pytest
import torch

from learning.ModelTrainer import nominal_dynamics
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


def random_speed_target_state_action(n):
    states = np.column_stack([
        RNG.uniform(-5, 5, n),          # x
        RNG.uniform(-5, 5, n),          # y
        RNG.uniform(0, 6, n),           # v
        RNG.uniform(-np.pi, np.pi, n),  # yaw
    ])
    actions = np.column_stack([
        RNG.uniform(0, 6, n),           # v_cmd
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


def test_training_nominal_matches_mpc_nominal(speed_target_mpc_params):
    """The residual target must use the MPC's nominal x_dot. With the speed target action it is not f_x_dot."""
    f, model = nominal_dynamics(speed_target_mpc_params), csDSKBM(speed_target_mpc_params)
    states, actions = random_speed_target_state_action(50)
    got = np.array([f(torch.tensor(x), torch.tensor(u)).numpy() for x, u in zip(states, actions)])
    want = np.array([model.nominal_x_dot(x, u).full().ravel() for x, u in zip(states, actions)])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


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


def test_missing_action_key_means_acceleration(mpc_params):
    """Configs without model.action must give the same model as before."""
    explicit = copy.deepcopy(mpc_params)
    explicit["model"]["action"] = "acceleration"
    a, b = csDSKBM(mpc_params), csDSKBM(explicit)
    assert a.action == "acceleration"
    for x, u in zip(*random_state_action(10)):
        np.testing.assert_array_equal(a.step(x, u)[0].full(), b.step(x, u)[0].full())
        np.testing.assert_array_equal(a.nominal_x_dot(x, u).full(), a.f_x_dot(x, u).full())


def test_unknown_action_is_rejected(mpc_params):
    bad = copy.deepcopy(mpc_params)
    bad["model"]["action"] = "speed"
    with pytest.raises(ValueError):
        csDSKBM(bad)


@pytest.mark.parametrize("key, value", [("tau", 0.0), ("aMax", -1.0), ("nSubsteps", 0)])
def test_speed_target_rejects_bad_parameters(speed_target_mpc_params, key, value):
    bad = copy.deepcopy(speed_target_mpc_params)
    bad["model"][key] = value
    with pytest.raises(ValueError):
        csDSKBM(bad)


def test_speed_target_rejects_substeps_too_coarse_for_tau(speed_target_mpc_params):
    """tau 0.05 and dt 0.1: one substep of 0.1 s is longer than tau."""
    bad = copy.deepcopy(speed_target_mpc_params)
    bad["model"]["nSubsteps"] = 1
    with pytest.raises(ValueError):
        csDSKBM(bad)


def test_speed_target_speed_is_a_lag_for_small_gaps_and_capped_for_large_gaps(speed_target_mpc_params):
    m = csDSKBM(speed_target_mpc_params)
    v, x = 2.0, np.array([0, 0, 2.0, 0])
    small = float(m.f_x_dot(x, [(v + 0.01) / m.k, 0, 0])[2])
    assert small == pytest.approx(0.01 / m.tau, rel=1e-3)
    big = float(m.f_x_dot(x, [v + 10, 0, 0])[2])
    assert 0.99 * m.a_max < big <= m.a_max


def test_speed_target_step_matches_idas(speed_target_mpc_params):
    """tau is below dt, so the step must integrate the nonlinear model accurately (substeps)."""
    m = csDSKBM(speed_target_mpc_params)
    errors = [np.abs(m.step(x, u)[0].full().ravel() - m.forward_one_step(x, u)).max()
              for x, u in zip(*random_speed_target_state_action(50))]
    assert max(errors) < 1e-4


def test_speed_target_step_jacobians_match_finite_differences(speed_target_mpc_params):
    m = csDSKBM(speed_target_mpc_params)
    for x, u in zip(*random_speed_target_state_action(10)):
        _, A_d, B_d, _ = m.step(x, u)
        A_fd, B_fd = finite_difference_jacobians(m, x, u)
        np.testing.assert_allclose(A_d.full(), A_fd, atol=1e-5)
        np.testing.assert_allclose(B_d.full(), B_fd, atol=1e-5)


def test_residual_target_reproduces_next_state_in_speed_target_step(speed_target_mpc_params):
    """A residual trained on x_dot_sim - nominal_x_dot must give the sim's next state in the MPC step."""
    m = csDSKBM(speed_target_mpc_params)
    for x, u in zip(*random_speed_target_state_action(20)):
        xdot_target = RNG.uniform(-2, 2, 4)
        c = xdot_target - m.nominal_x_dot(x, u).full().ravel()
        np.testing.assert_allclose(m.step(x, u, c)[0].full().ravel(), x + m.dt * xdot_target, atol=1e-9)


def test_pdn_moves_the_state_by_dt_mu_with_speed_target(model_params, speed_target_mpc_params):
    params = dict(model_params, mpc=speed_target_mpc_params)
    model = ResDSKBM(params, use_lstm=True, target="full")
    x, u = np.array([0.5, -0.2, 1.5, 0.3]), np.array([1.8, 0.1, -0.05])
    mu = mu_for_step(model, x, u)
    np.testing.assert_allclose(one_step_change(model, x, u), model.dt * mu, atol=1e-6)
