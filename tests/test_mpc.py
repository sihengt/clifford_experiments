import casadi as cs
import numpy as np
import pytest

from TrajProc.controls.MPC import MPC
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.models.ResDSKBM import ResDSKBM

from mpc_problems import circle_reference, half_throttle, nlp_optimum, rollout, straight_reference


def test_mpc_solution_respects_constraints_and_moves_forward(mpc_params):
    T = mpc_params["T"]
    x0 = np.array([0.0, 0.0, 1.0, 0.0])
    x_ref = straight_reference(mpc_params, x0)
    u_bar = np.zeros((3, T))

    mpc = MPC(mpc_params, csDSKBM(mpc_params))
    X, U, _ = mpc.predict(x0, x_ref, u_bar, n_iters=3)

    assert X.shape == (4, T + 1) and U.shape == (3, T)
    np.testing.assert_allclose(X[:, 0], x0, atol=1e-8)

    u_lb, u_ub = np.asarray(mpc_params["U_lb"]).ravel(), np.asarray(mpc_params["U_ub"]).ravel()
    assert np.all(U >= u_lb[:, None] - 1e-6) and np.all(U <= u_ub[:, None] + 1e-6)

    du_bound = np.asarray(mpc_params["dU_b"]).ravel()
    assert np.all(np.abs(np.diff(U, axis=1)) / mpc_params["dt"] <= du_bound[:, None] + 1e-6)

    # Straight path along +x: the car should move forward and stay near y = 0.
    assert np.all(np.diff(X[0]) > 0)
    assert np.max(np.abs(X[1])) < 0.05


@pytest.mark.parametrize("radius, x0", [
    (2.0, [0.0, -0.3, 3.0, 0.0]),  # full QP steps cycle between two points here
    (1.2, [0.0, -0.4, 4.0, 0.3]),  # with A_d, B_d not the Jacobians of the step, the iteration stopped 0.02 away
])
def test_predict_reaches_the_optimum_of_its_nlp(mpc_params, radius, x0):
    """SQP converges to a KKT point of the NLP it linearizes; checked against IPOPT on the same NLP."""
    x0 = np.array(x0)
    x_ref = circle_reference(mpc_params, x0, radius)
    u_bar = half_throttle(mpc_params)
    model = csDSKBM(mpc_params)
    U_star = nlp_optimum(mpc_params, model, x0, x_ref, u_bar)

    _, U, _ = MPC(mpc_params, model).predict(x0, x_ref, u_bar, n_iters=30, tol=1e-8)
    np.testing.assert_allclose(U, U_star, atol=1e-4)


def test_predict_stops_once_the_step_is_below_tolerance(mpc_params):
    """Near-linear problem: the first QP step reaches the optimum, so the second step is ~1e-13 and SQP stops."""
    x0 = np.array([0.0, 0.0, 1.0, 0.0])
    x_ref = straight_reference(mpc_params, x0)
    mpc = MPC(mpc_params, csDSKBM(mpc_params))

    mpc.predict(x0, x_ref, np.zeros((3, mpc_params["T"])), n_iters=10, tol=1e-6)
    assert mpc.stats["status"] == "converged"
    assert mpc.stats["iterations"] == 2


def test_predict_reports_when_it_runs_out_of_iterations(mpc_params):
    """This case needs about 12 iterations, so 2 are not enough."""
    x0 = np.array([0.0, -0.3, 3.0, 0.0])
    x_ref = circle_reference(mpc_params, x0, 2.0)
    mpc = MPC(mpc_params, csDSKBM(mpc_params))

    mpc.predict(x0, x_ref, half_throttle(mpc_params), n_iters=2, tol=1e-6)
    assert mpc.stats["status"] == "max_iterations"
    assert mpc.stats["iterations"] == 2


def test_accepted_steps_never_increase_the_rolled_out_cost(mpc_params):
    """Full QP steps make the cost go up and down by ~200 here; accepted SQP steps must never make it go up.
    (Close to the optimum the decrease is below the cost's floating-point resolution, so equal is allowed.)"""
    x0 = np.array([0.0, -0.3, 3.0, 0.0])
    x_ref = circle_reference(mpc_params, x0, 2.0)
    mpc = MPC(mpc_params, csDSKBM(mpc_params))

    mpc.predict(x0, x_ref, half_throttle(mpc_params), n_iters=30, tol=1e-8)
    cost = np.array(mpc.stats["cost"])
    assert len(cost) > 2
    assert np.all(np.diff(cost) <= 0)


class WrongSteeringJacobian:
    """Next state as the base model, but the QP is told steering acts with the opposite sign."""
    def integrate(self, X_bar, U_bar):
        x_kp1, A_d, B_d, _ = super().integrate(X_bar, U_bar)
        B_d = cs.horzcat(B_d[:, 0], -B_d[:, 1:])
        return x_kp1, A_d, B_d, x_kp1 - A_d @ X_bar - B_d @ U_bar


class WrongSteeringJacobianDSKBM(WrongSteeringJacobian, csDSKBM):
    pass


class WrongSteeringJacobianResDSKBM(WrongSteeringJacobian, ResDSKBM):
    pass


def test_predict_keeps_the_warm_start_when_no_step_decreases_the_cost(mpc_params):
    """With wrong derivatives the QP direction can go uphill; SQP must not return a plan worse than the warm start."""
    x0 = np.array([0.0, -0.3, 3.0, 0.0])
    x_ref = circle_reference(mpc_params, x0, 2.0)
    u_bar = half_throttle(mpc_params)
    mpc = MPC(mpc_params, WrongSteeringJacobianDSKBM(mpc_params))

    _, U, _ = mpc.predict(x0, x_ref, u_bar, n_iters=5, tol=1e-6)
    assert mpc.stats["status"] == "no_decrease"
    np.testing.assert_array_equal(U, u_bar)


def test_model_history_after_predict_ends_with_the_returned_plan(model_params):
    """After predict(), self.model holds the history of the returned plan, also when the last rollout tried was
    a rejected step (here every step is rejected and the warm start is returned)."""
    mpc_params = model_params["mpc"]
    x0 = np.array([0.0, -0.3, 3.0, 0.0])
    x_ref = circle_reference(mpc_params, x0, 2.0)
    mpc = MPC(mpc_params, WrongSteeringJacobianResDSKBM(model_params, use_lstm=True))

    X, U, _ = mpc.predict(x0, x_ref, half_throttle(mpc_params), n_iters=5, tol=1e-6)
    assert mpc.stats["status"] == "no_decrease"
    np.testing.assert_allclose(mpc.model.u_q[-1].cpu().numpy(), U[:, -1], atol=1e-6)
    np.testing.assert_allclose(mpc.model.x_q[-1].cpu().numpy(), X[2:4, -2], atol=1e-6)


def test_predict_returns_the_states_the_model_predicts_for_its_actions(mpc_params):
    """The returned X is the rollout of the returned U, not the QP's linear prediction (2 m off here)."""
    x0 = np.array([0.0, -0.3, 3.0, 0.0])
    x_ref = circle_reference(mpc_params, x0, 2.0)
    model = csDSKBM(mpc_params)

    X, U, _ = MPC(mpc_params, model).predict(x0, x_ref, half_throttle(mpc_params), n_iters=1)
    np.testing.assert_allclose(X, rollout(model, x0, U), atol=1e-9)


def test_mpc_does_not_change_model_history_between_calls(mpc_params):
    """predict() restores the model before each iteration, so repeated calls give the same answer."""
    T = mpc_params["T"]
    x0 = np.array([0.0, 0.0, 1.0, 0.0])
    x_ref = straight_reference(mpc_params, x0)

    mpc = MPC(mpc_params, csDSKBM(mpc_params))
    _, U_first, _ = mpc.predict(x0, x_ref, np.zeros((3, T)), n_iters=2)
    _, U_second, _ = mpc.predict(x0, x_ref, np.zeros((3, T)), n_iters=2)
    np.testing.assert_allclose(U_first, U_second, atol=1e-8)
