import numpy as np

from TrajProc.controls.MPC import MPC
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.TrajProc import TrajProc

PATH_STEP = 0.05


def straight_reference(mpc_params, x0):
    tp = TrajProc()
    path = tp.generate_path_from_wp(np.array([0.0, 20.0]), np.array([0.0, 0.0]), PATH_STEP)
    x_ref, _, _ = tp.get_reference_trajectory(x0, path, 2.0, PATH_STEP, mpc_params["T"], mpc_params["dt"])
    return x_ref


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


def test_mpc_does_not_change_model_history_between_calls(mpc_params):
    """predict() restores the model before each iteration, so repeated calls give the same answer."""
    T = mpc_params["T"]
    x0 = np.array([0.0, 0.0, 1.0, 0.0])
    x_ref = straight_reference(mpc_params, x0)

    mpc = MPC(mpc_params, csDSKBM(mpc_params))
    _, U_first, _ = mpc.predict(x0, x_ref, np.zeros((3, T)), n_iters=2)
    _, U_second, _ = mpc.predict(x0, x_ref, np.zeros((3, T)), n_iters=2)
    np.testing.assert_allclose(U_first, U_second, atol=1e-8)
