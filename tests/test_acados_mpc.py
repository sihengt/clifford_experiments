import casadi as cs
import numpy as np
import pytest

pytest.importorskip("acados_template")

from TrajProc.controls.AcadosMPC import MAX_ITERATIONS, AcadosMPC
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.models.HistoryModel import HistoryModel

from conftest import load_mpc_params
from mpc_problems import Integrator, circle_reference, half_throttle, nlp_optimum, rollout, straight_reference


@pytest.fixture(scope="module")
def dskbm_mpc(tmp_path_factory):
    """Building an acados solver compiles C code (seconds), so the DSKBM tests share one."""
    params = load_mpc_params()
    model = HistoryModel(csDSKBM(params), n_past=1)
    return AcadosMPC(params, model, build_dir=str(tmp_path_factory.mktemp("acados_dskbm")), tol=1e-8)


def scenario(params, name):
    """x0, reference and warm start of the test problems used for the SQP MPC."""
    if name == "straight":
        x0 = np.array([0.0, 0.0, 1.0, 0.0])
        return x0, straight_reference(params, x0), np.zeros((3, params["T"]))
    radius, x0 = {
        "circle_A": (2.0, [0.0, -0.3, 3.0, 0.0]),  # full QP steps cycle between two points here
        "circle_B": (1.2, [0.0, -0.4, 4.0, 0.3]),
    }[name]
    x0 = np.array(x0)
    return x0, circle_reference(params, x0, radius), half_throttle(params)


@pytest.mark.parametrize("name", ["straight", "circle_A", "circle_B"])
def test_dskbm_plan_is_the_optimum_of_the_mpc_nlp(dskbm_mpc, name):
    """Same NLP as the SQP MPC (step map, cost, bounds, rate limits), checked against IPOPT. Scenario A needs
    about 60 iterations to reach tol 1e-8 (the merit line search rarely takes full steps there)."""
    params = dskbm_mpc.params
    x0, x_ref, u_bar = scenario(params, name)
    U_star = nlp_optimum(params, csDSKBM(params), x0, x_ref, u_bar)

    _, U, _ = dskbm_mpc.predict(x0, x_ref, u_bar, n_iters=MAX_ITERATIONS)

    assert dskbm_mpc.stats["status"] == "converged"
    np.testing.assert_allclose(U, U_star, atol=1e-6)


def test_returned_states_are_the_rollout_of_the_returned_actions(dskbm_mpc):
    """When SQP stops early, the solver's states need not match its actions; the returned X must."""
    params = dskbm_mpc.params
    x0, x_ref, u_bar = scenario(params, "circle_A")

    X, U, _ = dskbm_mpc.predict(x0, x_ref, u_bar, n_iters=1)

    assert dskbm_mpc.stats["status"] == "max_iterations"
    np.testing.assert_allclose(X, rollout(csDSKBM(params), x0, U), atol=1e-9)


def integrator_params(T):
    return {
        "T": T, "dt": 0.1,
        "Q": cs.DM(np.diag([1.0, 2.0])), "Qf": cs.DM(np.diag([3.0, 3.0])),
        "R": cs.DM([[0.5]]), "R_": cs.DM([[0.3]]),
        "U_lb": cs.DM([-1.0]), "U_ub": cs.DM([1.0]), "dU_b": cs.DM([2.0]),
    }


def history_nlp_optimum(params, model, z0, x_ref, u_init):
    """IPOPT optimum of the MPC problem over a HistoryModel, with the cost on the robot states."""
    T, dt = params["T"], params["dt"]
    opti = cs.Opti()
    Z, U = opti.variable(model.nz, T + 1), opti.variable(model.nu, T)
    J = 0
    for k in range(T):
        e = Z[:model.nx, k] - x_ref[:, k]
        J += e.T @ params["Q"] @ e + U[:, k].T @ params["R"] @ U[:, k]
        if k < T - 1:
            du = U[:, k + 1] - U[:, k]
            J += du.T @ params["R_"] @ du
            opti.subject_to(opti.bounded(-params["dU_b"], du / dt, params["dU_b"]))
        opti.subject_to(Z[:, k + 1] == model.f(Z[:, k], U[:, k]))
        opti.subject_to(opti.bounded(params["U_lb"], U[:, k], params["U_ub"]))
    e = Z[:model.nx, T] - x_ref[:, T]
    opti.minimize(J + e.T @ params["Qf"] @ e)
    opti.subject_to(Z[:, 0] == z0)
    opti.set_initial(U, u_init)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "tol": 1e-12})
    return opti.solve().value(U).reshape(model.nu, T)


def test_plan_with_a_window_model_is_the_optimum_of_its_nlp(tmp_path):
    """Another robot whose dynamics read a window of past states and actions: the past passed to predict()
    enters the plan, and the plan is the optimum of the NLP over the window model (checked against IPOPT)."""
    T = 5
    window = cs.SX.sym("w", 3, 2)
    residual = cs.Function("mu", [window], [cs.vertcat(0.5 * cs.sin(window[0, 1] + window[1, 1]),
                                                       -0.3 * window[0, 0])])
    model = HistoryModel(Integrator(), n_past=2, state_idx=[0], residual=residual)
    params = integrator_params(T)
    x0, past = np.array([0.2, -0.1]), np.array([[0.4, 0.9], [-0.3, 0.5]])
    x_ref = np.vstack([np.linspace(0.2, 1.5, T + 1), np.zeros(T + 1)])
    u_bar = np.zeros((1, T))
    U_star = history_nlp_optimum(params, model, model.pack(x0, past), x_ref, u_bar)

    mpc = AcadosMPC(params, model, build_dir=str(tmp_path), tol=1e-8)
    _, U, _ = mpc.predict(x0, x_ref, u_bar, n_iters=50, past=past)

    assert mpc.stats["status"] == "converged"
    np.testing.assert_allclose(U, U_star, atol=1e-6)
