"""MPC test problems shared by the SQP and acados tests: references, rollouts and the NLP optimum."""
import casadi as cs
import numpy as np

from TrajProc.TrajProc import TrajProc

PATH_STEP = 0.05


def straight_reference(mpc_params, x0):
    tp = TrajProc()
    path = tp.generate_path_from_wp(np.array([0.0, 20.0]), np.array([0.0, 0.0]), PATH_STEP)
    x_ref, _, _ = tp.get_reference_trajectory(x0, path, 2.0, PATH_STEP, mpc_params["T"], mpc_params["dt"])
    return x_ref


def circle_reference(mpc_params, x0, radius):
    """Reference at refVel on a counter-clockwise circle that starts at the origin heading +x."""
    theta = np.linspace(-np.pi / 2, 1.5 * np.pi, 40)
    tp = TrajProc()
    path = tp.generate_path_from_wp(radius * np.cos(theta), radius * np.sin(theta) + radius, PATH_STEP)
    x_ref, _, _ = tp.get_reference_trajectory(x0, path, mpc_params["refVel"], PATH_STEP, mpc_params["T"],
                                              mpc_params["dt"])
    return x_ref


def rollout(model, x0, U):
    X = [np.asarray(x0, dtype=float)]
    for u in U.T:
        X.append(model.integrate(X[-1], u)[0].full().ravel())
    return np.column_stack(X)


def nlp_optimum(mpc_params, model, x0, x_ref, u_init):
    """Actions that solve the MPC problem with the model's nonlinear step, from IPOPT with a tight tolerance."""
    T, dt = mpc_params["T"], mpc_params["dt"]
    Q, Qf, R, R_ = (mpc_params[name] for name in ("Q", "Qf", "R", "R_"))
    opti = cs.Opti()
    X, U = opti.variable(4, T + 1), opti.variable(3, T)
    J = 0
    for k in range(T):
        e = X[:, k] - x_ref[:, k]
        J += e.T @ Q @ e + U[:, k].T @ R @ U[:, k]
        if k < T - 1:
            du = U[:, k + 1] - U[:, k]
            J += du.T @ R_ @ du
            opti.subject_to(opti.bounded(-mpc_params["dU_b"], du / dt, mpc_params["dU_b"]))
        opti.subject_to(X[:, k + 1] == model.f_step(X[:, k], U[:, k], 0)[0])
        opti.subject_to(opti.bounded(mpc_params["U_lb"], U[:, k], mpc_params["U_ub"]))
    e = X[:, T] - x_ref[:, T]
    opti.minimize(J + e.T @ Qf @ e)
    opti.subject_to(X[:, 0] == x0)
    opti.set_initial(U, u_init)
    opti.set_initial(X, rollout(model, x0, u_init))
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "tol": 1e-12})
    return opti.solve().value(U)


def half_throttle(mpc_params):
    u_bar = np.zeros((3, mpc_params["T"]))
    u_bar[0, :] = mpc_params["maxAcc"] / 2
    return u_bar


class Integrator:
    """Two states, one action: x_next = x + [u, 0] + c. Next states are easy to derive by hand."""
    N_STATES = 2
    N_ACTIONS = 1

    def next_state(self, x, u, c):
        return x + cs.vertcat(u, 0) + c
