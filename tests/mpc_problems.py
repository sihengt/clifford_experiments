"""MPC test problems shared by the SQP and acados tests. The rollout and the NLP optimum are in utils/nlp_reference.py."""
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
