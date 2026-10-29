"""
Fits the speed model of the speed_target action to the PyBullet sim:

    v_dot = a_max * tanh((k * v_cmd - v) / (tau * a_max))

The sim is driven with speed commands in short episodes on a flat 100 m map. Every episode starts with a
robot reset at the origin. The fit episodes hold v0 for 20 steps, then step to clip(v0 + delta, 0, 6) for 10
steps. The held-out episodes use random commands. Only steps where v_next is measured one control step after v
are used.

    python -m scripts.fit_speed_model --model-dir models/random_track --n-substeps 10 --out fit.yaml

The output has tau, k, aMax, the fit and held-out RMSE of v_next [m/s], and tau/k/aMax refit per steering pair.
"""
import argparse
import os
import time

import numpy as np
import yaml
from scipy.optimize import least_squares

from clifford_pybullet.SimController import ROBOT_FLIPPED, ROBOT_OUT_OF_BOUNDS
from utils.simulation.simulation_utils import WHEEL_RADIUS, init_pybullet_sim

V0S = (0, 1, 2, 3, 4, 5)
STEERING = ((0.0, 0.0), (0.2, 0.0), (0.35, -0.35))
DELTAS = (2, 1, 0.5, -0.5, -1, -2)
V_MAX = 6.0
N_HOLD = 20
N_STEP = 10


def speed_v_next(v, v_cmd, tau, k, a_max, dt, n_substeps):
    """One control step dt of the speed equation, with n_substeps RK4 substeps. Inputs are arrays."""
    def v_dot(v):
        return a_max * np.tanh((k * v_cmd - v) / (tau * a_max))

    h = dt / n_substeps
    for _ in range(n_substeps):
        k1 = v_dot(v)
        k2 = v_dot(v + h / 2 * k1)
        k3 = v_dot(v + h / 2 * k2)
        k4 = v_dot(v + h * k3)
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return v


def fit_speed_params(v, v_cmd, v_next, dt, n_substeps, x0=(0.05, 1.0, 5.0)):
    """Least squares fit of tau, k, aMax to (v, v_cmd) -> v_next. Returns dict(tau, k, aMax, rmse)."""
    def residual(theta):
        return speed_v_next(v, v_cmd, *theta, dt, n_substeps) - v_next

    result = least_squares(residual, x0, bounds=([1e-3, 0.5, 0.5], [1.0, 1.5, 20.0]))
    return dict(tau=float(result.x[0]), k=float(result.x[1]), aMax=float(result.x[2]),
                rmse=float(np.sqrt(np.mean(result.fun ** 2))))


def step_programs(seed):
    """
    Returns (fit_episodes, heldout_episodes). An episode is a list of (v_cmd, steer_f, steer_r) per control step.
    """
    fit = []
    for v0 in V0S:
        for steer_f, steer_r in STEERING:
            for delta in DELTAS:
                v1 = float(np.clip(v0 + delta, 0, V_MAX))
                fit.append([(float(v0), steer_f, steer_r)] * N_HOLD + [(v1, steer_f, steer_r)] * N_STEP)

    rng = np.random.default_rng(seed)
    heldout = []
    for _ in range(6):
        episode = []
        for _ in range(10):
            v_cmd = float(rng.uniform(0, V_MAX))
            steer_f, steer_r = (float(s) for s in rng.uniform(-0.5, 0.5, 2))
            episode += [(v_cmd, steer_f, steer_r)] * 5
        heldout.append(episode)
    return fit, heldout


def collect_step_data(sim, episodes):
    """
    Drives the sim through the episodes. Each episode starts with a robot reset at the origin. If the robot flips or
    leaves the map, that step and the rest of the episode are dropped.

    Returns dict of 1D arrays: v, v_cmd, steer_f, steer_r, v_next.
    """
    data = {key: [] for key in ('v', 'v_cmd', 'steer_f', 'steer_r', 'v_next')}
    for episode in episodes:
        sim.resetRobot(pose=((0, 0), (0, 0, 0, 1)))
        for v_cmd, steer_f, steer_r in episode:
            prev, _, nxt, term = sim.controlLoopStep([v_cmd / WHEEL_RADIUS, steer_f, steer_r],
                                                     commandInRealUnits=True)
            if term in (ROBOT_FLIPPED, ROBOT_OUT_OF_BOUNDS):
                break
            data['v'].append(float(prev[2]))
            data['v_cmd'].append(v_cmd)
            data['steer_f'].append(steer_f)
            data['steer_r'].append(steer_r)
            data['v_next'].append(float(nxt[2]))
    return {key: np.array(value) for key, value in data.items()}


def rmse(data, fit, dt, n_substeps):
    pred = speed_v_next(data['v'], data['v_cmd'], fit['tau'], fit['k'], fit['aMax'], dt, n_substeps)
    return float(np.sqrt(np.mean((pred - data['v_next']) ** 2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="model dir with config/*.yaml")
    parser.add_argument("--n-substeps", type=int, default=10, help="RK4 substeps per control step")
    parser.add_argument("--seed", type=int, default=1, help="seed of the held-out episodes")
    parser.add_argument("--out", required=True, help="output YAML file")
    args = parser.parse_args(argv)

    start = time.time()
    _, mpc_params, sim = init_pybullet_sim(args.model_dir, terrain_overrides={'mapScale': 0.2})
    dt = float(mpc_params['dt'])
    fit_episodes, heldout_episodes = step_programs(args.seed)
    pair_data = {}
    for steer_f, steer_r in STEERING:
        episodes = [e for e in fit_episodes if e[0][1:] == (steer_f, steer_r)]
        pair_data[(steer_f, steer_r)] = collect_step_data(sim, episodes)
    fit_data = {key: np.concatenate([d[key] for d in pair_data.values()]) for key in pair_data[STEERING[0]]}
    heldout_data = collect_step_data(sim, heldout_episodes)

    fit = fit_speed_params(fit_data['v'], fit_data['v_cmd'], fit_data['v_next'], dt, args.n_substeps)
    result = dict(tau=fit['tau'], k=fit['k'], aMax=fit['aMax'], nSubsteps=args.n_substeps, dt=dt,
                  fit_rmse=fit['rmse'], heldout_rmse=rmse(heldout_data, fit, dt, args.n_substeps),
                  n_fit_samples=len(fit_data['v']), n_heldout_samples=len(heldout_data['v']))

    per_steering = {}
    for (steer_f, steer_r), d in pair_data.items():
        pair_fit = fit_speed_params(d['v'], d['v_cmd'], d['v_next'], dt, args.n_substeps)
        per_steering[f"{steer_f},{steer_r}"] = {key: float(value) for key, value in pair_fit.items()}
    result['per_steering'] = per_steering
    result['run_time_s'] = float(time.time() - start)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
