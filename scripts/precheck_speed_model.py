"""
Compares how well the speed target model and the acceleration model (DSKBM) predict the next speed on logged
closed-loop runs. The logged wheel speed commands are turned into the action of each model.

    python -m scripts.precheck_speed_model --model-dir models/speed_target --runs results/dskbm results/crn

--model-dir has the speed target config. The acceleration model is built from the same params with
action: acceleration. Prints both RMSEs [m/s] and their ratio. PASS if rmse_speed_target <= 0.5 * rmse_dskbm,
otherwise FAIL and exit code 1.
"""
import argparse
import copy
import sys

import numpy as np

from TrajProc.models.DSKBM import csDSKBM
from utils.results import load_run, track_ids
from utils.simulation.simulation_utils import WHEEL_RADIUS, load_params


def model_actions(action, x_sim, u_sim, dt):
    """(3, n) model actions from the logged sim commands u_sim (3, n) [wheel speed, d_f, d_r] and states x_sim."""
    n = u_sim.shape[1]
    v_cmd = u_sim[0] * WHEEL_RADIUS
    if action == 'speed_target':
        first = v_cmd
    elif action == 'acceleration':
        first = (v_cmd - x_sim[2, :n]) / dt
    else:
        raise ValueError(f"unknown action {action!r}")
    return np.vstack([first, u_sim[1], u_sim[2]])


def next_speed_errors(model, x_sim, u_model):
    """(n,) predicted minus logged speed, one model step from each logged state."""
    return np.array([float(np.asarray(model.step(x_sim[:, t], u_model[:, t])[0]).ravel()[2]) - x_sim[2, t + 1]
                     for t in range(u_model.shape[1])])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="model dir with the speed target config")
    parser.add_argument("--runs", nargs="+", required=True, help="results dirs written by scripts.evaluate")
    args = parser.parse_args(argv)

    _, mpc_params = load_params(args.model_dir)
    dskbm_params = copy.deepcopy(mpc_params)
    dskbm_params['model']['action'] = 'acceleration'
    models = {'speed_target': csDSKBM(mpc_params), 'acceleration': csDSKBM(dskbm_params)}

    errors = {action: [] for action in models}
    for run_dir in args.runs:
        for i in track_ids(run_dir):
            run = load_run(run_dir, i)
            u_sim = np.load(f"{run_dir}/numpy/u_sim_track{i}.npy")[:, :run.n_steps]
            for action, model in models.items():
                u_model = model_actions(action, run.x_sim, u_sim, mpc_params['dt'])
                errors[action].append(next_speed_errors(model, run.x_sim, u_model))
    rmse_target, rmse_dskbm = (float(np.sqrt(np.mean(np.concatenate(errors[a]) ** 2)))
                               for a in ('speed_target', 'acceleration'))
    ratio = rmse_target / rmse_dskbm
    passed = rmse_target <= 0.5 * rmse_dskbm
    print(f"rmse_speed_target {rmse_target:.4f}  rmse_dskbm {rmse_dskbm:.4f}  ratio {ratio:.3f}  "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
