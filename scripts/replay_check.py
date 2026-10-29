"""
Re-solves the closed-loop MPC problems of a logged run with a solver and compares with the IPOPT optimum of the
same problem (nonlinear model step, tol 1e-12). Decides if the solver is correct and fast with the model.

    python -m scripts.replay_check --model-dir models/speed_target --run results/speed_target/sqp \
        --solver acados --n-iters 10 --build-dir /tmp/acados_build

States come from the log. References are rebuilt with TrajProc, in step order. The first warm start is
init_u_bar, after that it is the replay's own previous plan. Per step the output has the largest difference of
the first action to the optimum, the relative cost gap and the solve time. The verdict passes if no first action
is off by more than 0.05, the worst cost gap is below 0.1% and the maximum solve time is below 0.1 s.
Needs ACADOS_SOURCE_DIR and LD_LIBRARY_PATH for --solver acados. Exit code 1 on FAIL.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from TrajProc.TrajProc import TrajProc
from TrajProc.controls.MPC import MPC
from TrajProc.models.DSKBM import csDSKBM
from utils.nlp_reference import nlp_optimum, rollout
from utils.results import track_ids as list_track_ids
from utils.simulation.simulation_utils import PATH_STEP, init_u_bar, load_params


def replay_run(mpc, model, mpc_params, run_dir, n_iters, track_ids=None):
    """
    One row per logged step: track, t, dist_u0 (max abs difference of the first action to the IPOPT optimum),
    gap_rel (relative cost gap to the optimum), solve_time [s] and status.
    """
    if track_ids is None:
        track_ids = list_track_ids(run_dir)
    T, dt = mpc_params['T'], mpc_params['dt']
    # The cost is evaluated with the SQP MPC's cost function, for any solver under test
    cost = mpc.cost if isinstance(mpc, MPC) else MPC(mpc_params, model).cost
    rows = []
    for i in track_ids:
        track = np.load(f"{run_dir}/track{i}.npy")
        x_sim = np.load(f"{run_dir}/numpy/x_sim_track{i}.npy")
        n_steps = len(np.load(f"{run_dir}/cte_track{i}.npy"))
        tp = TrajProc()
        u_bar = init_u_bar(mpc_params, x_sim[2, 0])
        for t in range(n_steps):
            x0 = x_sim[:, t]
            x_ref, _, _ = tp.get_reference_trajectory(x0, track, mpc_params['refVel'], PATH_STEP, T, dt)
            _, U, x_ref = mpc.predict(x0, x_ref, u_bar, n_iters=n_iters)
            U_star = nlp_optimum(mpc_params, model, x0, x_ref, u_bar)
            J = float(cost(rollout(model, x0, U), U, x_ref)[0])
            J_star = float(cost(rollout(model, x0, U_star), U_star, x_ref)[0])
            rows.append(dict(track=i, t=t, dist_u0=float(np.abs(U[:, 0] - U_star[:, 0]).max()),
                             gap_rel=(J - J_star) / J_star, solve_time=mpc.stats['time'],
                             status=mpc.stats['status']))
            u_bar = U
    return pd.DataFrame(rows, columns=['track', 't', 'dist_u0', 'gap_rel', 'solve_time', 'status'])


def replay_verdict(df, max_u0_dist=0.05, max_gap=1e-3, max_time=0.1):
    frac_u0_over = float(np.mean(df.dist_u0 > max_u0_dist))
    worst_gap = float(df.gap_rel.max())
    verdict = dict(frac_u0_over=frac_u0_over, worst_gap=worst_gap, max_time=float(df.solve_time.max()),
                   median_time=float(df.solve_time.median()))
    verdict['passed'] = bool(frac_u0_over == 0 and worst_gap < max_gap and verdict['max_time'] < max_time)
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="model dir with config/mpc.yaml")
    parser.add_argument("--run", required=True, help="results dir written by scripts.evaluate")
    parser.add_argument("--solver", choices=["acados", "sqp"], required=True)
    parser.add_argument("--n-iters", type=int, default=10, help="maximum solver iterations per step")
    parser.add_argument("--build-dir", default=None, help="dir for the generated acados solver")
    parser.add_argument("--tracks", type=int, nargs="+", default=None, help="track ids, default all in the run")
    args = parser.parse_args(argv)

    _, mpc_params = load_params(args.model_dir)
    model = csDSKBM(mpc_params)
    if args.solver == "acados":
        from TrajProc.controls.AcadosMPC import AcadosMPC
        from TrajProc.models.HistoryModel import HistoryModel
        mpc = AcadosMPC(mpc_params, HistoryModel(model, n_past=1), build_dir=args.build_dir)
    else:
        mpc = MPC(mpc_params, model)

    df = replay_run(mpc, model, mpc_params, args.run, args.n_iters, args.tracks)
    print(f"{len(df)} steps, status {df.status.value_counts(normalize=True).round(3).to_dict()}")
    verdict = replay_verdict(df)
    print(f"frac_u0_over {verdict['frac_u0_over']:.3f}  worst_gap {verdict['worst_gap']:.1e}  "
          f"max_time {1e3 * verdict['max_time']:.1f} ms  median_time {1e3 * verdict['median_time']:.1f} ms  "
          f"{'PASS' if verdict['passed'] else 'FAIL'}")
    return 0 if verdict['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
