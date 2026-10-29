"""
Runs the evaluation for each point of a grid of MPC settings and picks the best point. Use seeded random
tracks here, never the evaluation tracks.

    python -m scripts.tune_speed_settings --model-dir models/speed_target --grid grid.yaml \
        --tracks tune_tracks.npy --solver sqp --out results/tune

The grid is a YAML list of override dicts for mpc.yaml, e.g.
    - {"R_[0]": 1, maxCmdAcc: 1.75}
A key like "R_[0]" sets one element of a list, any other key sets the value.

Each point gets its own copy of the model dir (OUT/point_<i>/model), its own acados build dir and its own
evaluation run (--model dskbm, the nominal model only). OUT/summary.csv has one row per point with the
overrides, the completed laps and the mean cross-track error. The best point has the most laps, then the
lowest mean cross-track error.
"""
import argparse
import copy
import os
import re
import shutil

import pandas as pd
import yaml

from scripts.evaluate import main as evaluate_main


def apply_overrides(mpc_cfg, overrides):
    """Returns a deep copy of mpc_cfg with the overrides applied."""
    cfg = copy.deepcopy(mpc_cfg)
    for key, value in overrides.items():
        match = re.fullmatch(r"(\w+)\[(\d+)\]", key)
        if match:
            cfg[match.group(1)][int(match.group(2))] = value
        else:
            cfg[key] = value
    return cfg


def evaluate_argv(model, point_dir, tracks, solver, n_iters=None):
    """Arguments of scripts.evaluate for one grid point. The acados build dir is private to the point."""
    argv = ["--model", model, "--model-dir", os.path.join(point_dir, "model"), "--out", os.path.join(point_dir, "eval"),
            "--tracks", tracks, "--solver", solver, "--acados-build-dir", os.path.join(point_dir, "build")]
    if n_iters is not None:
        argv += ["--n-iters", str(n_iters)]
    return argv


def pick_best(summary):
    """Point of the row with the most laps, then the lowest mean CTE."""
    best = summary.sort_values(["laps", "mean_cte"], ascending=[False, True]).iloc[0]
    return int(best["point"])


def run_point(i, overrides, args):
    point_dir = os.path.join(args.out, f"point_{i}")
    model_dir = os.path.join(point_dir, "model")
    shutil.copytree(args.model_dir, model_dir)
    mpc_path = os.path.join(model_dir, "config", "mpc.yaml")
    with open(mpc_path) as f:
        mpc_cfg = yaml.safe_load(f)
    with open(mpc_path, "w") as f:
        yaml.safe_dump(apply_overrides(mpc_cfg, overrides), f)
    evaluate_main(evaluate_argv("dskbm", point_dir, args.tracks, args.solver, args.n_iters))
    results = pd.read_csv(os.path.join(point_dir, "eval", "results.csv"))
    with open(os.path.join(point_dir, "eval", "run.yaml")) as f:
        max_steps = yaml.safe_load(f)["args"]["max_steps"]
    return {"point": i, "overrides": yaml.safe_dump(overrides, default_flow_style=True).strip(),
            "laps": int((results["time_taken"] < max_steps - 1).sum()), "mean_cte": results["cte"].mean()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="dir with config/*.yaml and adm.pt")
    parser.add_argument("--grid", required=True, help="YAML list of override dicts for mpc.yaml")
    parser.add_argument("--tracks", required=True, help="(n_tracks, n_waypoints, 2) array of tuning tracks")
    parser.add_argument("--solver", default="sqp", choices=["sqp", "acados"], help="MPC solver")
    parser.add_argument("--n-iters", type=int, help="maximum MPC SQP iterations per step (default: evaluate's)")
    parser.add_argument("--out", required=True, help="output dir (must not exist)")
    args = parser.parse_args(argv)

    with open(args.grid) as f:
        grid = yaml.safe_load(f)
    if os.path.exists(args.out):
        parser.error(f"{args.out} exists")
    rows = [run_point(i, overrides, args) for i, overrides in enumerate(grid)]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    best = pick_best(summary)
    print(f"best point: {best} ({rows[best]['overrides']})")


if __name__ == "__main__":
    main()
