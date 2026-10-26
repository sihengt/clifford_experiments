"""
Per-track RMSE of x_dot predicted by the KBM and by the learned model, against the sim, plus two
bar plots. Replaces data_processing.py.

    python -m scripts.dynamics_metrics results/crn
    python -m scripts.dynamics_metrics OLD_RESULTS/results_pdn --target full --label PDN

The target (what the network predicts) is read from run.yaml. Results dirs made before run.yaml existed
need --target: 'residual' for CRN and DSKBM runs, 'full' for PDN runs.

Writes <results_dir>/analysis/{rmse_kbm.csv, rmse_model.csv, AverageRMSEperState.pdf, RMSEReduction.pdf}.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.evaluate import MODELS  # noqa: E402
from utils.results import load_runs, run_model  # noqa: E402
from utils.simulation.simulation_utils import dynamics_rmse  # noqa: E402

STATE_COLUMNS = ["xdot", "ydot", "vdot", "psidot"]
STATE_LABELS = [r"$\dot{x}$", r"$\dot{y}$", "a", r"$\dot{\psi}$"]


def per_track_rmse(runs, target):
    """Returns (rmse_kbm, rmse_model): DataFrames indexed by track id, one column per state."""
    kbm_rows, model_rows = {}, {}
    for track_id, run in runs.items():
        kbm_rows[track_id], model_rows[track_id] = dynamics_rmse(run.xdot_sim, run.xdot_kbm, run.xdot_net, target)
    as_frame = lambda rows: pd.DataFrame.from_dict(rows, orient="index", columns=STATE_COLUMNS).rename_axis("track")
    return as_frame(kbm_rows), as_frame(model_rows)


def plot_average_rmse(rmse_kbm, rmse_model, label, path):
    x = np.arange(len(STATE_COLUMNS))
    width = 0.35
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(x - width / 2, rmse_model.mean().values, width, label=label)
    plt.bar(x + width / 2, rmse_kbm.mean().values, width, label='KBM')
    plt.ylabel('Average RMSE')
    plt.xticks(x, STATE_LABELS)
    plt.title(f'Average RMSE per State Variable ({label} vs KBM)')
    plt.legend()
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_rmse_reduction(rmse_kbm, rmse_model, label, path):
    reduction = (rmse_kbm - rmse_model).rename(columns=dict(zip(STATE_COLUMNS, STATE_LABELS)))
    ax = reduction.plot(kind="bar", figsize=(10, 4), title=f"RMSE Reduction (KBM - {label})")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("RMSE Difference")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--target", choices=["residual", "full"], help="default: from run.yaml")
    parser.add_argument("--label", help="name of the learned model in plots (default: CRN or PDN)")
    args = parser.parse_args(argv)

    target = args.target
    if target is None:
        model = run_model(args.results_dir)
        if model is None:
            sys.exit(f"{args.results_dir} has no run.yaml. Pass --target residual (CRN, DSKBM) or full (PDN).")
        target = MODELS[model].target
    label = args.label or ("CRN" if target == "residual" else "PDN")

    rmse_kbm, rmse_model = per_track_rmse(load_runs(args.results_dir), target)

    out_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    rmse_kbm.to_csv(os.path.join(out_dir, "rmse_kbm.csv"))
    rmse_model.to_csv(os.path.join(out_dir, "rmse_model.csv"))
    plot_average_rmse(rmse_kbm, rmse_model, label, os.path.join(out_dir, "AverageRMSEperState.pdf"))
    plot_rmse_reduction(rmse_kbm, rmse_model, label, os.path.join(out_dir, "RMSEReduction.pdf"))

    print(f"mean RMSE ({target} target)")
    print(pd.DataFrame({"KBM": rmse_kbm.mean(), label: rmse_model.mean()}).T.round(4).to_string())


if __name__ == "__main__":
    main()
