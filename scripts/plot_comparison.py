"""
Overlays two evaluation runs per track: robot paths on the track, and the MPC actions over time.

    python -m scripts.plot_comparison results/crn results/dskbm \
        --names "CRN with CSLQ-MPC" "DSKBM with SLQ-MPC" --out figures/crn_vs_dskbm

Writes <out>/track_<i>.png for every track id present in both results dirs.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

from plotting.MPCFigureComparisonPlotter import MPCFigureComparisonPlotter  # noqa: E402
from utils.results import load_runs  # noqa: E402


def paired_runs(results_dir_1, results_dir_2):
    """[(track_id, run_1, run_2)] for the tracks in both dirs. Each run is cut to its own length."""
    runs_1, runs_2 = load_runs(results_dir_1), load_runs(results_dir_2)
    return [(i, runs_1[i], runs_2[i]) for i in runs_1 if i in runs_2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir_1")
    parser.add_argument("results_dir_2")
    parser.add_argument("--names", nargs=2, required=True, metavar=("NAME_1", "NAME_2"), help="legend names")
    parser.add_argument("--out", required=True, help="output dir for the figures")
    parser.add_argument("--format", default="png", help="figure file format (png, pdf, ...)")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for track_id, run_1, run_2 in paired_runs(args.results_dir_1, args.results_dir_2):
        plotter = MPCFigureComparisonPlotter(run_1.track, *args.names, track_id)
        plotter.plot_full_data(run_1.x_sim, run_1.u_mpc, run_2.x_sim, run_2.u_mpc)
        plotter.save_plot_and_close(os.path.join(args.out, f"track_{track_id}.{args.format}"))


if __name__ == "__main__":
    main()
