"""
One figure per track: robot path on the track and the MPC actions over time. Replaces data_postprocess.py.

    python -m scripts.plot_runs results/crn --name "CRN with CSLQ-MPC"

Writes <out>/track_<i>.png (default <out> = <results_dir>/figures).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

from plotting.MPCFigurePlotter import MPCFigurePlotter  # noqa: E402
from utils.results import load_runs  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--name", required=True, help="model name for the figure titles")
    parser.add_argument("--out", help="output dir (default: <results_dir>/figures)")
    parser.add_argument("--format", default="png", help="figure file format (png, pdf, ...)")
    args = parser.parse_args(argv)

    out = args.out or os.path.join(args.results_dir, "figures")
    os.makedirs(out, exist_ok=True)
    for track_id, run in load_runs(args.results_dir).items():
        plotter = MPCFigurePlotter(run.track, args.name, track_id)
        plotter.plot_run(run.x_sim, run.u_mpc)
        plotter.save_plot_and_close(os.path.join(out, f"track_{track_id}.{args.format}"))


if __name__ == "__main__":
    main()
