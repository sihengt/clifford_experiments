"""
Compares two model variants over repeated evaluation runs and decides pass or fail with a fixed rule.

    python -m scripts.compare_variants --baseline results/a_run{0..4} --candidate results/b_run{0..4}

Per variant, the metrics of one run are the mean cross-track error over the tracks, the number of completed
laps and the mean steps of the tracks completed in every run of both variants. The spread of a metric is the
larger of the two variants' standard deviations over runs.
    not worse  every track: completion rate of the candidate >= baseline, and
               mean CTE of the candidate <= mean CTE of the baseline + spread
    better     any of: CTE lower by more than the spread, laps higher by more than the spread,
               steps lower by more than the spread
    PASS       not worse and better
Exit code 1 on FAIL.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml


def load_results(run_dir):
    """results.csv of one run, indexed by track, with a bool column completed (time_taken < max_steps - 1)."""
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    with open(os.path.join(run_dir, "run.yaml")) as f:
        max_steps = yaml.safe_load(f)["args"]["max_steps"]
    df["completed"] = df["time_taken"] < max_steps - 1
    return df.set_index("track")


def load_variant(run_dirs):
    """One results table per run (see load_results)."""
    return [load_results(run_dir) for run_dir in run_dirs]


def _std(values):
    return float(np.std(values, ddof=1 if len(values) > 1 else 0))


def _metrics(runs, steps_tracks):
    """Per-run lists of mean CTE, laps and mean steps over steps_tracks."""
    cte = [r["cte"].mean() for r in runs]
    laps = [int(r["completed"].sum()) for r in runs]
    steps = [r.loc[steps_tracks, "time_taken"].mean() for r in runs] if len(steps_tracks) else []
    return {"cte": cte, "laps": laps, "steps": steps}


def verdict(baseline, candidate):
    """baseline, candidate: output of load_variant. Returns not_worse, better, passed and the metrics."""
    tracks = baseline[0].index
    completion_b = np.mean([r.loc[tracks, "completed"].to_numpy() for r in baseline], axis=0)
    completion_c = np.mean([r.loc[tracks, "completed"].to_numpy() for r in candidate], axis=0)
    always = np.all([r.loc[tracks, "completed"].to_numpy() for r in baseline + candidate], axis=0)
    steps_tracks = list(tracks[always])

    b, c = _metrics(baseline, steps_tracks), _metrics(candidate, steps_tracks)
    spread = {m: max(_std(b[m]), _std(c[m])) if len(b[m]) else 0.0 for m in b}
    mean = {m: (np.mean(b[m]) if len(b[m]) else np.nan, np.mean(c[m]) if len(c[m]) else np.nan) for m in b}

    not_worse = bool(np.all(completion_c >= completion_b)) and mean["cte"][1] <= mean["cte"][0] + spread["cte"]
    better = bool(mean["cte"][0] - mean["cte"][1] > spread["cte"]
                  or mean["laps"][1] - mean["laps"][0] > spread["laps"]
                  or (len(steps_tracks) > 0 and mean["steps"][0] - mean["steps"][1] > spread["steps"]))
    return {"not_worse": not_worse, "better": better, "passed": not_worse and better,
            "metrics": {"mean": mean, "spread": spread, "steps_tracks": steps_tracks,
                        "completion": {"baseline": completion_b, "candidate": completion_c}}}


def table(baseline, candidate, steps_tracks):
    lines = ["| variant | runs | laps completed | steps (tracks completed in all runs) | mean CTE (m) |",
             "|---|---|---|---|---|"]
    for name, runs in (("baseline", baseline), ("candidate", candidate)):
        m = _metrics(runs, steps_tracks)
        n_tracks = len(runs[0])
        steps = f"{np.mean(m['steps']):.1f}" if m["steps"] else "-"
        lines.append(f"| {name} | {len(runs)} | {np.mean(m['laps']):.1f}/{n_tracks} | {steps} | "
                     f"{np.mean(m['cte']):.3f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", nargs="+", required=True, help="results dirs of the baseline runs")
    parser.add_argument("--candidate", nargs="+", required=True, help="results dirs of the candidate runs")
    args = parser.parse_args(argv)

    baseline, candidate = load_variant(args.baseline), load_variant(args.candidate)
    result = verdict(baseline, candidate)
    m = result["metrics"]
    print(table(baseline, candidate, m["steps_tracks"]))
    print()
    print(f"spread (larger std over runs): CTE {m['spread']['cte']:.3f} m, laps {m['spread']['laps']:.2f}, "
          f"steps {m['spread']['steps']:.2f} ({len(m['steps_tracks'])} tracks completed in all runs)")
    print(f"not worse: {result['not_worse']}, better: {result['better']}")
    print("PASS" if result["passed"] else "FAIL")
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
