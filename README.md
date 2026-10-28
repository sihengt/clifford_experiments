# clifford_experiments

Learning-augmented MPC for the Clifford double-steer car, in PyBullet simulation.

A double-steer kinematic bicycle model (DSKBM) is used inside an SQP MPC (successive linearization with a
line search). An LSTM
either corrects it (**CRN**, corrective residual network: predicts `x_dot_sim - x_dot_kbm`) or replaces it
(**PDN**, pure data-driven network: predicts `x_dot_sim`). The controllers are compared on 9 closed
evaluation tracks.

State `[x, y, v, yaw]`, action `[a, delta_f, delta_r]`, `dt = 0.1 s`, MPC horizon `T = 6`.

The code behind the thesis results is tag `1.0.0` (in this repo and both submodules). Bugs found since
then are fixed in later commits. Some affect the thesis results.

## Setup

```bash
git clone --recurse-submodules git@github.com:sihengt/clifford_experiments.git
cd clifford_experiments
pip install -r requirements.txt      # tested with Python 3.8
python -m pytest                     # about 1 min; add -m "not sim" to skip the PyBullet tests
```

Submodules:
- `TrajProc/`: MPC (`controls/MPC.py`), nominal model (`models/DSKBM.py`), KBM + LSTM model
  (`models/ResDSKBM.py`), path and reference generation (`TrajProc.py`).
- `clifford_pybullet/`: Clifford robot, terrain and sim controller.

Not in git (copy them in by hand): trained weights `models/<name>/adm.pt` and training data
`models/<name>/trajData/`. The configs of `models/random_track` and the evaluation tracks
`poisson_generated_tracks.npy` are in git.

## Model directories

A model dir holds `config/*.yaml` (`controls`, `mpc`, `network`, `robotRange`, `sim`, `terrain`, `train`)
and, after training, `adm.pt`, `trajData/` and `robotMeta.yaml`.

| dir | network | notes |
|---|---|---|
| `models/random_track` | CRN (`target: residual`) | nominal `l_f = l_r = 0.342` |
| `models/full` | PDN (`target: full`) | same configs |
| `models/sysid` | CRN | mismatched `l_f = 0.484`, `l_r = 0.200` in `mpc.yaml` |

## Train

`train.yaml` must set `target: residual` (CRN) or `target: full` (PDN).

```bash
python -m learning.server models/my_model          # trains; data and adm.pt go into models/my_model
python -m learning.client                           # start one or more data collection clients
python -m learning.client --gui                     # a client with the PyBullet GUI
```

The server hands out data collection tasks (random tracks, driven with the DSKBM MPC), trains on the
collected windows, and saves `adm.pt` every 1000 iterations. TensorBoard logs go to `runs/`.

## Evaluate

```bash
python -m scripts.evaluate --model dskbm --model-dir models/random_track --out results/dskbm
python -m scripts.evaluate --model crn   --model-dir models/random_track --out results/crn
python -m scripts.evaluate --model pdn   --model-dir models/full         --out results/pdn
python -m scripts.evaluate --model crn   --model-dir models/sysid        --out results/crn_mismatch
```

Options: `--track-ids 0 3`, `--max-steps 200`, `--n-iters 10`, `--gui`, `--record` (camera video, needs
`--gui`), `--plot` (live plot video), `--overwrite`. One track takes about a minute headless.

Each output dir gets `run.yaml` (command, git commits of all three repos, `adm.pt` checksum, versions),
a copy of the configs, `results.csv` (per-track RMSE, CTE, steps) and the per-step arrays in `numpy/`.

## Figures and metrics

```bash
python -m scripts.dynamics_metrics results/crn                      # x_dot RMSE per state, bar plots
python -m scripts.plot_runs results/crn --name "CRN with CSLQ-MPC"  # one figure per track
python -m scripts.plot_comparison results/crn results/dskbm \
    --names "CRN with CSLQ-MPC" "DSKBM with SLQ-MPC" --out figures/crn_vs_dskbm
```

These also read results dirs made before `run.yaml` existed (`dynamics_metrics` then needs `--target`).

## Layout

```
learning/     training server, data collection client, trainer, LSTM model
scripts/      evaluate, dynamics_metrics, plot_runs, plot_comparison
plotting/     figure plotters
utils/        sim helpers, config loader, results loader
tests/        pytest suite
```
