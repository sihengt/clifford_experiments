import numpy as np
import pybullet as p
import pytest
import torch

import learning.client as client
from tests.conftest import load_config
from tests.test_simulation import MODEL_DIR
from TrajProc.TrajProc import TrajProc
from TrajProc.controls.MPC import MPC
from TrajProc.models.DSKBM import csDSKBM
from utils.simulation.simulation_utils import init_pybullet_sim


def split_trajectories(data):
    """Row counts of (states, actions, xdot) per trajectory, using the inf action rows as separators."""
    ends = torch.nonzero(data['actions'][:, 0] == torch.inf).flatten().tolist()
    counts, start = [], 0
    for end in ends:
        n_actions = end - start + 1  # including the inf row
        counts.append(n_actions)
        start = end + 1
    return counts


def test_close_trajectory_adds_final_state_and_xdot():
    data = {'states': [torch.zeros(5, 4)], 'actions': [torch.zeros(5, 3)], 'xdot': [torch.zeros(5, 4)]}
    final_state = torch.arange(7, dtype=torch.float32)  # [x, y, v, yaw, x_dot, y_dot, yaw_dot]
    client.close_trajectory(data, final_state)
    assert data['states'][-1].shape[0] == 6 and data['xdot'][-1].shape[0] == 6
    torch.testing.assert_close(data['states'][-1][-1], final_state[:4])
    torch.testing.assert_close(data['xdot'][-1][-1], torch.tensor([4.0, 5.0, 0.0, 6.0]))


@pytest.mark.sim
def test_collection_after_a_reset_plans_from_the_reset_state_and_keeps_rows_aligned(monkeypatch):
    """Fix 08: after a reset the MPC planned from the last state before the reset, and the last
    trajectory of a batch had one state / x_dot row fewer than actions."""
    params, mpc_params, sim = init_pybullet_sim(MODEL_DIR)
    params['mpc'] = load_config('mpc')
    params['train'] = dict(params['train'], simStepsPerBatch=20, trainPredSeqLen=8)
    try:
        planned_from = []
        real_plan = client.plan_mpc_step

        def plan(current_state, *args, **kwargs):
            planned_from.append(np.array(current_state, dtype=float))
            return real_plan(current_state, *args, **kwargs)
        monkeypatch.setattr(client, 'plan_mpc_step', plan)

        # Force a termination after the 10th planned step, so a second trajectory starts after a reset.
        real_step, reset_at = sim.controlLoopStep, {}

        def step(command, **kwargs):
            prev, cmd, cur, term = real_step(command, **kwargs)
            if len(planned_from) == 10 and 'plan_index' not in reset_at:
                reset_at['plan_index'] = len(planned_from)
                term = True
            return prev, cmd, cur, term
        monkeypatch.setattr(sim, 'controlLoopStep', step)

        data = client.collect_tracking_data(sim, TrajProc(), MPC(mpc_params, csDSKBM(mpc_params)), mpc_params, params)
    finally:
        p.disconnect(sim.physicsClientId)

    last_step_before_reset = planned_from[reset_at['plan_index'] - 1]
    first_step_after_reset = planned_from[reset_at['plan_index']]
    assert np.linalg.norm(last_step_before_reset[:2]) > 0.3, "test setup: the robot should have moved"
    assert np.linalg.norm(first_step_after_reset[:2]) < 0.1, "should plan from the reset pose"

    n_states, n_actions, n_xdot = data['states'].shape[0], data['actions'].shape[0], data['xdot'].shape[0]
    assert n_states == n_actions == n_xdot, "every trajectory needs a final state and x_dot row"
    assert len(split_trajectories(data)) == 2
