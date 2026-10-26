"""Checks how training windows and targets are cut from collected trajectories."""
import pytest
import torch

from learning.ModelTrainer import kbm_nominal, sample_training_window

SEQ_LEN = 8
L_F = L_R = 0.342


def dynamics(state, action):
    return kbm_nominal(state, action, L_F, L_R)


def synthetic_sample(traj_lengths, seed=0):
    """Builds data in the collector format: one inf action row closes each trajectory.

    Every terminated trajectory with n steps has n + 1 states / xdot rows and n + 1 action rows (last one inf).
    """
    gen = torch.Generator().manual_seed(seed)
    states, actions, xdot = [], [], []
    for n in traj_lengths:
        states.append(torch.rand(n + 1, 4, generator=gen, dtype=torch.float64))
        xdot.append(torch.rand(n + 1, 4, generator=gen, dtype=torch.float64))
        a = torch.rand(n + 1, 3, generator=gen, dtype=torch.float64)
        a[-1] = torch.inf
        actions.append(a)
    return [torch.cat(states), torch.cat(actions), torch.cat(xdot)]


def find_window_start(sample, window):
    states, actions, _ = sample
    for start in range(actions.shape[0] - SEQ_LEN + 1):
        if torch.equal(actions[start:start + SEQ_LEN].float(), window["actionWindow"]) and \
                torch.equal(states[start:start + SEQ_LEN, 2:].float(), window["stateWindow"]):
            return start
    raise AssertionError("window not found in sample")


def test_residual_target_is_for_last_step_of_window():
    sample = synthetic_sample([30, 25])
    states, actions, xdot = [t.float() for t in sample]
    torch.manual_seed(0)
    for _ in range(50):
        window = sample_training_window(sample, SEQ_LEN, "residual", dynamics)
        start = find_window_start(sample, window)
        last = start + SEQ_LEN - 1

        assert torch.all(torch.isfinite(window["actionWindow"])), "window must not cross a trajectory end"
        expected = xdot[last] - dynamics(states[last], actions[last])
        torch.testing.assert_close(window["target"].reshape(-1), expected)


def test_full_target_is_xdot_of_last_step_of_window():
    sample = synthetic_sample([30, 25])
    xdot = sample[2].float()
    torch.manual_seed(1)
    for _ in range(50):
        window = sample_training_window(sample, SEQ_LEN, "full", dynamics)
        start = find_window_start(sample, window)
        assert torch.all(torch.isfinite(window["actionWindow"]))
        torch.testing.assert_close(window["target"].reshape(-1), xdot[start + SEQ_LEN - 1])


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError):
        sample_training_window(synthetic_sample([20]), SEQ_LEN, "resid", dynamics)


@pytest.mark.xfail(strict=True, reason="sampler draws from [1, N), so the first window is never used")
def test_every_valid_window_can_be_sampled():
    """Each trajectory of n steps has n - SEQ_LEN + 1 windows that end on a real action."""
    lengths = [12, 10]
    sample = synthetic_sample(lengths)
    torch.manual_seed(2)
    starts = {find_window_start(sample, sample_training_window(sample, SEQ_LEN, "residual", dynamics))
              for _ in range(400)}

    expected, offset = set(), 0
    for n in lengths:
        expected |= {offset + s for s in range(n - SEQ_LEN + 1)}
        offset += n + 1
    assert starts <= expected
    missing = expected - starts
    assert not missing, f"windows never sampled: {sorted(missing)}"
