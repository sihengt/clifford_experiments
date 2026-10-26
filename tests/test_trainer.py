import copy

import pytest
import torch

from learning.ModelTrainer import ModelTrainer
from tests.conftest import load_config
from tests.test_training_windows import synthetic_sample


def trainer_params(tmp_path, target="residual"):
    train = load_config("train")
    train["target"] = target
    return {"trainDir": str(tmp_path), "dataDir": str(tmp_path), "train": train,
            "network": load_config("network"), "controls": load_config("controls"), "mpc": load_config("mpc")}


def test_target_must_be_set(tmp_path):
    params = trainer_params(tmp_path)
    del params["train"]["target"]
    with pytest.raises(ValueError, match="target"):
        ModelTrainer(server=None, params=params, log_dir=str(tmp_path / "tb"))


@pytest.mark.parametrize("target", ["residual", "full"])
def test_train_step_updates_weights_and_fits_one_window(tmp_path, target):
    params = trainer_params(tmp_path, target)
    params["train"]["warmup_steps"] = [1]  # skip the learning rate warmup for this test
    trainer = ModelTrainer(server=None, params=params, log_dir=str(tmp_path / "tb"))
    sample = synthetic_sample([9])  # two windows

    before = copy.deepcopy(trainer.adaptiveDynamicsModel.state_dict())
    losses = [trainer.train_step(sample) for _ in range(200)]

    after = trainer.adaptiveDynamicsModel.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert losses[-1] < losses[0] - 1.0
