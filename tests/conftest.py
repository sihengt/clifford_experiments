import os
import shutil

import pytest
import torch
import yaml

from utils.MPCConfigLoader import MPCConfigLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This config dir is tracked in git, so tests do not depend on untracked model weights.
CONFIG_DIR = os.path.join(REPO_ROOT, "models", "random_track", "config")


def load_config(name):
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def load_mpc_params():
    return MPCConfigLoader(os.path.join(CONFIG_DIR, "mpc.yaml")).construct_casadi_params()


@pytest.fixture
def mpc_params():
    return load_mpc_params()


@pytest.fixture
def model_dir(tmp_path):
    """A model dir with the tracked configs and an untrained adm.pt."""
    from learning.architecture.dynamicsModel import AdaptiveDynamicsModel

    model_dir = tmp_path / "model"
    shutil.copytree(CONFIG_DIR, model_dir / "config")
    mpc = MPCConfigLoader(os.path.join(CONFIG_DIR, "mpc.yaml")).construct_casadi_params()
    torch.manual_seed(0)
    network = AdaptiveDynamicsModel(load_config("network"), load_config("controls"), mpc)
    torch.save(network.state_dict(), model_dir / "adm.pt")
    return str(model_dir)


@pytest.fixture
def model_params(model_dir, mpc_params):
    """Params dict in the same layout as init_pybullet_sim, pointing at model_dir."""
    params = {name: load_config(name) for name in ("controls", "robotRange", "sim", "terrain", "train", "network")}
    params["mpc"] = mpc_params
    params["dataDir"] = model_dir
    return params
