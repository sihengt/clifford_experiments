import os

import pytest
import torch
import yaml

from MPCConfigLoader import MPCConfigLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This config dir is tracked in git, so tests do not depend on untracked model weights.
CONFIG_DIR = os.path.join(REPO_ROOT, "models", "random_track", "config")


def load_config(name):
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mpc_params():
    return MPCConfigLoader(os.path.join(CONFIG_DIR, "mpc.yaml")).construct_casadi_params()


@pytest.fixture
def model_params(tmp_path, mpc_params):
    """Params dict in the same layout as init_pybullet_sim, with an untrained adm.pt in a temp model dir."""
    from learning.architecture.dynamicsModel import AdaptiveDynamicsModel

    params = {name: load_config(name) for name in ("controls", "robotRange", "sim", "terrain", "train", "network")}
    params["mpc"] = mpc_params
    params["dataDir"] = str(tmp_path)

    torch.manual_seed(0)
    network = AdaptiveDynamicsModel(params["network"], params["controls"], mpc_params)
    torch.save(network.state_dict(), tmp_path / "adm.pt")
    return params
