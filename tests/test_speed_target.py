import numpy as np
import pytest

from tests.conftest import speed_target_config
from TrajProc.controls.translator import MpcSpeedAction, SpeedTargetToSimAction
from utils.MPCConfigLoader import MPCConfigLoader
from utils.simulation.simulation_utils import init_u_bar


def test_speed_target_bounds_and_rate_limit(speed_target_mpc_params):
    p = speed_target_mpc_params
    assert float(p["U_lb"][0]) == 0.0 and float(p["U_ub"][0]) == p["maxSpeed"]
    assert float(p["dU_b"][0]) == 3.5
    assert float(p["U_ub"][1]) == p["maxSteer"]  # steering unchanged


def test_speed_target_without_max_cmd_acc_is_rejected():
    cfg = speed_target_config()
    del cfg["maxCmdAcc"]
    with pytest.raises(ValueError):
        MPCConfigLoader(cfg).construct_casadi_params()


def test_acceleration_config_keeps_old_bounds_and_warm_start(mpc_params):
    assert float(mpc_params["U_lb"][0]) == -mpc_params["maxAcc"]
    assert float(mpc_params["dU_b"][0]) == mpc_params["maxDAcc"]
    np.testing.assert_array_equal(init_u_bar(mpc_params, v0=3.0)[0], mpc_params["maxAcc"] / 2)


def test_speed_target_warm_start_is_current_speed(speed_target_mpc_params):
    np.testing.assert_array_equal(init_u_bar(speed_target_mpc_params, v0=2.5)[0], 2.5)


def test_speed_target_sim_command_is_v_cmd_over_wheel_radius():
    np.testing.assert_allclose(SpeedTargetToSimAction(0.1, MpcSpeedAction(2.0, 0.1, -0.2)).numpy(), [20.0, 0.1, -0.2])
