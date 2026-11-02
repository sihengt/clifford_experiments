import logging
import os

import numpy as np
import pybullet as p
import pytest

from clifford_pybullet.Terrain import Terrain
from tests.conftest import CONFIG_DIR, load_config
from utils.simulation.simulation_utils import delay_steps, init_pybullet_sim, step_sim_and_log

MODEL_DIR = os.path.dirname(CONFIG_DIR)
SIM_TIME_STEP = 0.004166666666666  # sim.yaml: 1/240 s cut after 15 digits


@pytest.fixture
def sim():
    _, mpc_params, sim = init_pybullet_sim(MODEL_DIR, gui=False)
    sim.resetRobot(pose=((0, 0), (0, 0, 0, 1)))
    yield sim, mpc_params
    p.disconnect(sim.physicsClientId)


class CountingRecorder:
    def __init__(self):
        self.frames = 0

    def grab(self):
        self.frames += 1


@pytest.mark.sim
def test_recorder_grabs_one_frame_per_physics_step():
    recorder = CountingRecorder()
    _, _, sim = init_pybullet_sim(MODEL_DIR, gui=False, recorder=recorder)
    try:
        sim.resetRobot(pose=((0, 0), (0, 0, 0, 1)))
        frames_before = recorder.frames
        sim.controlLoopStep([0.0, 0.0, 0.0], commandInRealUnits=True)
        assert recorder.frames - frames_before == sim.simParams['numStepsPerControl']
    finally:
        p.disconnect(sim.physicsClientId)


class WheelSpeedRecorder:
    """Records the front right wheel speed (rad/s) after every physics step."""
    def __init__(self, sim):
        self.sim = sim
        self.speeds = []

    def grab(self):
        robot = self.sim.robot
        joint = robot.jointNameToID['frwheel2tire']
        self.speeds.append(p.getJointState(robot.robotID, joint, physicsClientId=self.sim.physicsClientId)[1])


@pytest.mark.sim
def test_previous_command_is_held_for_the_delay(sim):
    sim, _ = sim
    recorder = WheelSpeedRecorder(sim)
    sim.recorder = recorder
    sim.controlLoopStep([10.0, 0.0, 0.0], commandInRealUnits=True, delaySteps=10)

    speeds = np.abs(recorder.speeds)
    assert len(speeds) == sim.simParams['numStepsPerControl']
    # At rest the wheel chatters below 2 rad/s; on the 10 rad/s command it stays above 6.5 rad/s.
    assert np.all(speeds[:10] < 3.0)  # still on the zero command of the reset
    assert np.all(speeds[10:] > 5.0)  # 10 rad/s command applied


@pytest.mark.sim
def test_late_command_stretches_the_control_step(sim):
    sim, _ = sim
    recorder = CountingRecorder()
    sim.recorder = recorder
    n = sim.simParams['numStepsPerControl']
    sim.controlLoopStep([10.0, 0.0, 0.0], commandInRealUnits=True, delaySteps=n + 6)
    assert recorder.frames == n + 6


@pytest.mark.parametrize("latency, compute_time, steps", [
    (None, 0.123, 0),         # lockstep: no delay, whatever the compute time
    (0.05, 0.123, 12),        # fixed: 0.05 s is 12 physics steps
    (0.051, 0.123, 13),       # the command lands at the first physics step after the delay
    (0.1, 0.0, 24),           # the cut 1/240 s must not add a step
    ("measured", 0.123, 30),  # 0.123 s is 29.5 physics steps
    ("measured", 0.001, 1),
])
def test_delay_steps(latency, compute_time, steps):
    assert delay_steps(latency, compute_time, SIM_TIME_STEP) == steps


@pytest.mark.sim
def test_driving_forward_increases_x_and_speed(sim):
    sim, mpc_params = sim
    steps = 15
    x_sim = np.zeros((4, steps + 1))
    u_sim = np.tile(np.array([[10.0], [0.0], [0.0]]), (1, steps))  # wheel speed (rad/s), front, rear steer
    logger = logging.getLogger("test")

    for k in range(steps):
        x_dot = step_sim_and_log(sim, k, x_sim, u_sim, logger, mpc_params["dt"])
        assert x_dot.shape == (4,)

    assert x_sim[0, -1] > 0.3       # moved forward along +x
    assert x_sim[2, -1] > 0.3       # has speed
    assert abs(x_sim[1, -1]) < 0.2  # roughly straight


@pytest.fixture
def cid():
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(cid)


def terrain_friction(terrain, cid):
    return p.getDynamicsInfo(terrain.terrainBody, -1, physicsClientId=cid)[1]


@pytest.mark.sim
def test_terrain_friction_comes_from_params(cid):
    params = dict(load_config("terrain"), lateralFriction=1.0)
    terrain = Terrain(params, physicsClientId=cid)
    assert terrain_friction(terrain, cid) == pytest.approx(1.0)


@pytest.mark.sim
def test_missing_friction_key_keeps_bullet_default(cid):
    terrain = Terrain(load_config("terrain"), physicsClientId=cid)
    assert terrain_friction(terrain, cid) == pytest.approx(0.5)


@pytest.mark.sim
def test_set_params_updates_terrain_friction(cid):
    terrain = Terrain(dict(load_config("terrain"), lateralFriction=1.0), physicsClientId=cid)
    terrain.setParams(dict(load_config("terrain"), lateralFriction=0.8))
    assert terrain_friction(terrain, cid) == pytest.approx(0.8)
