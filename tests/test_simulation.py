import logging
import os

import numpy as np
import pybullet as p
import pytest

from tests.conftest import CONFIG_DIR
from utils.simulation.simulation_utils import init_pybullet_sim, step_sim_and_log

MODEL_DIR = os.path.dirname(CONFIG_DIR)


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
