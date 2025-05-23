import yaml
import os
import logging
import pybullet as p
from utils.planarRobotState import convert_planar
from utils.planarRobotState import convert_planar_world_frame
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.RandomRockyTerrain import RandomRockyTerrain
from clifford_pybullet.Terrain import Terrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam

import torch
import numpy as np
from control.PurePursuit import PurePursuit
from plotting.PurePursuitPlot import PurePursuitPlot
import matplotlib.pyplot as plt

from TrajProc.TrajProc import TrajProc
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.controls.MPC import MPC

from MPCPlotter import MPCPlotter
from VelAccPlotter import VelAccPlotter

# from TrajProc.scripts import *
import casadi as cs
import time

from simple_pid import PID

############
## FOR MPC #
############
N_STATES = 4
N_ACTIONS = 3
L = 0.684
l_f = 0.342
l_r = 0.342
T = 10
DT = 0.1 # "simTimeStep": 0.004166666666666 * number of steps per control: 24
SIM_DURATION = 50  # time steps

MAX_SPEED = 10.0
MAX_STEER = np.radians(30)
MAX_D_ACC = 2.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 5.0
REF_VEL = 1.0

def main(data_dir):    
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }
    
    # plt.ion()
    tp = TrajProc()

    # PyBullet + robot setup
    physicsClientID = p.connect(p.GUI)
    # physicsClientID = p.connect(p.DIRECT)
    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    robot = CliffordRobot(robotParams)
    
    # Terrain
    terrain = Terrain(params['terrain'], physicsClientId=physicsClientID)

    targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -6.0, 6.0, 6.0)
    maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10.0, 2.5)

    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=True,
        camFollowBot=True,
        stateProcessor=convert_planar_world_frame
    )
    
    sim.resetRobot(pose=((0,0),(0, 0, 0, np.sqrt(2)/2)))

    plotter = VelAccPlotter(SIM_DURATION)
    plt.pause(0.0001)

    # VARIABLES FOR TRACKING
    x_sim = np.zeros((N_STATES, SIM_DURATION))
    u_sim = np.zeros((N_ACTIONS, SIM_DURATION - 1))

    # Step 2: Create starting conditions x0
    x_sim[:, 0] = np.array([0.0, 0.0, 0.0, 0.0]).T
    
    # Create and configure logger
    logging.basicConfig(
        filename="logs/drivestate.log",
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO)
    state_logger = logging.getLogger('States')
    
    for sim_time in range(SIM_DURATION - 1):
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        maxForce = p.readUserDebugParameter(maxForceSlider)

        plt.pause(0.0001)
        iter_start = time.time()

        # Wheel velocity of 5.0 (vehicular) / 0.1 (radius)
        u_sim[:, sim_time] = np.array([targetVelocity / 0.1, 0, 0])

        current_state = x_sim[:, sim_time]
        # Construct data: [wheel_velocity, com_velocity, com_acceleration]
        
        if sim_time > 1:
            current_acc = (current_state[2] - x_sim[:, sim_time - 1][2]) / 0.1
        else:
            current_acc = 0
        data = np.array([targetVelocity, current_state[2], current_acc])
        plotter.plot_new_data(current_state, data, sim_time)
                
        with open("log_actions.txt", "a") as f:
            f.write("Sim_time={}\t u_sim={}\n".format(sim_time, u_sim[:, sim_time]))
        
        # TODO: add check for termFlag.
        # Sends action to simulator to drive the robot by "simTimeStep" "numStepsPerControl" times.
        lastState, action, current_state, termFlag = sim.controlLoopStep(u_sim[:, sim_time], commandInRealUnits=True)

    # current_state: [x, y, heading, vel_x, vel_y, vel_theta]
        current_state = current_state.numpy()
        current_state[[2, 3]] = current_state[[3, 2]]
        x_sim[:, sim_time + 1] = current_state[:4]

        state_logger.info(x_sim[:, sim_time + 1])
    
    breakpoint()

if __name__ == "__main__":
    main("nominal_dec_8")
