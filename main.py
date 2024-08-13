import yaml
import os
import pybullet as p
from utils.planarRobotState import convert_planar
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.RandomRockyTerrain import RandomRockyTerrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam
from utils.plot_car import plot_car_without_circle

import torch
import numpy as np
from utils.PurePursuit import PurePursuit
import matplotlib.pyplot as plt

# TODO: are we able to compute everything in quaternions, and only use Euler angles for plotting?

def main(data_dir):
    MAX_TIME_STEPS = 1000

    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }
    
    physicsClientID = p.connect(p.GUI)
    robot = CliffordRobot(physicsClientID)
    terrain = RandomRockyTerrain(params['terrain'], physicsClientId=physicsClientID)
    terrain.generate()
    sim = SimController(
        robot,
        terrain,
        params['sim'],
        physicsClientId=physicsClientID,
        realtime=True,
        camFollowBot=True,
        stateProcessor=convert_planar
    )

    plt.grid(True)
    plt.legend()

    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    sim.robot.setParams(robotParams)
    sim.resetRobot()

    pure_pursuit = PurePursuit(**params['controls']['purePursuit'])
    
    ref_traj = None
    while ref_traj is None:
        start = torch.tensor([0, 0, 0])
        goalRange = torch.tensor([[-10, 10],[-10, 10],[-3.14,3.14]])
        target = torch.rand(goalRange.shape[0])*(goalRange[:,1]-goalRange[:,0]) + goalRange[:,0]

        # target = torch.tensor([10, 0, torch.pi/4])
        ref_traj, _ = pure_pursuit.gen_traj(start, target, 0.1)
    
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'green', label="Generated trajectory", linewidth=2, alpha=0.5)
    plot_car_without_circle(ref_traj[-1, :], params['controls']['purePursuit']['wheel_base'], color='r')
    plt.gca().legend()
    plt.gca().axis('equal')

    current_state = start
    current_marker = None
    lastFoundIndex = 0
    lookahead_circle = None
    lookahead_goal = None
    for t in range(MAX_TIME_STEPS):
        # Reference trajectory is in traj

        ref_state = pure_pursuit.get_lookahead_state(current_state, ref_traj)
        action = pure_pursuit.track_traj(current_state, ref_state)
        lastState, action, current_state, termFlag = sim.controlLoopStep(action.cpu().squeeze())

        # print("Timestep: {}\t Action taken: {}\t termFlag: {}".format(t, action, termFlag))
        
        current_marker = plot_car_without_circle(current_state, params['controls']['purePursuit']['wheel_base'], color='b', current_arrow=current_marker)
        
        if lookahead_circle:
            lookahead_circle.remove()
        lookahead_circle = plt.Circle(current_state[:2], params['controls']['purePursuit']['lookahead'], edgecolor='orange', facecolor='none')
        plt.gca().add_patch(lookahead_circle)

        plt.scatter(ref_state[0], ref_state[1], color='orange', s=0.5, label="Goal Point")
        # plt.scatter(newState[0], newState[1], color='black', s=0.5, zorder=-1, label="Pure Pursuit trajectory")
        plt.pause(0.01)
    input()
if __name__ == "__main__":
    main("nominal_dec_8")