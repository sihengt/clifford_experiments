import yaml
import os
import pybullet as p
from utils.planarRobotState import convert_planar
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.TestEnv import TestEnv
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam

import torch
from utils.PurePursuit import PurePursuit
import matplotlib.pyplot as plt

# TODO: are we able to compute everything in quaternions, and only use Euler angles for plotting?

def main(data_dir):
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
    terrain = TestEnv(params['terrain'], physicsClientId=physicsClientID)
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
    sim.terrain.generate(stepWidth=5)
    sim.resetRobot()

    pure_pursuit = PurePursuit(**params['controls']['purePursuit'])
    traj = None
    while traj is None:
        start = torch.tensor([0, 0, 0])
        target = torch.tensor([10, 0, torch.pi/4])
        traj, _ = pure_pursuit.gen_traj(start, target, 0.1)
    
    breakpoint()
    for t in range(traj.shape[0]):
        refWindow = (t + torch.arange(params['controls']['purePursuit']['lookahead'])).clip(max=traj.shape[0] - 1)
        timeStepRef = traj[refWindow]
        action = pure_pursuit.track_traj(start, timeStepRef)
        lastState,action,newState,termFlag = sim.controlLoopStep(action.cpu().squeeze())
        # plt.scatter(newState[0],newState[1],color='black',zorder=-1)
        # plt.pause(0.01)

if __name__ == "__main__":
    main("nominal_dec_8")