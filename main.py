import yaml
import os
import pybullet as p
from utils.planarRobotState import convert_planar
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.RandomRockyTerrain import RandomRockyTerrain
from clifford_pybullet.SimController import SimController
from clifford_pybullet.utils.genParam import genParam

import torch
import numpy as np
from control.PurePursuit import PurePursuit
from plotting.PurePursuitPlot import PurePursuitPlot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# TODO: are we able to compute everything in quaternions, and only use Euler angles for plotting?

def main(data_dir):
    plt.ion()
    MAX_TIME_STEPS = 200
    N_TRAJECTORIES_TO_COLLECT = 10
    params = {
        'dataDir':      data_dir,
        'controls':     yaml.safe_load(open(os.path.join(data_dir,'config/controls.yaml'),'r')),
        'robotRange':   yaml.safe_load(open(os.path.join(data_dir,'config/robotRange.yaml'),'r')),
        'sim':          yaml.safe_load(open(os.path.join(data_dir,'config/sim.yaml'),'r')),
        'terrain':      yaml.safe_load(open(os.path.join(data_dir,'config/terrain.yaml'),'r')),
        'train':        yaml.safe_load(open(os.path.join(data_dir,'config/train.yaml'),'r')),
        }

    # PyBullet + robot setup
    physicsClientID = p.connect(p.GUI)
    #physicsClientID = p.connect(p.DIRECT)
    robot = CliffordRobot(physicsClientID)
    
    # Terrain
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

    # Default params file for robotRange = robotRange.yaml
    robotParams = genParam(params['robotRange'], gen_mean=params['train']['useNominal'])
    sim.robot.setParams(robotParams)

    # Experiment plot
    car_plot = PurePursuitPlot(
        params['controls']['purePursuit']['wheel_base'],
        params['controls']['purePursuit']['lookahead']
    )

    # Control
    pure_pursuit = PurePursuit(**params['controls']['purePursuit'])

    d_states = []
    d_previous_states = []
    d_actions = []

    for i_traj in range(N_TRAJECTORIES_TO_COLLECT):
        print("CURRENTLY COLLECTING TRAJECTORY {}.".format(i_traj))
        # Reset pure pursuit parameters (TODO: tidy)
        pure_pursuit.lastFoundIndex = 1
        # Reset robot pose
        sim.resetRobot(pose=((0,0),(0, 0, np.sqrt(2)/2, np.sqrt(2)/2)))

        ref_traj = None
        while ref_traj is None:
            start = torch.tensor([0, 0, 0])

            # Sample code for sampling (lol)
            goalRange = torch.tensor([[-11, 11],[-11, 11],[-3.14, 3.14]])
            target = torch.rand(goalRange.shape[0]) * (goalRange[:, 1] - goalRange[:, 0]) + goalRange[:, 0]
            ref_traj, _ = pure_pursuit.gen_traj(start, target, 0.1)

            # <<CIRCLE TRAJECTORY>>
            # radius = 4.0
            # ref_traj = pure_pursuit.gen_traj_circle(-radius, 0, radius, 1000)

        # Plot feasible reference trajectory and goal:
        car_plot.plot_ref_traj(ref_traj[:,:2])
        car_plot.plot_car(ref_traj[-1, :], is_goal=True)

        current_state = start
        goal_state = ref_traj[-1, :]
        plt.pause(0.01)

        # TODO: move into data collection.
        for t in range(MAX_TIME_STEPS):
            # We're close enough to the goal. Move to goal.
            if torch.norm(current_state[:2] - goal_state[:2]) < 2e-1 and t > 0.3*MAX_TIME_STEPS:
                print("reached")
                break
            else:
                ref_state = pure_pursuit.get_lookahead_state(current_state, ref_traj)

            # [throttle, front_angle, rear_angle]
            action = pure_pursuit.track_traj(current_state, ref_state)
            lastState, action, current_state, termFlag = sim.controlLoopStep(action.cpu().squeeze())

            d_states.append(current_state)
            d_previous_states.append(lastState)
            d_actions.append(action)

            car_plot.plot_car(current_state)
            car_plot.plot_pure_pursuit(current_state[:2], ref_state)            
            plt.pause(0.01)

    print("Done with timesteps.")
    d_states = torch.stack(d_states)
    d_previous_states = torch.stack(d_previous_states)
    d_actions = torch.stack(d_actions)
    torch.save(d_states, 'd_states.pt')
    torch.save(d_previous_states, 'd_previous_states.pt')
    torch.save(d_actions, 'd_actions.pt')
    print("Dataset saved.")

if __name__ == "__main__":
    main("nominal_dec_8")
