import argparse
import yaml
import socketio
import torch
from tensordict import TensorDict as td
import time
import numpy as np
import pybullet as p
import os

from utils.planarRobotState import convert_planar_world_frame_with_vel
from utils.StatusPrint import StatusPrint

# MPC
from utils.MPCConfigLoader import MPCConfigLoader
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.TrajProc import TrajProc
from TrajProc.controls.MPC import MPC
from utils.simulation.simulation_utils import init_u_bar, plan_mpc_step, create_random_track, state_derivative

# simulation
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.SimController import SimController
from clifford_pybullet.Terrain import Terrain

def close_trajectory(data, final_state):
    """
    Ends the current trajectory in data: appends its final state and an x_dot row for it (velocities at
    the final state, v_dot = 0), so states and x_dot have one row per action row including the inf row.
    """
    final_xdot = torch.cat((final_state[4:6], torch.zeros(1, dtype=final_state.dtype), final_state[6:]))
    data['states'][-1] = torch.cat((data['states'][-1], final_state[:4].unsqueeze(0)), dim=0)
    data['xdot'][-1] = torch.cat((data['xdot'][-1], final_xdot.unsqueeze(0)), dim=0)


def collect_tracking_data(sim, tp, mpc, mpc_params, params, keep_running=lambda: True):
    """
    Drives random tracks with the MPC until simStepsPerBatch usable steps are collected. A new random
    track starts after every reset (robot stuck, flipped or out of bounds).

    Returns:
        dict of tensors (or None if keep_running() turns False):
            states  (N, 4)  [x, y, v, yaw] at the start of each step, plus the final state of each trajectory
            actions (N, 3)  applied MPC action [a, delta_f, delta_r]; an inf row ends each trajectory
            xdot    (N, 4)  x_dot target of each step (see state_derivative)
    """
    x_sim = [np.array([0.0, 0.0, 0.0, 0.0])]
    u_bar_start = init_u_bar(mpc_params)
    data = {'states': [], 'actions': [], 'xdot': []}

    termFlag = True
    usableSimSteps = 0
    stepCount = 0
    while usableSimSteps < params['train']['simStepsPerBatch']:
        if not keep_running():
            return None

        # Start a new trajectory on a new random track (at the start, and after the robot was reset).
        if termFlag:
            StatusPrint('resetting robot')
            sim.resetRobot()
            _, _, newState, _ = sim.controlLoopStep(
                torch.zeros(params['controls']['actionDim']),
                commandInRealUnits=True,
            )
            x_sim.append(newState[:4].numpy())  # plan from where the robot is after the reset
            track = create_random_track(tp, params['terrain'], 6)
            tp.lastIndex = 0
            for key in data:
                data[key].append(torch.tensor([]))

        # u_mpc is the action that is applied this step
        u_sim_opt, U_mpc, _ = plan_mpc_step(x_sim[-1], u_bar_start, track, tp, mpc, mpc_params)
        u_mpc = U_mpc[:, 0]

        previous_state, _, current_state, termFlag = sim.controlLoopStep(
            u_sim_opt.numpy(),
            commandInRealUnits=True,
        )
        # current_state = [x, y, vel_mag, yaw, xdot, ydot, yaw_dot]
        previous_xdot = state_derivative(previous_state, current_state, params['mpc']['dt'])
        x_sim.append(current_state[:4].numpy())

        data['states'][-1]  = torch.cat((data['states'][-1],  previous_state[:4].unsqueeze(0)), dim=0)
        data['actions'][-1] = torch.cat((data['actions'][-1], torch.tensor(u_mpc).unsqueeze(0)), dim=0)
        data['xdot'][-1]    = torch.cat((data['xdot'][-1],    previous_xdot.unsqueeze(0)), dim=0)
        stepCount += 1

        if data['actions'][-1].shape[0] >= params['train']['trainPredSeqLen']:
            usableSimSteps = stepCount

        if termFlag:
            # A trajectory shorter than trainPredSeqLen has no training window: discard it.
            if data['actions'][-1].shape[0] < params['train']['trainPredSeqLen']:
                stepCount -= data['actions'][-1].shape[0]
                for key in data:
                    data[key].pop()
            else:
                close_trajectory(data, current_state)

        StatusPrint('steps: {}\t usableSimSteps: {}'.format(stepCount, usableSimSteps), isTemp=True)

    # The last trajectory is still running when the batch is full: close it like a terminated one.
    if not termFlag:
        close_trajectory(data, current_state)

    # An inf action row marks the end of each trajectory (there is no action at its final state).
    data['actions'] = [torch.cat((a, torch.ones_like(a[-1:]) * torch.inf), dim=0) for a in data['actions']]

    for key in data:
        data[key] = torch.cat(data[key], dim=0)
    return data


class SimClient(object):
    STATUS_DISCONNECTED = 0
    STATUS_CONNECTED = 1
    STATUS_PARAMS_SET = 2

    def __init__(self, server_url, useGUI=False, altControls=None):
        self.useGUI = useGUI
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.physicsClientId = p.connect(p.GUI if self.useGUI else p.DIRECT)
        sio = socketio.Client()
        self.dataBuffer = {}
        self.taskRunning = False
        self.connectionStatus = self.STATUS_DISCONNECTED
        #self.connected = False

        @sio.on('connect')
        def connect():
            StatusPrint('[Client] Connected to server.')
            self.connectionStatus = self.STATUS_CONNECTED

        @sio.on('disconnect')
        def disconnect():
            StatusPrint('[Client] Disconnected from server.')
            #self.connected = False
            self.connectionStatus = self.STATUS_DISCONNECTED

        @sio.on('set_params')
        def setParams(data):
            """
            In the server, in handleSimConnections(), the server will emit set_params with self.params as data when 
            it receives 'connect'. 'connect' is automatically sent by Socket.IO as soon as TCP/WebSocket handshake 
            succeeds.

            setParams also initializes the robot and the terrain that will be used according to the parameters it 
            receives.

            Emits 'sim_ready' when the robot and terrain has been set-up successfully.

            Params:
                data: [dict] containing loaded YAMLs of all the configuration files involved for simulation.
            """
            StatusPrint('received parameters')
            # make sure task stopped
            while self.connectionStatus == self.STATUS_DISCONNECTED:
                time.sleep(1)
            
            self.connectionStatus = self.STATUS_CONNECTED

            while self.taskRunning:
                time.sleep(1)
            
            self.connectionStatus = self.STATUS_PARAMS_SET
            
            self.params = data
            StatusPrint('parameters set')
            
            if not altControls is None:
                self.params['controls'] = yaml.safe_load(open(altControls))
        
            # load simulation robot and terrain
            if hasattr(self, 'sim'):
                StatusPrint('resetting sim params')
                self.sim.terrain.setParams(self.params['terrain'])
                self.sim.setParams(self.params['sim'])
            
            # Initialization code
            else:
                # [MPC] For MPC / casadi
                mpc_config_loader = MPCConfigLoader(self.params['mpc'])
                self.mpc_params = mpc_config_loader.construct_casadi_params()
                self.cs_kbm = csDSKBM(self.mpc_params)
                self.tp = TrajProc()
                self.mpc = MPC(self.mpc_params, self.cs_kbm)
        
                robot = CliffordRobot(physicsClientId=self.physicsClientId)
                terrain = Terrain(self.params['terrain'], physicsClientId=self.physicsClientId)
                self.sim = SimController(
                    robot,
                    terrain,
                    self.params['sim'],
                    physicsClientId=self.physicsClientId,
                    realtime=False,
                    stateProcessor=convert_planar_world_frame_with_vel
                    # stateProcessor=lambda state: convert_planar(state, position_only=True)
                )

            sio.emit('sim_ready')
            if self.useGUI:
                self.sim.camFollowBot=True
                self.sim.camDist = 1
                self.sim.camPitch = -45

        @sio.on('run_sim')
        def runSim(task):
            """
            Params:
                task: [dict] e.g.,
                    {'key': key, 'robotParam':self.robotParams[key], 'type':'trackTraj'}
            """
            StatusPrint("[Client] Running sim for robot {}\t Task: {}".format(task['key'], task['type']))
            self.taskRunning = True

            # set the parameters of the robot
            self.sim.robot.setParams(task['robotParam'])

            StatusPrint('[Client] Fric: ',task['robotParam']['tireContact']['lateralFriction'][0])

            if task['type'] == 'trackTraj':
                # collect training data by running the traj tracker policy
                results = self.trajTrackDataCollect()

            #if not self.connected:
            if self.connectionStatus != 2:
                self.taskRunning = False
                StatusPrint('disconnected task aborted')
                return

            # for key in results:
            #     results[key] = fromTensor(results[key])
            results_td = td(results)
            tensor_dir = os.path.join(self.params['dataDir'], "temp")
            
            if not os.path.exists(tensor_dir):
                os.makedirs(tensor_dir)
            
            tensor_fp = os.path.join(tensor_dir, f"{sio.sid}_data")
            results_td.memmap(tensor_fp)
            
            StatusPrint('sending results')
            sio.emit('results', tensor_fp)
            StatusPrint('results sent')

            self.taskRunning = False

        def connect_server():
            notConnected = True
            while notConnected:
                notConnected = False
                try:
                    sio.connect(server_url)
                except:
                    StatusPrint('connecting...')
                    time.sleep(1)
                    notConnected=True
        connect_server()
        
        #idleStartTime = time.time()
        while True:
            """
            if self.taskRunning or not self.connected:
                idleStartTime = time.time()
            else:
                idleTime = time.time()-idleStartTime
                StatusPrint('idle: ',int(idleTime),isTemp=True)
                if idleTime > 60:
                    sio.disconnect()
                    connect_server()
                    idleStartTime = time.time()
                    """
            time.sleep(1)

    def trajTrackDataCollect(self):
        return collect_tracking_data(self.sim, self.tp, self.mpc, self.mpc_params, self.params,
                                     keep_running=lambda: self.connectionStatus == self.STATUS_PARAMS_SET)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='simulation to collect data')
    parser.add_argument('--url',type=str,nargs='?',default='http://localhost:8080',help='server url')
    parser.add_argument('--gui',action='store_true',help='use gui')
    parser.add_argument('--altControls',type=str,nargs='?',default=None,help='alternative controls params to use')
    args = parser.parse_args()
    SimClient(args.url,args.gui,args.altControls)
