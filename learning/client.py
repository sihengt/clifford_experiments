import argparse
import yaml
import socketio
import torch
from tensordict import TensorDict as td
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pickle
import os

from utils.planarRobotState import convert_planar, get_relative_state, getLocalMap, convert_planar_world_frame, convert_planar_world_frame_with_vel
from utils.tensorIO import fromTensor,toTensor,toStateDict
from utils.StatusPrint import StatusPrint

# MPC
from MPCConfigLoader import MPCConfigLoader
from TrajProc.models.DSKBM import csDSKBM
from TrajProc.TrajProc import TrajProc
from TrajProc.controls.MPC import MPC
from utils.simulation.simulation_utils import init_u_bar, \
    init_logger, plan_mpc_step, step_sim_and_log, create_debug_track, create_random_track, init_pybullet_sim, mpc_worker

# simulation
from clifford_pybullet.CliffordRobot import CliffordRobot
from clifford_pybullet.SimController import SimController
from clifford_pybullet.Terrain import Terrain

# tracking policy
from utils.purePursuit import purePursuit
from utils.RRTStar import RRTStar

class SimClient(object):
    STATUS_DISCONNECTED = 0
    STATUS_CONNECTED = 1
    STATUS_PARAMS_SET = 2

    def __init__(self, server_url, useGUI=False, plotMPPI = False,altControls=None):
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
        x_sim = []
        u_sim = []
        
        # Create starting conditions x0
        x_sim.append(np.array([0.0, 0.0, 0.0, 0.0]).T)

        # Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
        u_bar_start = init_u_bar(self.mpc_params)

        data = {'states':[],
                'actions':[],
                'xdot':[]}

        # simulation policy and collect data
        termFlag = True
        usableSimSteps = 0
        stepCount = 0
        
        # Data collection loop
        while usableSimSteps < self.params['train']['simStepsPerBatch']:
            # Only proceed with data collection if robot parameters have been specified.
            if self.connectionStatus != self.STATUS_PARAMS_SET:
                return None
            
            # When termFlag is True:
            #   1. We're initializing
            #   2. We've hit a stop condition and need to re-initialize.
            if termFlag:
                StatusPrint('resetting robot')
                self.sim.resetRobot()
                lastState, action, newState, _ = self.sim.controlLoopStep(
                    torch.zeros(self.params['controls']['actionDim']),
                    commandInRealUnits=True,
                )
                
                xBounds = [0, self.sim.terrain.gridX.max()]
                yBounds = [self.sim.terrain.gridY.min() / 2.0, self.sim.terrain.gridY.max() / 2.0]
                traj = torch.tensor([])

                # TODO: eventually replace with randomly sampled track
                track = create_debug_track(self.tp, self.params['terrain'])

                # Insert an empty tensor at the back of the list in order to populate with new data.
                for key in data:
                    data[key].append(torch.tensor([]))

            # TODO: replace this whole section with MPC.
            # [MPC]
            u_sim_opt, u_mpc, l_ref_idx = plan_mpc_step(
                x_sim[-1],
                u_bar_start,
                track,
                self.tp,
                self.mpc,
                self.mpc_params,
                return_mpc_action=True
            )
            
            # u_k
            u_sim.append(u_sim_opt.numpy())
            
            previous_state, action, current_state, termFlag = self.sim.controlLoopStep(
                u_sim_opt.numpy(),
                commandInRealUnits=True,
            )
            
            # current_state = [x, y, vel_mag, yaw, xdot, ydot, yaw_dot]
            # x_kp1
            xy_dot      = current_state[4:6]
            a_dot       = torch.tensor([(current_state[2] - previous_state[2]) / self.params['mpc']['dt']])
            theta_dot   = current_state[6:]
            previous_xdot = torch.cat((xy_dot, a_dot, theta_dot))

            x_sim.append(current_state[:4].numpy())

            data['states'][-1]      = torch.cat((data['states'][-1],    previous_state[:4].unsqueeze(0)), dim=0)
            data['actions'][-1]     = torch.cat((data['actions'][-1],   torch.tensor(u_mpc).unsqueeze(0)), dim=0)
            data['xdot'][-1]        = torch.cat((data['xdot'][-1],      previous_xdot.unsqueeze(0)), dim=0)
            stepCount += 1

            # distToTrackTarget = torch.norm(trajRef[-1,:2] - newState[:2])
            # termFlag = termFlag or traj.shape[0] == 0 or distToTrackTarget > self.params['sim']['trackerTermDist']
            
            if data['actions'][-1].shape[0] >= self.params['train']['trainPredSeqLen']:
                usableSimSteps = stepCount

            if termFlag:
                # If we're terminating before we collected at least trainPredSeqLen, the data is worthless, discard.
                if data['actions'][-1].shape[0] < self.params['train']['trainPredSeqLen']:
                    stepCount -= data['actions'][-1].shape[0]
                    for key in data:
                        data[key].pop()
                # else:
                #     xy_dot      = current_state[4:6]
                #     a_dot       = torch.tensor([0.0])
                #     theta_dot   = current_state[6:]
                #     final_xdot = torch.cat((xy_dot, a_dot, theta_dot))
                #     data['states'][-1] = torch.cat((data['states'][-1], current_state[:4].unsqueeze(0)), dim=0)
                #     data['xdot'][-1] = torch.cat((data['xdot'][-1], final_xdot.unsqueeze(0)), dim=0)

            StatusPrint('steps: {}\t usableSimSteps: {}'.format(stepCount, usableSimSteps), isTemp=True)

        # Tidying up data
  
        # Padding a row of infinity to the end of each refTraj and action.
            # len(data['actions']) is the number of trajectories we collected.
            # data[key][i][-1:] is number of actions I bet (YEP)
            # data['actions'][i] is the ith trajectory collected in this run.
            # Since we're adding the final state onto training, we add inf to mark that there's no action taken there.
        
        # for key in ['actions','trajRefs']:
        for key in ['actions']:
            for i in range(len(data[key])):
                data[key][i] = torch.cat((data[key][i], torch.ones_like(data[key][i][-1:]) * torch.inf), dim=0)

        # Converting all data into tensors.
        for key in data:
            data[key] = torch.cat(data[key], dim=0)

        # x_sim = np.array(x_sim)
        # plt.scatter(x_sim[:, 0], x_sim[:, 1])
        # plt.savefig('debug_trajectory.png')
        # breakpoint()

        return data

    def genPredictionData(self,data):
        states = data['states']
        actions = data['actions']
        worldMap = data['worldMap']
        worldMapBounds = data['worldMapBounds']

        stateNoise = torch.randn(data['states'].shape)*torch.tensor(self.params['train']['stateNoise'])
        noisyStates = data['states'] + stateNoise

        # generate maps
        mapIndex = (actions[:,0] == torch.inf).cumsum(dim=0).tolist()
        mapIndex = [0]+mapIndex[:-1]
        noisyLocalMaps = getLocalMap(noisyStates,
                                worldMap[mapIndex,:].unsqueeze(1),
                                worldMapBounds[mapIndex,:],
                                self.params['network']['localMap'])

        # generate state transitions
        targetTransitions = get_relative_state(states,states.roll(-1,dims=-2))
        noisyTransitions  = get_relative_state(noisyStates,noisyStates.roll(-1,dims=-2))

        # transition at end of trajectory is invalid
        trajEnd = actions[:,0] == torch.inf
        noisyTransitions[trajEnd] = torch.inf

        # transition at start of trajectory should be stationary
        noisyPriorTransitions = noisyTransitions.roll(1,dims=-2)
        trajStart = noisyPriorTransitions[:,0] == torch.inf
        noisyPriorTransitions[trajStart] = get_relative_state(states[trajStart],states[trajStart])

        # add to data
        data['targetTransitions'] = targetTransitions
        data['noisyTransitions'] = noisyTransitions
        data['noisyPriorTransitions'] = noisyPriorTransitions
        data['noisyLocalMaps'] = noisyLocalMaps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='simulation to collect data')
    parser.add_argument('--url',type=str,nargs='?',default='http://localhost:8080',help='server url')
    parser.add_argument('--gui',action='store_true',help='use gui')
    parser.add_argument('--plot',action='store_true',help='plot mppi')
    parser.add_argument('--altControls',type=str,nargs='?',default=None,help='alternative controls params to use')
    args = parser.parse_args()
    SimClient(args.url,args.gui,args.plot,args.altControls)
