import argparse
import yaml
import socketio
import torch
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pickle

from utils.planarRobotState import convertPlanar,getRelativeState,planarRobotState,getLocalMap
from utils.tensorIO import fromTensor,toTensor,toStateDict
from utils.StatusPrint import StatusPrint

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
            else:
                robot = CliffordRobot(self.physicsClientId)
                terrain = Terrain(self.params['terrain'], physicsClientId=self.physicsClientId)
                # TODO: check sim controller
                self.sim = SimController(
                    robot,
                    terrain,
                    self.params['sim'],
                    physicsClientId=self.physicsClientId,
                    realtime=False,
                    stateProcessor=convertPlanar
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

            for key in results:
                results[key] = fromTensor(results[key])

            StatusPrint('sending results')
            sio.emit('results',results)
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

# def trajTrackDataCollect(self):
#         pp = purePursuit(**self.params['controls']['purePursuit'])
#         #rrt = RRT(pp,**self.params['controls']['RRT'])
#         planner = RRTStar(pp,**self.params['controls']['RRT'])

#         data = {'states':[],
#                 'actions':[],
#                 'trajRefs':[],
#                 'worldMap':[],
#                 'worldMapBounds':[]}

#         # simulation policy and collect data
#         termFlag = True
#         usableSimSteps = 0
#         stepCount = 0
#         while usableSimSteps < self.params['train']['simStepsPerBatch']:
#             #if not self.connected:
#             if self.connectionStatus != 2:
#                 return None
#             # if new trajectory
#             if termFlag:
#                 # reset robot
#                 StatusPrint('resetting robot')
#                 self.sim.terrain.generate()
#                 self.sim.resetRobot()
#                 lastState,action,newState,_ = self.sim.controlLoopStep(torch.zeros(self.params['controls']['actionDim']))
#                 #xBounds = [self.sim.terrain.gridX.min(),self.sim.terrain.gridX.max()]
#                 #yBounds = [self.sim.terrain.gridY.min(),self.sim.terrain.gridY.max()]
#                 xBounds = [0,self.sim.terrain.gridX.max()]
#                 yBounds = [self.sim.terrain.gridY.min()/2.0,self.sim.terrain.gridY.max()/2.0]
#                 traj = torch.tensor([])
#                 while traj.shape[0] < self.params['train']['trainPredSeqLen']:
#                     randGoal = planner.sample(xBounds,yBounds)[:2]
#                     traj = planner.search(newState,randGoal,xBounds,yBounds)
#                     #goal = torch.tensor([4,2,0])
#                     #traj,_ = pp.calcSteerTraj(newState,goal,0.05)

#                 completeTraj = traj
#                 if self.useGUI:
#                     self.sim.bufferConstHeightLine(completeTraj[:,:2],0.2,alpha=0.5,lineWidth=5,color=(0,1,0),flush=True)

#                 # prepare data
#                 for key in data:
#                     data[key].append(torch.tensor([]))

#                 data['worldMap'][-1] = torch.tensor(self.sim.terrain.gridZ).float().unsqueeze(0)
#                 data['worldMapBounds'][-1] = self.sim.terrain.mapBounds.unsqueeze(0)

#             lookAhead = self.params['controls']['purePursuit']['lookAhead']
#             refEnd = min(lookAhead,len(traj))
#             indices = list(range(refEnd))+[refEnd-1]*max(lookAhead-refEnd,0)
#             trajRef = traj[indices]
#             action = pp.trackTraj(newState,trajRef)
#             traj = traj[1:]

#             if self.useGUI:
#                 self.sim.bufferConstHeightLine(completeTraj[:,:2],0.2,alpha=0.5,lineWidth=5,color=(0,1,0))
#                 self.sim.bufferTerrainLine(torch.cat((trajRef[-1,:2],torch.zeros(1)),dim=-1),
#                                     torch.cat((trajRef[-1,:2],torch.ones(1)*10),dim=-1),
#                                     lineWidth=5, color= (1,0,0),flush=True)
            

#             # drive robot
#             lastState,action,newState,termFlag = self.sim.controlLoopStep(action.cpu().squeeze())

#             data['states'][-1] = torch.cat((data['states'][-1],lastState.unsqueeze(0)),dim=0)
#             data['actions'][-1] = torch.cat((data['actions'][-1],action.unsqueeze(0)),dim=0)
#             data['trajRefs'][-1] = torch.cat((data['trajRefs'][-1],trajRef.unsqueeze(0)),dim=0)
#             stepCount += 1

#             distToTrackTarget = torch.norm(trajRef[-1,:2] - newState[:2])
#             termFlag = termFlag or traj.shape[0] == 0 or distToTrackTarget > self.params['sim']['trackerTermDist']

#             if data['actions'][-1].shape[0] >= self.params['train']['trainPredSeqLen']:
#                 usableSimSteps = stepCount

#             if termFlag:
#                 if data['actions'][-1].shape[0] < self.params['train']['trainPredSeqLen']:
#                     stepCount -= data['actions'][-1].shape[0]
#                     for key in data:
#                         data[key].pop()
#                 else:
#                     data['states'][-1] = torch.cat((data['states'][-1],newState.unsqueeze(0)),dim=0)
#             StatusPrint('steps: ', stepCount,usableSimSteps,isTemp=True)

#         # finish processing data
#         if data['states'][-1].shape[0] == data['actions'][-1].shape[0]:
#             data['states'][-1] = torch.cat((data['states'][-1],newState.unsqueeze(0)),dim=0)

#         for key in ['actions','trajRefs']:
#             for i in range(len(data[key])):
#                 data[key][i] = torch.cat((data[key][i],torch.ones_like(data[key][i][-1:])*torch.inf),dim=0)

#         for key in data:
#             data[key] = torch.cat(data[key],dim=0)

#         #self.genPredictionData(data)

#         return data

    def trajTrackDataCollect(self):
        pp = purePursuit(**self.params['controls']['purePursuit'])
        planner = RRTStar(pp, **self.params['controls']['RRT'])

        data = {'states':[],
                'actions':[],
                'trajRefs':[],
                'worldMap':[],
                'worldMapBounds':[]}

        # simulation policy and collect data
        termFlag = True
        usableSimSteps = 0
        stepCount = 0
        
        # Data collection loop
        while usableSimSteps < self.params['train']['simStepsPerBatch']:
            # Only proceed with data collection if robot parameters have been specified.
            if self.connectionStatus != self.STATUS_PARAMS_SET:
                return None
            
            # When termFlag is True, we have hit a stop condition. Create a new trajectory.
            if termFlag:
                StatusPrint('resetting robot')
                
                # TODO: uncomment if generating terrain
                # self.sim.terrain.generate()
                self.sim.resetRobot()
                
                lastState, action, newState, _ = self.sim.controlLoopStep(torch.zeros(self.params['controls']['actionDim']))
                
                #xBounds = [self.sim.terrain.gridX.min(),self.sim.terrain.gridX.max()]
                #yBounds = [self.sim.terrain.gridY.min(),self.sim.terrain.gridY.max()]
                xBounds = [0, self.sim.terrain.gridX.max()]
                yBounds = [self.sim.terrain.gridY.min() / 2.0, self.sim.terrain.gridY.max() / 2.0]
                traj = torch.tensor([])
                
                # TODO: replace this whole chunk with your own random generation code.
                # While trajectory is less than the minimum trainPredSeqLen, sample a new goal and find a path to it.
                while traj.shape[0] < self.params['train']['trainPredSeqLen']:
                    randGoal = planner.sample(xBounds,yBounds)[:2]
                    traj = planner.search(newState,randGoal,xBounds,yBounds)

                # [DEBUG] Draw trajectory
                completeTraj = traj
                if self.useGUI:
                    self.sim.bufferConstHeightLine(
                        completeTraj[:,:2],
                        0.2,
                        alpha=0.5,
                        lineWidth=5,
                        color=(0,1,0),
                        flush=True
                    )

                # Insert an empty tensor at the back of the list in order to populate with new data.
                for key in data:
                    data[key].append(torch.tensor([]))

                data['worldMap'][-1] = torch.tensor(self.sim.terrain.gridZ).float().unsqueeze(0)
                data['worldMapBounds'][-1] = self.sim.terrain.mapBounds.unsqueeze(0)

            # TODO: replace this whole section with MPC.
            # trajRef is actually trajectory coordinates that will always be lookAhead (desired K waypoints). It repeats
            # if we're near the end of the trajectory.
            lookAhead = self.params['controls']['purePursuit']['lookAhead']
            refEnd = min(lookAhead, len(traj))
            indices = list(range(refEnd)) + [refEnd - 1] * max(lookAhead - refEnd, 0)
            trajRef = traj[indices]
            action = pp.trackTraj(newState,trajRef)

            
            traj = traj[1:]

            if self.useGUI:
                self.sim.bufferConstHeightLine(completeTraj[:,:2], 0.2, alpha=0.5, lineWidth=5, color=(0,1,0))
                self.sim.bufferTerrainLine(torch.cat((trajRef[-1,:2],torch.zeros(1)),dim=-1),
                                    torch.cat((trajRef[-1,:2],torch.ones(1)*10),dim=-1),
                                    lineWidth=5, color= (1,0,0),flush=True)
            

            # drive robot
            # TODO: To change into commandInRealUnits to use MPC for data collection
            lastState, action, newState, termFlag = self.sim.controlLoopStep(action.cpu().squeeze())

            data['states'][-1]      = torch.cat((data['states'][-1], lastState.unsqueeze(0)), dim=0)
            data['actions'][-1]     = torch.cat((data['actions'][-1], action.unsqueeze(0)), dim=0)
            data['trajRefs'][-1]    = torch.cat((data['trajRefs'][-1], trajRef.unsqueeze(0)), dim=0)
            stepCount += 1

            distToTrackTarget = torch.norm(trajRef[-1,:2] - newState[:2])
            termFlag = termFlag or traj.shape[0] == 0 or distToTrackTarget > self.params['sim']['trackerTermDist']

            if data['actions'][-1].shape[0] >= self.params['train']['trainPredSeqLen']:
                usableSimSteps = stepCount

            if termFlag:
                if data['actions'][-1].shape[0] < self.params['train']['trainPredSeqLen']:
                    stepCount -= data['actions'][-1].shape[0]
                    for key in data:
                        data[key].pop()
                else:
                    data['states'][-1] = torch.cat((data['states'][-1],newState.unsqueeze(0)),dim=0)

            StatusPrint('steps: {}\t usableSimSteps: {}'.format(stepCount, usableSimSteps), isTemp=True)

        # Tidying up data
        if data['states'][-1].shape[0] == data['actions'][-1].shape[0]:
            data['states'][-1] = torch.cat((data['states'][-1], newState.unsqueeze(0)), dim=0)

        # Padding a row of infinity to the end of each refTraj and action.
        for key in ['actions','trajRefs']:
            for i in range(len(data[key])):
                data[key][i] = torch.cat((data[key][i], torch.ones_like(data[key][i][-1:]) * torch.inf), dim=0)

        # Converting all data into tensors.
        for key in data:
            data[key] = torch.cat(data[key], dim=0)

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
        targetTransitions = getRelativeState(states,states.roll(-1,dims=-2))
        noisyTransitions = getRelativeState(noisyStates,noisyStates.roll(-1,dims=-2))

        # transition at end of trajectory is invalid
        trajEnd = actions[:,0] == torch.inf
        noisyTransitions[trajEnd] = torch.inf

        # transition at start of trajectory should be stationary
        noisyPriorTransitions = noisyTransitions.roll(1,dims=-2)
        trajStart = noisyPriorTransitions[:,0] == torch.inf
        noisyPriorTransitions[trajStart] = getRelativeState(states[trajStart],states[trajStart])

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
