import os
import shutil
import argparse
import yaml
import torch
from tensordict import TensorDict as td
from aiohttp import web
import socketio
import asyncio
import time
from utils.tensorIO import fromTensor, toTensor, fromStateDict
from .architecture.StatusPrint import StatusPrint
from .ModelTrainer import ModelTrainer
from collections import defaultdict,deque

# from msTrainer import msModelTrainer

from clifford_pybullet.utils.genParam import genParam

import multiprocessing

DATA_COLLECTED = ('states','actions','trajRefs','worldMap','worldMapBounds')
                #'targetTransitions','noisyTransitions','noisyPriorTransitions','noisyLocalMaps')

class Server(object):
    def __init__(self,args):
        if args.dataDir is None:
            args.dataDir = args.trainDir
        else:
            for fn in ['robotRange.yaml','sim.yaml','terrain.yaml']:
                shutil.copy(os.path.join(args.dataDir,'config',fn),
                            os.path.join(args.trainDir,'config',fn))
        self.params = {'trainDir': args.trainDir,
            'dataDir': args.dataDir,
            'train': yaml.safe_load(open(os.path.join(args.trainDir,'config/train.yaml'),'r')),
            'network': yaml.safe_load(open(os.path.join(args.trainDir,'config/network.yaml'),'r')),
            'controls': yaml.safe_load(open(os.path.join(args.trainDir,'config/controls.yaml'),'r')),
            'robotRange': yaml.safe_load(open(os.path.join(args.trainDir,'config/robotRange.yaml'),'r')),
            'sim': yaml.safe_load(open(os.path.join(args.trainDir,'config/sim.yaml'),'r')),
            'terrain': yaml.safe_load(open(os.path.join(args.trainDir,'config/terrain.yaml'),'r')),
            }
        
        self.simTaskQueue = deque()         # Tasks waiting to run
        self.idleSims = set()               # Socket IDs that are ready
        self.runningSims = {}               # sid->task currently executing
        self.activeCollectingKeys = set()   # robot indices still gathering data
        
        # Starts simulation server in the background
        asyncio.ensure_future(self.runSimServer())

        # either continue previous training or start over
        robotMetaFn = os.path.join(self.params['dataDir'],'robotMeta.yaml')
        trajDataDir = os.path.join(self.params['dataDir'],'trajData')
        if self.params['train']['useOldData'] and os.path.exists(robotMetaFn):
            self.robotParams = yaml.safe_load(open(robotMetaFn,'r'))
            for key in self.robotParams.keys():
                self.activeCollectingKeys.add(key)
                self.processTrajData(key)
        else:
            # remove old data and start over
            if os.path.exists(robotMetaFn):
                os.remove(robotMetaFn)
            if os.path.exists(trajDataDir):
                shutil.rmtree(trajDataDir)
            self.robotParams = {}
        
        loop = asyncio.get_event_loop()
        numFinishedRobots = len(self.robotParams) - len(self.activeCollectingKeys)
        if numFinishedRobots < self.params['train']['numParallelRobots']:
            loop.run_until_complete(self.addBatchToSimQueue())
        
        # start training
        self.training = ModelTrainer(self, self.params)
        asyncio.ensure_future(self.training.run_training())

        # Program is now driven by events (socket messages, task completions, training-loop timers etc.)
        loop.run_forever()

    async def runSimServer(self):
        """
        Initializes server, mounts it inside aiohttp application, registers callbacks, sets up runner, binds 
        runner to a concrete socket / port, and asks event loop to start listening.
        """
        StatusPrint('Server Starting.')
        self.sio = socketio.AsyncServer(max_http_buffer_size=10e6)
        app = web.Application()
        
        # Mounts Socket.IO server inside the aiohttp application
        self.sio.attach(app)

        # Registers event callbacks:
        self.handleSimConnections()
        self.handleSimResults()
        
        # AppRunner wraps the app without blocking
        runner = web.AppRunner(app)
        await runner.setup()

        # Binds the runner to a concrete socket/port, and asks the event loop to start listening.
        site = web.TCPSite(runner)
        await site.start()

    async def taskProcessor(self):
        if len(self.idleSims) == 0:
            return
        if len(self.simTaskQueue) == 0:
            return
        
        # Gets a task from the simTaskQueue, and an idle sim, and assigns the task to the idle sim.
        task = self.simTaskQueue.popleft()
        sid = self.idleSims.pop()
        self.runningSims[sid] = task
        StatusPrint('sending sim task ',sid)
        await self.sio.emit('run_sim', task, room=sid)
        StatusPrint('sim task sent')
        await self.taskProcessor()

    async def addBatchToSimQueue(self):
        StatusPrint('adding new sim tasks')
        while len(self.activeCollectingKeys) < self.params['train']['numParallelRobots']:
            if len(self.robotParams) >= self.params['train']['totalBots']:
                break
            newKey = self.generateRobot()
            await asyncio.sleep(0.01)
        
        # Overwrites 'robotMeta.yaml' with the current list of robot parameters.
        yaml.dump(self.robotParams, open(os.path.join(self.params['dataDir'], 'robotMeta.yaml'), 'w'))

        collectingKeys = tuple(self.activeCollectingKeys)
        for key in self.activeCollectingKeys:
            self.simTaskQueue.append({'key': key,
                                      'robotParam': self.robotParams[key],
                                      'type': 'trackTraj',})

        StatusPrint('added task for: ',collectingKeys)
        await self.taskProcessor()
        return collectingKeys
    
    def checkSimFinished(self):
        return len(self.simTaskQueue) + len(self.runningSims) == 0
    
    def handleSimConnections(self):
        # 'connect' is sent automatically by Socket.IO as soon as TCP/WebSocket handshake succeeds.
        # Sends self.params to the client.
        @self.sio.on('connect')
        async def connect(sid,environ):
            StatusPrint('Simulator connected.',sid)
            await self.sio.emit('set_params', self.params, room=sid)

        # Simulator client indicates readiness.
        @self.sio.on('sim_ready')
        async def startSim(sid):
            StatusPrint('Simulator ready.',sid)
            self.idleSims.add(sid)
            await self.taskProcessor()

        # Simulator client disconnected. 
        @self.sio.on('disconnect')
        async def disconnect(sid):
            StatusPrint('sim disconnected ',sid)
            # If the disconnected client was running midway, we add it back to the task queue.
            if sid in self.runningSims:
                self.simTaskQueue.append(self.runningSims.pop(sid))
            # If the disconnected client was idle, we remove it.
            if sid in self.idleSims:
                self.idleSims.remove(sid)
            await self.taskProcessor()

    def handleSimResults(self):
        """
        If results were emitted by a simulator client, and the client was assigned a task (in self.runningSims), we 
        take the data collected by the simulator client, convert it to a tensor, and process it with 
        self.processTrajData.
        """
        # Results were emitted by a simulator client
        @self.sio.on('results')
        async def addSimData(sid, results_fp):
            """

            Params:
                results_fp [str]: string pointing to directory where TensorDict is stored.
            """
            StatusPrint('got results')

            # socket_id not found in runningSims.
            if not sid in self.runningSims:
                StatusPrint('ignoring results')
                return
            
            # Check if fp is valid
            newData = td.load_memmap(results_fp)

            # Converts data collected by robot into tensors.
            # newData = {}
            # for dataKey in DATA_COLLECTED:
            #     newData[dataKey] = toTensor(results[dataKey])
            
            # sid emitting results is done - remove from runningSims and add to idleSims.
            robotKey = self.runningSims[sid]['key']
            self.runningSims.pop(sid)
            self.idleSims.add(sid)

            # Process trajectory data obtained through socket.
            self.processTrajData(robotKey, **newData)
            breakpoint()

            await self.taskProcessor()

    def generateRobot(self):
        """
        Creates a newKey (based on length of self.robotParams dictionary). Generates new parameters for a robot using
        genParam. Updates self.robotParams with new robot, and adds key to activeCollectingKeys.

        Processes TrajData for the new key.
        """
        StatusPrint('generating robot')
        newKey = len(self.robotParams)
        newParams = genParam(self.params['robotRange'], gen_mean=self.params['train']['useNominal'])
        self.robotParams[newKey] = newParams
        self.activeCollectingKeys.add(newKey)
        self.processTrajData(newKey)
        
        return newKey
    
    def processTrajData(self, key, **newDataKwargs):
        """
        Retrieves or adds trajectory data into the data folder.
        If the new data collected (from newDataKwargs) matches the data we're hoping to collect (defined in const 
        DATA_COLLECTED), we save the data into {trajDataDir}robotKey{.pt}.

        Params:
            key: [int] robot key, queried from self.runningSims[socketID]
            **newDataKwargs: incoming data
        """
        
        # check path of data
        trajDataDir = os.path.join(self.params['dataDir'], 'trajData/')
        if not os.path.exists(trajDataDir):
            os.mkdir(trajDataDir)
        fn = os.path.join(trajDataDir, f"{key}.pt")
        
        breakpoint()
        # Load old data. If old data doesn't exist, create a tuple of empty tensors of dims according to data format.
        try:
            data = torch.load(fn)
        except:
            data = tuple([torch.tensor([])] * len(DATA_COLLECTED))
            torch.save(data,fn)
        
        # If the newDataKwargs fed into processTrajData matches the dataformat of what we're expecting to collect,
        # Add data into file.
        if len(newDataKwargs) == len(DATA_COLLECTED):
            # data is a tuple of tensors
            data = list(data)
            # TODO: guard this with proper parameters to prevent problems downstream
            for i in range(len(DATA_COLLECTED)):
                data[i] = torch.cat((data[i], newDataKwargs[DATA_COLLECTED[i]]), dim=0)
            data = tuple(data)
            torch.save(data, fn)

        # Gets index corresponding to actions taken.
        actionIndex = DATA_COLLECTED.index('actions')
        
        # Get numSteps according to number of actions within data.
        # Note the second condition - we only count non torch.inf actions.
        numSteps = 0 if data[actionIndex].shape[0] == 0 else torch.sum(data[actionIndex][:, 0] != torch.inf)

        # If the number of actions exceed the maxStepsPerRobot, the robot is done working, and 
        # we remove the robot from self.activeCollectingKeys.
        if numSteps >= self.params['train']['maxStepsPerRobot']:
            self.activeCollectingKeys.remove(key)
            StatusPrint('finished collecting for robot: ',key)

        return data

if __name__=="__main__":
    # load arguments
    parser = argparse.ArgumentParser(description='script for handled decentralized simulation and training')
    parser.add_argument('trainDir',type=str,help='directory where model is stored (should also contain config dir)')
    parser.add_argument('dataDir',nargs="?",type=str,help='directory for storing training data',default=None)
    args = parser.parse_args()
    
    Server(args)
