import pdb
import math
import torch
import numpy as np
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

from .architecture.dynamicsModel import SysIDTransformer,AdaptiveDynamicsModel, TerrainNet, ParamNet
from .architecture.multiRobotDataset import SampleLoader
from .architecture.tools import gausLogLikelihood
from .architecture.StatusPrint import StatusPrint

from utils.planarRobotState import get_relative_state,getLocalMap
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import asyncio
import time
from collections import defaultdict
import torch.nn.functional as F
import random

def SwitchRequiresGrad(model,state):
    for param in model.parameters():
        param.requires_grad = state

class WarmupInverseSqrtDecay(_LRScheduler):
    """
    Learning rate scheduler from the "Attention is All You Need" paper. Starts out with a constant learning rate for 
    warmup_steps, then exponentially decays the learning rate.
    """
    def __init__(self, optimizer, warmup_steps, use_decay=True, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.use_decay = use_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        
        # Linear warmup
        if current_epoch <= self.warmup_steps:
            lr = [base_lr * current_epoch / self.warmup_steps for base_lr in self.base_lrs]
        
        # Inverse square root decay
        elif self.use_decay:
            lr = [base_lr * (self.warmup_steps ** 0.5) * (current_epoch ** -0.5) for base_lr in self.base_lrs]
        
        else:
            lr = [base_lr for base_lr in self.base_lrs]
        
        return lr

class ModelTrainer(object):
    TENSORS = 0
    ROBOT_KEY = 1

    def __init__(self,server,params):
        self.server = server
        self.params = params
            
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # define network structure
        self.sysIDTransformer       = SysIDTransformer(self.params['network'],self.params['controls']).to(self.device)
        self.adaptiveDynamicsModel  = AdaptiveDynamicsModel(self.params['network'],self.params['controls']).to(self.device)
        self.param_net              = ParamNet(self.params['network'],self.params['robotRange']).to(self.device)

        # make train
        self.sysIDTransformer.train()
        self.adaptiveDynamicsModel.train()
        self.param_net.train()
        
        # file name for storing model
        self.adm_file = os.path.join(self.params['trainDir'],'adm.pt')
        self.sit_file = os.path.join(self.params['trainDir'],'sit.pt')
        self.pn_file = os.path.join(self.params['trainDir'],'pn.pt')

        if self.params['train']['useOldModel']:
            #self.sysIDTransformer.load_state_dict(torch.load(self.sit_file))
            self.adaptiveDynamicsModel.load_state_dict(torch.load(self.adm_file))
            self.param_net.load_state_dict(torch.load(self.pn_file))

        # Training Phases. 
        # Phase 0: param_net and adaptiveDynamicsModel. 
        # Phase 1: SysID Transformer
        # Phase 2: both the dynamics model and the SysID Transformer.
        self.phases = []
        for i in range(len(self.params['train']['phase_lens'])+1):
            phase = {}
            if i == 0:
                phase['params'] = list(self.param_net.parameters()) + list(self.adaptiveDynamicsModel.parameters()) #+ list(self.sysIDTransformer.parameters())
            elif i == 1:
                phase['params'] = self.sysIDTransformer.parameters()
            else:
                phase['params'] = list(self.adaptiveDynamicsModel.parameters()) + list(self.sysIDTransformer.parameters())
            
            phase['optimizer'] = Adam(phase['params'], lr=self.params['train']['lrs'][i])
            phase['scheduler'] = WarmupInverseSqrtDecay(phase['optimizer'],
                                                        self.params['train']['warmup_steps'][i],
                                                        use_decay = self.params['train']['use_decay'][i])
            phase['gradClip'] = self.params['train']['gradClips'][i]
            self.phases.append(phase)

        # tensorboard logger
        self.writer = SummaryWriter(comment=self.params['trainDir'])

    async def run_training(self):
        trainIt = 0 # global iterations
        phase   = 0 # flag for tracking which phase we are in

        # self.params['train']['phase_lens'] tells when the phase should change from one stage to the next.
        # self.params['train']['phase_lens'] = number of iterations in each phase.
        phaseChangeIts = np.cumsum(self.params['train']['phase_lens'] + [float('inf')])

        while True:
            # handle data generation
            if trainIt % self.params['train']['batchTrainIts'] == 0:
                torch.cuda.empty_cache()
                
                # wait for simulations to finish before starting next training batch
                while not self.server.checkSimFinished():
                    StatusPrint('[ModelTrainer] Pending simTasks {}, runningSims {}'.format(
                        len(self.server.simTaskQueue),
                        len(self.server.runningSims)), isTemp=True)
                    await asyncio.sleep(1)

                # reload data from simulations
                StatusPrint('[ModelTrainer] reloading simulation data')
                self.reloadData() # samples will be reloaded once training batch finishes
                newDataKeys = await self.server.addBatchToSimQueue()

            # Select training phase based on current iterations
            while trainIt >= phaseChangeIts[phase]:
                phase += 1

            # Fetch one trajectory sample randomly from the SampleLoader
            dataset_sample  = self.samples.getSample()
            robotKey        = dataset_sample[self.ROBOT_KEY][0].item()
            robotParam      = self.samples.data.robotMeta[robotKey] # TODO: remove.

            # Data processing function that transforms data into a dictionary containing
            # [predLocalMaps, predPriorTrans, predRelTrajRefs, predTargetTransitions,
            # histLocalMaps, histPriorTransitions, histActions, histTransitions]
            x_train = self.extractTrainTraj(dataset_sample[self.TENSORS])
            for key in x_train:
                x_train[key] = x_train[key].unsqueeze(0)

            # Estimate context vector given training phase
            param_mean, param_std = self.param_net(robotParam) # TODO: remove.
            sysID_mean, sysID_std = self.sysIDTransformer(
                x_train['histLocalMaps'],
                x_train['histPriorTransitions'],
                x_train['histActions'],
                x_train['histTransitions'])

            # TODO: learn what context means.
            if phase == 0:
                context_mean = param_mean
                context_std = param_std
                #kl_div = 0.5*torch.sum((param_std/sysID_std)**2 + ((param_mean-sysID_mean)/sysID_std)**2 - 1 + 2*(torch.log(sysID_std)-torch.log(param_std)),dim=-1)
                #kl_div = kl_div.mean()
            else:
                context_mean = sysID_mean
                context_std = sysID_std

            num_mixtures    = self.params['train']['num_mixtures']
            context_mean    = context_mean.expand(num_mixtures, -1, -1) # [1, ]
            context_std     = context_std.expand(num_mixtures, -1, -1)
            context_samples = torch.randn_like(context_mean) * context_std + context_mean
            
            # Infer using adaptiveDynamicsModel
            predMean, predLVar = self.adaptiveDynamicsModel(
                x_train['predLocalMaps'],
                x_train['predPriorTrans'].expand(num_mixtures,-1,-1),
                x_train['predRelTrajRefs'].expand(num_mixtures,-1,-1,-1),
                context=context_samples
            )

            try:
                mode_log_likelihoods = gausLogLikelihood(
                    predMean,
                    predLVar,
                    x_train['predTargetTransitions'].expand(num_mixtures, -1, -1, -1)
                )
            except:
                pdb.set_trace()
            
            maxLog, _ = torch.max(mode_log_likelihoods, dim=0, keepdim=True)
            mixture_log_likelihood = (mode_log_likelihoods - maxLog).exp().mean(dim=0).log() + maxLog

            loss = -mixture_log_likelihood.mean() #+ kl_div.mean()*self.params['train']['kl_div_scale']
            #self.samples.sampler.updateLoss(loss,robotKey)

            # Backpropagation
            self.phases[phase]['optimizer'].zero_grad()
            loss.backward()
            for param in self.phases[phase]['params']:
                if not param.grad is None and param.grad.isnan().any():
                    print('nan grad')
                    pdb.set_trace()
            
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(self.phases[phase]['params'], self.phases[phase]['gradClip'])
            
            # Step after backpropagation
            self.phases[phase]['optimizer'].step()
            self.phases[phase]['scheduler'].step()
            
            StatusPrint('trainit: ',trainIt,' loss: ',loss.item(),isTemp=True)
            self.writer.add_scalar('Loss/train', loss.item(), trainIt)
            self.writer.add_scalar('lr', self.phases[phase]['scheduler'].get_lr()[0], trainIt)
            
            if trainIt % 1000 == 0:
                torch.save(self.sysIDTransformer.state_dict(), self.sit_file)
                torch.save(self.adaptiveDynamicsModel.state_dict(), self.adm_file)
                torch.save(self.param_net.state_dict(), self.pn_file)

            trainIt += 1
            await asyncio.sleep(0.01)

    def extractTrainTraj(self, sample):
        """ 
        Data wrangling before training. Constructs a dictionary with a uniformly sampled valid trajectory "removed" for
        training purposes. For history context, includes all other trajectories as well.

        # Step 1: Adds noise to states and gets stateTransitions / noisyTransitions. Pads with row to ensure dims.
        # Step 2: Gets relative reference trajectory (from robot's noisy state to reference trajectory)
        # Step 3: Gets all local maps of different trajectories (once for each trajectory).
        # Step 4: Randomly sample one random trajectory to blank out as "pred"
        # Step 5: Generates history segment from all other trajectories.

        Params:
            sample: List of flattened tensors of the following form: [states, actions, trajRefs, worldMap, worldMapBounds]

        Returns:
            [dict] 
            predLocalMaps   : sliced local maps corresponding to the trajectory randomly chosen from all valid trajs.
            predPriorTrans  : noisy transitions within the chosen trajectory
            predRelTrajRefs : relative transformation from robot's noisy state to the reference state.
            predTargetTransitions   : clean transitions within the chosen trajectory
            histLocalMaps           : all local maps before/after choice trajectory. Includes other trajectories.
            histPriorTransitions    : all prior noisy transitions before/after choice trajectory.
            histActions             : all actions before/after choice trajectory.
            histTransitions         : all noisy transitions before/after choice trajectory.
        """
        # states, actions, trajRefs, worldMap, worldMapBounds = sample
        
        # states: [x, y, v, theta]
        # actions: [a, d_f, d_r]
        # xdot: [xdot, ydot, vdot, thetadot] <--- this is the "ground truth
        # TODO: you were here.
        states, actions, xdot = sample
        breakpoint()
        
        with torch.no_grad():
            # add noise to states
            noise = torch.randn(states.shape) * torch.tensor(self.params['train']['stateNoise'])
            noisyStates = states + noise.to(states.device)

            # Get transitions
            stateTransitions = get_relative_state(states[:-1, :], states[1:, :])
            noisyTransitions = get_relative_state(noisyStates[:-1, :], noisyStates[1:, :])
            invalidTransitions = torch.arange(actions.shape[0] - 1)[(actions[:-1, 0] == torch.inf).to('cpu')]
            
            # Make final state transitions relative state = 0
            stateTransitions[invalidTransitions, :] = get_relative_state(states[invalidTransitions + 1, :],
                                                                         states[invalidTransitions + 1, :])
            noisyTransitions[invalidTransitions, :] = get_relative_state(noisyStates[invalidTransitions+1,:],
                                                                         noisyStates[invalidTransitions+1,:])
            stateTransitions = torch.cat((stateTransitions, get_relative_state(states[:1, :], states[:1, :])),dim=0)
            noisyTransitions = torch.cat((noisyTransitions, get_relative_state(noisyStates[:1, :], noisyStates[:1, :])),dim=0)
            priorNoisyTransitions = noisyTransitions.roll(1, dims=0)

            # randomly choose a trajectory
            trainPredSeqLen = self.params['train']['trainPredSeqLen']
            
            # get numTrainTrajs, the cumulative sum of number of valid trajectories e.g. [1, 2, 2]. If certain 
            # trajectories are longer than trainPredSeqLen, code accounts for that.
            trajEnd = (actions[:, 0] == torch.inf).to('cpu')
            trajEnd[-1] = True
            trajEnd = torch.arange(actions.shape[0])[trajEnd]
            trajEnd = torch.cat((torch.tensor([-1]), trajEnd), dim=0) # Prepend with -1 so first traj has correct #
            numTrainTrajs = torch.clamp(trajEnd[1:] - trajEnd[:-1] - trainPredSeqLen, min=0).cumsum(dim=0)

            # Step 4: Choose a trajectory from the available training trajectories
            choice = torch.randint(1, numTrainTrajs[-1], (1,))
            choiceTraj = torch.arange(len(numTrainTrajs))[numTrainTrajs > choice].min() # Map choice to trajectory # 
            choiceIndex = choice - torch.cat((torch.tensor([0]), numTrainTrajs))[choiceTraj] # Index offset of traj
            choiceStart = trajEnd[choiceTraj] + 1 + choiceIndex
            choiceEnd   = choiceStart + trainPredSeqLen

            # Generate prediction segment using random choice of trajectory
            # predRelTrajRefs = relativeTrajRefs[choiceStart:choiceEnd, :]
            # predLocalMaps   = localMaps[choiceStart:choiceEnd, :]
            # predStartState  = states[choiceStart, :]
            predPriorTrans  = priorNoisyTransitions[choiceStart:choiceEnd, :]
            predTargetTransitions = stateTransitions[choiceStart:choiceEnd, :]

            # Step 5: Generate history segment from all other trajectories.
            histActions = torch.cat((
                actions[:choiceStart],
                torch.ones_like(actions[:1]) * torch.inf,
                actions[choiceEnd:]),dim=0)
            histPriorTransitions = torch.cat((
                priorNoisyTransitions[:choiceStart],
                torch.zeros_like(priorNoisyTransitions[:1]),
                priorNoisyTransitions[choiceEnd:]),dim=0)
            histTransitions = torch.cat((
                noisyTransitions[:choiceStart],
                torch.zeros_like(noisyTransitions[:1]),
                noisyTransitions[choiceEnd:]),dim=0)
            # histLocalMaps = torch.cat((
            #     localMaps[:choiceStart],
            #     torch.zeros_like(localMaps[:1]),
            #     localMaps[choiceEnd:]),dim=0)

            # generate history segment
            #histIndices = torch.arange(max(trajEnd[choiceTraj]+1,1))
            #histActions = actions[histIndices,:]
            #histTransitions = noisyTransitions[histIndices,:]
            #histLocalMaps = localMaps[histIndices,:]

        return {'predPriorTrans'        : predPriorTrans,
                'predTargetTransitions' : predTargetTransitions,
                'histPriorTransitions'  : histPriorTransitions,
                'histActions'           : histActions,
                'histTransitions'       : histTransitions}
                # 'predLocalMaps'         : predLocalMaps,
                # 'predRelTrajRefs'       : predRelTrajRefs,
                # 'histLocalMaps'         : histLocalMaps,
                
                
                

    # def extractTrainTraj(self, sample):
    #     """ 
    #     Data wrangling before training. Constructs a dictionary with a uniformly sampled valid trajectory "removed" for
    #     training purposes. For history context, includes all other trajectories as well.

    #     # Step 1: Adds noise to states and gets stateTransitions / noisyTransitions. Pads with row to ensure dims.
    #     # Step 2: Gets relative reference trajectory (from robot's noisy state to reference trajectory)
    #     # Step 3: Gets all local maps of different trajectories (once for each trajectory).
    #     # Step 4: Randomly sample one random trajectory to blank out as "pred"
    #     # Step 5: Generates history segment from all other trajectories.

    #     Params:
    #         sample: List of flattened tensors of the following form: [states, actions, trajRefs, worldMap, worldMapBounds]

    #     Returns:
    #         [dict] 
    #         predLocalMaps   : sliced local maps corresponding to the trajectory randomly chosen from all valid trajs.
    #         predPriorTrans  : noisy transitions within the chosen trajectory
    #         predRelTrajRefs : relative transformation from robot's noisy state to the reference state.
    #         predTargetTransitions   : clean transitions within the chosen trajectory
    #         histLocalMaps           : all local maps before/after choice trajectory. Includes other trajectories.
    #         histPriorTransitions    : all prior noisy transitions before/after choice trajectory.
    #         histActions             : all actions before/after choice trajectory.
    #         histTransitions         : all noisy transitions before/after choice trajectory.
    #     """
    #     states, actions, trajRefs, worldMap, worldMapBounds = sample

    #     with torch.no_grad():
    #         # add noise to states
    #         noise = torch.randn(states.shape)*torch.tensor(self.params['train']['stateNoise'])
    #         noisyStates = states+noise.to(states.device)

    #         stateTransitions = get_relative_state(states[:-1, :],states[1:, :])
    #         noisyTransitions = get_relative_state(noisyStates[:-1, :],noisyStates[1:, :])
    #         invalidTransitions = torch.arange(actions.shape[0] - 1)[(actions[:-1, 0] == torch.inf).to('cpu')]
    #         stateTransitions[invalidTransitions, :] = get_relative_state( states[invalidTransitions+1,:],
    #                                                                     states[invalidTransitions+1,:])
    #         noisyTransitions[invalidTransitions, :] = get_relative_state( noisyStates[invalidTransitions+1,:],
    #                                                                     noisyStates[invalidTransitions+1,:])
    #         stateTransitions = torch.cat((stateTransitions, get_relative_state(states[:1, :], states[:1, :])),dim=0)
    #         noisyTransitions = torch.cat((noisyTransitions, get_relative_state(noisyStates[:1, :], noisyStates[:1, :])),dim=0)
    #         priorNoisyTransitions = noisyTransitions.roll(1, dims=0)

    #         # calculate relative reference trajectory
    #         relativeTrajRefs = get_relative_state(noisyStates.unsqueeze(-2), trajRefs)

    #         # calculate local maps
    #         mapIndex = (actions[:, 0] == torch.inf).cumsum(dim=0).tolist()
    #         mapIndex = [0] + mapIndex[:-1]
    #         localMaps = getLocalMap(noisyStates,
    #                                 worldMap[mapIndex, :].unsqueeze(1),
    #                                 worldMapBounds[mapIndex,:],
    #                                 self.params['network']['localMap'])

    #         # randomly choose a trajectory
    #         trainPredSeqLen = self.params['train']['trainPredSeqLen']
            
    #         # get numTrainTrajs, the cumulative sum of number of valid trajectories e.g. [1, 2, 2]. If certain 
    #         # trajectories are longer than trainPredSeqLen, code accounts for that.
    #         trajEnd = (actions[:, 0] == torch.inf).to('cpu')
    #         trajEnd[-1] = True
    #         trajEnd = torch.arange(actions.shape[0])[trajEnd]
    #         trajEnd = torch.cat((torch.tensor([-1]), trajEnd), dim=0) # Prepend with -1 so first traj has correct #
    #         numTrainTrajs = torch.clamp(trajEnd[1:] - trajEnd[:-1] - trainPredSeqLen, min=0).cumsum(dim=0)

    #         # Step 4: Choose a trajectory from the available training trajectories
    #         choice = torch.randint(1, numTrainTrajs[-1], (1,))
    #         choiceTraj = torch.arange(len(numTrainTrajs))[numTrainTrajs > choice].min() # Map choice to trajectory # 
    #         choiceIndex = choice - torch.cat((torch.tensor([0]), numTrainTrajs))[choiceTraj] # Index offset of traj
    #         choiceStart = trajEnd[choiceTraj] + 1 + choiceIndex
    #         choiceEnd   = choiceStart + trainPredSeqLen

    #         # Generate prediction segment using random choice of trajectory
    #         predRelTrajRefs = relativeTrajRefs[choiceStart:choiceEnd, :]
    #         predLocalMaps   = localMaps[choiceStart:choiceEnd, :]
    #         predStartState  = states[choiceStart, :]
    #         predPriorTrans  = priorNoisyTransitions[choiceStart:choiceEnd, :]
    #         predTargetTransitions = stateTransitions[choiceStart:choiceEnd, :]

    #         # Step 5: Generate history segment from all other trajectories.
    #         histActions = torch.cat((
    #             actions[:choiceStart],
    #             torch.ones_like(actions[:1]) * torch.inf,
    #             actions[choiceEnd:]),dim=0)
    #         histPriorTransitions = torch.cat((
    #             priorNoisyTransitions[:choiceStart],
    #             torch.zeros_like(priorNoisyTransitions[:1]),
    #             priorNoisyTransitions[choiceEnd:]),dim=0)
    #         histTransitions = torch.cat((
    #             noisyTransitions[:choiceStart],
    #             torch.zeros_like(noisyTransitions[:1]),
    #             noisyTransitions[choiceEnd:]),dim=0)
    #         histLocalMaps = torch.cat((
    #             localMaps[:choiceStart],
    #             torch.zeros_like(localMaps[:1]),
    #             localMaps[choiceEnd:]),dim=0)

    #         # generate history segment
    #         #histIndices = torch.arange(max(trajEnd[choiceTraj]+1,1))
    #         #histActions = actions[histIndices,:]
    #         #histTransitions = noisyTransitions[histIndices,:]
    #         #histLocalMaps = localMaps[histIndices,:]

    #     return {'predLocalMaps'         : predLocalMaps,
    #             'predPriorTrans'        : predPriorTrans,
    #             'predRelTrajRefs'       : predRelTrajRefs,
    #             'predTargetTransitions' : predTargetTransitions,
    #             'histLocalMaps'         : histLocalMaps,
    #             'histPriorTransitions'  : histPriorTransitions,
    #             'histActions'           : histActions,
    #             'histTransitions'       : histTransitions}

    def reloadData(self):
        """
        Updates self.samples with sampleLoader
        """
        if hasattr(self,'samples'):
            self.samples.reloadData()
        else:
            self.samples = SampleLoader(self.params['dataDir'], device=self.device)
