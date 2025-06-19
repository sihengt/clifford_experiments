import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence,pad_packed_sequence
from torch import nn,Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder,TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer
from torch.nn.modules.normalization import LayerNorm
from torchvision.models import ResNet

import math
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

class TerrainNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TerrainNetBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        #out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TerrainNet(nn.Module):
    def __init__(self, terrainNetParams):
        super(TerrainNet, self).__init__()

        # Extract Params
        in_channels = terrainNetParams['in_channels']
        compressedDim = terrainNetParams['compressedDim']
        layer_channels = terrainNetParams['layer_channels']
        layer_strides = terrainNetParams['layer_strides']
        num_blocks = terrainNetParams['num_blocks']

        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_channels[0])
    
        self.in_channels = layer_channels[0]

        # Create ResNet layers
        self.layers = nn.Sequential(*[self._make_layer(layer_channels[i],num_blocks[i],layer_strides[i]) for i in range(len(layer_channels))])

        # Linear layer for output
        self.linear = nn.Linear(layer_channels[-1], compressedDim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) 
        layers = []
        for stride in strides:
            layers.append(TerrainNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, context=None):
        contextLayers = torch.ones_like(x[...,:1,:,:])
        origBatchShape = x.shape[:-3]
        x = x.view(-1,*x.shape[-3:])
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.mean(dim=[-1,-2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.view(*origBatchShape,out.shape[-1])

class SysIDTransformer(nn.Module):
    def __init__(self,networkParams,controlParams):
        super(SysIDTransformer, self).__init__()
        
        # initialize terrain resnet
        self.terrainNet = TerrainNet(networkParams['sysID']['terrain'])

        d_model = networkParams['sysID']['d_model']
        input_size = controlParams['actionDim'] + \
                networkParams['stateTransitionDim']*2 + \
                networkParams['sysID']['terrain']['compressedDim'] 

        if 'lstm_preprocess' in networkParams['sysID']:
            self.lstm = nn.LSTM(input_size,
                                d_model,
                                networkParams['sysID']['lstm_preprocess']['num_layers'],
                                dropout=networkParams['sysID']['lstm_preprocess']['dropout'],
                                batch_first=True)
        else:
            self.inputFC = nn.Linear(input_size,d_model)
            # define positional encoding matrix
            max_seq_len = networkParams['sysID']['max_seq_len']
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            self.pe = torch.zeros(max_seq_len, d_model)
            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = self.pe.unsqueeze(0)

        """
        encoder_layer = TransformerEncoderLayer(d_model,
                                                networkParams['sysID']['nhead'],
                                                networkParams['sysID']['dim_feedforward'],
                                                networkParams['sysID']['dropout'],
                                                batch_first=True)

        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer,
                                        networkParams['sysID']['num_layers'],
                                        encoder_norm)
        """
        self.encoder = nn.MultiheadAttention(d_model,
                                            networkParams['sysID']['nhead'],
                                            networkParams['sysID']['dropout'],
                                            batch_first=True)

        # pooling operation for compressing transformer output in to single vector
        self.pooling_op = networkParams['sysID']['pooling_op']
        if self.pooling_op == "weighted_mean":
            self.poolWeightFC = nn.Linear(d_model,d_model)
            #self.poolWeightFC = nn.Linear(d_model,networkParams['contextDim'])

        # add final linear layer to further compress encoding
        self.finalFC_mean = nn.Linear(d_model,networkParams['contextDim'])
        self.finalFC_std = nn.Linear(d_model,networkParams['contextDim'])
        self.stochastic = 'stochastic' in networkParams['sysID'] and networkParams['sysID']['stochastic']

    def to(self, *args, **kwargs):
        super().to(*args,**kwargs)
        if hasattr(self,'pe'):
            self.pe = self.pe.to(*args,**kwargs)
        return self

    def forward(self,localMap,priorTransitions,actions,stateTransitions):
        compressedMap = self.terrainNet(localMap)
        if hasattr(self,'lstm'):
            connected = torch.cat((compressedMap,priorTransitions,actions,stateTransitions),dim=-1)
            traj_end_index = torch.arange(actions.shape[1],device=actions.device)[actions[0,:,0]==torch.inf]
            sequences = []
            for i in range(traj_end_index.shape[0]):
                start = traj_end_index[i-1] + 1 if i>0 else 0
                end = traj_end_index[i]
                if start == end:
                    continue
                sequences.append(connected[0,start:end])
            seq_lens = [seq.shape[0] for seq in sequences]
            sequences = pad_sequence(sequences,batch_first=True)
            packed_input = pack_padded_sequence(sequences,seq_lens,batch_first=True,enforce_sorted=False)
            connected,_ = self.lstm(packed_input)
            connected,seq_lens = pad_packed_sequence(connected, batch_first=True)
            connected = torch.cat([connected[i,:seq_lens[i]] for i in range(connected.shape[0])],dim=0).unsqueeze(0)
        else:
            traj_end = actions[...,0] == torch.inf
            actions[traj_end] = 0
            connected = torch.cat((compressedMap,priorTransitions,actions,stateTransitions),dim=-1)
            connected = self.inputFC(connected)
            connected = connected*(~traj_end).unsqueeze(-1)
            #connected = connected + self.pe[:,:connected.shape[-2],:]

        if self.pooling_op == "last":
            connected = torch.cat((connected,torch.ones_like(connected[...,-1:,:])),dim=-2)

        #connected = self.encoder(connected, mask=None, src_key_padding_mask=None)
        connected, attn_weights = self.encoder(connected,connected,connected)

        # pool operation
        if self.pooling_op == "global_mean":
            connected = torch.mean(connected,dim=-2,keepdim=True)
        elif self.pooling_op == "max":
            connected = torch.max(connected,dim=-2,keepdim=True)[0]
        elif self.pooling_op == "weighted_mean":
            poolWeights = self.poolWeightFC(connected)
            poolWeights = F.softmax(poolWeights,dim=-2)
            connected = torch.sum(connected*poolWeights,dim=-2,keepdim=True)
        elif self.pooling_op == "last":
            connected = connected[...,-1:,:]

        context_mean = self.finalFC_mean(connected)
        context_std = F.softplus(self.finalFC_mean(connected))
        if not self.stochastic:
            context_std = torch.zeros_like(context_std)
        return context_mean,context_std

class AdaptiveDynamicsModelNoCoords(nn.Module):
    def __init__(self, networkParams, controlParams, mpcParams):
        super(AdaptiveDynamicsModelNoCoords, self).__init__()
        # initialize LSTM for dynamics prediction
        self.networkParams = networkParams

        # TODO: stateTransitionDim should now be 4 instead of 3.
        
        # Reference: T (horizon for MPC) * number of state dimensions for reference.
        # refTrajDim = int(mpcParams['T'] * networkParams['refStateDim'])
        
        # TODO: include all the extras you want to include in input_size. IIRC, action and Xdot transitions included.
        # input_size = refTrajDim + networkParams['stateTransitionDim']
        # input_size = networkParams['xdotDim'] + networkParams['xdotDim'] + networkParams['actionDim']
        input_size = networkParams['xdotDim'] + networkParams['actionDim']
        # inputVelocityTransition   = [8, 4]
        # inputActions.shape        = [8, 3]
        # targetVelocityTransition  = [1, 4]
        self.lstm = nn.LSTM(input_size,
                            networkParams['dynamicsModel']['hidden_size'],
                            networkParams['dynamicsModel']['num_layers'],
                            dropout=networkParams['dynamicsModel']['dropout'],
                            batch_first=True)

        self.meanFC = nn.Linear(networkParams['dynamicsModel']['hidden_size'],
                                networkParams['stateTransitionDim'])
        self.LVarFC = nn.Linear(networkParams['dynamicsModel']['hidden_size'],
                                networkParams['stateTransitionDim']**2)
        
        # Parameter determining lower bound of variance.
        self.varLowBound = networkParams['dynamicsModel']['varLowBound']
        
        # Parameter that clips reference points from being too far from the current point
        self.maxTrajRefDist = networkParams['dynamicsModel']['maxTrajRefDist']

        # Parameter determining if variance depends on input.
        self.staticVar = 'staticVar' in networkParams['dynamicsModel'] and networkParams['dynamicsModel']['staticVar']
    
    def to(self,device):
        self.device = device
        return super().to(device)

    def forward(self, x_window, action_window, context=None, hidden = None, returnHidden = False):
        """
        Params:
            xdot_window:
            action_window: 
            returnHidden: [boolean] flag for returning hidden layer
        """
        # TODO: hardcoded batch and seq len for now
        # connected = input.reshape(1, 8, self.networkParams['xdotDim'] + self.networkParams['actionDim'])
        # shape = (window length, actions + states)
        connected = torch.cat((
            x_window,
            action_window), dim=-1)
        
        # Forward pass
        connected, hidden = self.lstm(connected, hidden)
        connected = connected[:, -1, :]
        
        # Adds the context to the fully connected layer for the mean as well
        mean = self.meanFC(connected) # (batch, trainPredSeqLen, n_states)

        # Dealing with variance:
        # self.staticVar: variance only depend on the weights learned by the layer
        if self.staticVar:
            LVar = self.LVarFC(torch.zeros_like(connected))
        else:
            LVar = self.LVarFC(connected)

        # Mean shape: (B, T, D), variance shape: (B, T, D, D), a square matrix
        LVar = LVar.view(*mean.shape, mean.shape[-1])

        # Makes the diagonal strictly positive
        diag = (F.softplus(LVar) + self.varLowBound).diagonal(dim1=-2, dim2=-1).diag_embed()

        # Assembles covariance with strict lower triangle
        # Also known as Cholesky factor, L.
        # To reconstruct variance L^T L
        offDiag = LVar.tril(diagonal=-1)
        LVar = diag + offDiag
        
        if returnHidden:
            return mean, LVar, hidden
        
        # mean = [batch_size, n_states]
        # LVar = [batch_size, n_states, n_states]
        
        # TODO: hardcoded batch
        # return torch.cat((mean, LVar.reshape(1, -1)), axis=1)
        return mean, LVar

class AdaptiveDynamicsModel(nn.Module):
    def __init__(self, networkParams, controlParams, mpcParams):
        super(AdaptiveDynamicsModel, self).__init__()
        # initialize LSTM for dynamics prediction
        self.networkParams = networkParams

        # TODO: stateTransitionDim should now be 4 instead of 3.
        
        # Reference: T (horizon for MPC) * number of state dimensions for reference.
        # refTrajDim = int(mpcParams['T'] * networkParams['refStateDim'])
        
        # TODO: include all the extras you want to include in input_size. IIRC, action and Xdot transitions included.
        # input_size = refTrajDim + networkParams['stateTransitionDim']
        # input_size = networkParams['xdotDim'] + networkParams['xdotDim'] + networkParams['actionDim']
        input_size = networkParams['xdotDim'] + networkParams['actionDim']
        # inputVelocityTransition   = [8, 4]
        # inputActions.shape        = [8, 3]
        # targetVelocityTransition  = [1, 4]
        self.lstm = nn.LSTM(input_size,
                            networkParams['dynamicsModel']['hidden_size'],
                            networkParams['dynamicsModel']['num_layers'],
                            dropout=networkParams['dynamicsModel']['dropout'],
                            batch_first=True)

        self.meanFC = nn.Linear(networkParams['dynamicsModel']['hidden_size'],
                                networkParams['stateTransitionDim'])
        self.LVarFC = nn.Linear(networkParams['dynamicsModel']['hidden_size'],
                                networkParams['stateTransitionDim']**2)
        
        # Parameter determining lower bound of variance.
        self.varLowBound = networkParams['dynamicsModel']['varLowBound']
        
        # Parameter that clips reference points from being too far from the current point
        self.maxTrajRefDist = networkParams['dynamicsModel']['maxTrajRefDist']

        # Parameter determining if variance depends on input.
        self.staticVar = 'staticVar' in networkParams['dynamicsModel'] and networkParams['dynamicsModel']['staticVar']
    
    def to(self,device):
        self.device = device
        return super().to(device)

    def forward(self, x_window, action_window, context=None, hidden = None, returnHidden = False):
        """
        Params:
            xdot_window:
            action_window: 
            returnHidden: [boolean] flag for returning hidden layer
        """
        # TODO: hardcoded batch and seq len for now
        # connected = input.reshape(1, 8, self.networkParams['xdotDim'] + self.networkParams['actionDim'])
        # shape = (window length, actions + states)
        connected = torch.cat((
            x_window,
            action_window), dim=-1)
        
        # Forward pass
        connected, hidden = self.lstm(connected, hidden)
        connected = connected[:, -1, :]
        
        # Adds the context to the fully connected layer for the mean as well
        mean = self.meanFC(connected) # (batch, trainPredSeqLen, n_states)

        # Dealing with variance:
        # self.staticVar: variance only depend on the weights learned by the layer
        if self.staticVar:
            LVar = self.LVarFC(torch.zeros_like(connected))
        else:
            LVar = self.LVarFC(connected)

        # Mean shape: (B, T, D), variance shape: (B, T, D, D), a square matrix
        LVar = LVar.view(*mean.shape, mean.shape[-1])

        # Makes the diagonal strictly positive
        diag = (F.softplus(LVar) + self.varLowBound).diagonal(dim1=-2, dim2=-1).diag_embed()

        # Assembles covariance with strict lower triangle
        # Also known as Cholesky factor, L.
        # To reconstruct variance L^T L
        offDiag = LVar.tril(diagonal=-1)
        LVar = diag + offDiag
        
        if returnHidden:
            return mean, LVar, hidden
        
        # mean = [batch_size, n_states]
        # LVar = [batch_size, n_states, n_states]
        
        # TODO: hardcoded batch
        # return torch.cat((mean, LVar.reshape(1, -1)), axis=1)
        return mean, LVar

class ParamNet(nn.Module):
    def __init__(self,networkParams,robotRange):
        super().__init__()
        # init fc network
        self.listed = self.listParams(robotRange)
        last_size = sum([item[1] for item in self.listed])
        self.layers = nn.ModuleList()
        self.device = 'cpu'
        for size in networkParams['robot_param']['fc_sizes']:
            self.layers.append(nn.Linear(last_size,size))
            self.layers.append(nn.LeakyReLU())
            last_size = size
        self.finalFC_mean = nn.Linear(last_size,networkParams['contextDim'])
        self.finalFC_std = nn.Linear(last_size,networkParams['contextDim'])
        self.fric_only = 'fric_only' in networkParams['robot_param'] and networkParams['robot_param']['fric_only']
        self.stochastic = 'stochastic' in networkParams['robot_param'] and networkParams['robot_param']['stochastic']

    def to(self,device):
        self.device = device
        return super().to(device)

    def forward(self,robotParams):
        connected = self.extractParams(robotParams)
        if self.fric_only:
            connected = torch.ones_like(connected)*connected[...,-5:-4]
            #connected = torch.ones_like(connected)*robotParams['tireContact']['lateralFriction']
        for layer in self.layers:
            connected = layer(connected)
        context_mean = self.finalFC_mean(connected).unsqueeze(0).unsqueeze(0)
        context_std = self.finalFC_std(connected).unsqueeze(0).unsqueeze(0)
        context_std = F.softplus(context_std)
        if not self.stochastic:
            context_std = torch.zeros_like(context_std)
        return context_mean,context_std

    def extractParams(self,robotParams):
        tensor_items = []
        for item in self.listed:
            val = robotParams
            for key in item[0]:
                val = val[key]
            if not isinstance(val,list):
                val = [val]
            tensor_items += val
        return torch.tensor(tensor_items).to(self.device)

    def listParams(self,params,prefix=tuple(),multiplier=1):
        if 'members' in params:
            multiplier *= len(params['members'])
        res = tuple()
        for key in params:
            if key == "members":
                continue
            if isinstance(params[key],dict):
                res += self.listParams(params[key],prefix=prefix+(key,),multiplier=multiplier)
            else:
                res += ((prefix + (key,),multiplier),)
        return res
