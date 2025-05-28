import torch
from torch.utils.data import Dataset,DataLoader, random_split,Sampler
import yaml
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from StatusPrint import StatusPrint

def countTimeSteps(actions):
    if actions.shape[0] == 0:
        return torch.tensor([0])
    endPoints = actions[:,0] == torch.inf
    endPoints = torch.cat((endPoints,torch.tensor([True],device=actions.device)))
    endPoints = torch.arange(len(endPoints))[endPoints]
    endPoints = torch.cat((torch.tensor([-1]),endPoints))
    return endPoints[1:]-endPoints[:-1]-1

class MultiRobotDataset(Dataset):
    def __init__(self, dataDir, botsToUse=None, device='cpu'):
        self.to(device)
        self.dataDir = dataDir
        self.nonEmptyKeys = set()
        self.reloadData(botsToUse)

    def reloadData(self, botsToUse=None):
        """
        Populates self.robotKeys, a list which contains indices of robots with data. Only 

        Params: 
            botsToUse: indices of robots to be used for training. If None, uses all robots within robotMeta.yaml.
        """
        StatusPrint('[MultiRobotDataset] Reloading Data')
        self.robotMeta = yaml.safe_load(open(os.path.join(self.dataDir,'robotMeta.yaml')))

        # self.robotKeys are list of robots to be used for sampling.
        if botsToUse is None:
            self.robotKeys = list(self.robotMeta.keys())
        else:
            self.robotKeys = botsToUse

        # For robots in self.robotKeys, if any of them have no data, remove them from the list.
        toRemove = []
        for key in self.robotKeys:
            if not key in self.nonEmptyKeys:
                if self.__getitem__(key)[0][0].shape[0] != 0:
                    self.nonEmptyKeys.add(key)
                else:
                    toRemove.append(key)
        for key in toRemove:
            self.robotKeys.remove(key)
        
        StatusPrint('loaded: ', len(self))

    def to(self,device):
        self.device = device
        return self

    def __len__(self):
        return len(self.robotKeys)
    
    def __getitem__(self, idx):
        """
        Always returns (data, self.robotKeys[idx])
        """
        fn = os.path.join(self.dataDir, 'trajData', str(self.robotKeys[idx]) + '.pt')
        data = torch.load(fn, map_location=self.device)
        return data, self.robotKeys[idx]

class SampleLoader:
    """
    Main purpose: wraps a DataLoader around the data read by MultiRobotDataset. Shuffles the data read by 
    MultiRobotDataset and returns data at each call of getSample.

    Also handles reloading of the MultiRobotDataset (i.e. when number of batchTrainIts have been hit).
    """

    def __init__ (self, dataDir, botsToUse=None, device='cpu'):
        self.dataDir = dataDir
        self.device = device
        self.samples = iter([])
        self.reloadData(botsToUse)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    def reloadData(self, botsToUse=None, reloadNow=False):
        self.needReloading = True
        self.botsToUse = botsToUse
        if reloadNow:
            self.reloadNow()

    def reloadNow(self):
        if not self.needReloading:
            return
        if hasattr(self,'data'):
            self.data.reloadData(self.botsToUse)
        # During the very first getSample() call, self.samples is iter([]), and self.data will be created here.
        else:
            self.data = MultiRobotDataset(self.dataDir, self.botsToUse, self.device)
        self.samples = iter([])
        self.needReloading = False

    def getSample(self):
        try:
            sample = next(self.samples)
        # 1. Very first getSample() call - initializes MultiRobotDataset
        # 2. Subsequent getSample() calls reload data, then creates a new DataLoader.
        except StopIteration:
            self.reloadNow()
            self.samples = iter(DataLoader(self.data, shuffle=True))
            sample = next(self.samples)
        
        # Removes the batch size added by the DataLoader for the data components.
        # From shape (1, ...) to just (...)
        sample = [item.squeeze(0) for item in sample[0]], sample[1]

        return sample

class adaptiveSampler(Sampler):
    def __init__(self):
        self.keyToIdx = {}
        self.keys = []
        self.smoothedLoss = torch.tensor([])
        self.alpha = 0

    def updateData(self,data):
        oldNumSamples = self.smoothedLoss.shape[0]
        oldAvg = 0 if oldNumSamples == 0 else self.smoothedLoss.mean()
        self.smoothedLoss = torch.cat((self.smoothedLoss,
                                        oldAvg*torch.ones(len(data.robotKeys)-oldNumSamples)),
                                        dim=0)
        idx = oldNumSamples
        for key in data.robotKeys:
            if not key in self.keyToIdx:
                self.keyToIdx[key] = idx
                self.keys.append(key)
                idx+=1

    def updateLoss(self,loss,key):
        idx = self.keyToIdx(key)
        self.smoothedLoss[idx] = (1-self.alpha)*loss + \
                            self.alpha*self.smoothedLoss[idx]
