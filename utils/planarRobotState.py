import torch
import torch.nn.functional as F
import time

def getLocalMap(states, worldMaps, worldMapBounds, localMapParams):
    sTime = time.time()
    device = states.device
    # build local map points
    localX = torch.linspace(-0.5,0.5,localMapParams['resolution'][0],device=device)*localMapParams['dim'][0]
    localY = torch.linspace(-0.5,0.5,localMapParams['resolution'][1],device=device)*localMapParams['dim'][1]
    localX,localY = torch.meshgrid(localX,localY,indexing='xy')

    # rotate and translate local map according to states
    x_pos = states[...,0].unsqueeze(-1).unsqueeze(-1)
    y_pos = states[...,1].unsqueeze(-1).unsqueeze(-1)
    c_head = states[...,2].unsqueeze(-1).unsqueeze(-1).cos()
    s_head = states[...,2].unsqueeze(-1).unsqueeze(-1).sin()
    head = states[...,2].unsqueeze(-1).unsqueeze(-1)
    worldX = c_head*localX - s_head*localY + x_pos
    worldY = s_head*localX + c_head*localY + y_pos
    worldCoords = torch.cat((worldX.unsqueeze(-1),worldY.unsqueeze(-1)),dim=-1)
    sTime = time.time()

    mapOrigins = worldMapBounds[:,0:1]
    mapDirs = worldMapBounds[:,1:3] - mapOrigins
    scaledMapDirs = mapDirs/torch.sum(mapDirs*mapDirs,dim=-1,keepdim=True)
    pixelXY = worldCoords.view(worldCoords.shape[0],-1,2) - mapOrigins
    pixelXY = pixelXY.bmm(scaledMapDirs.transpose(-1,-2))*2-1
    pixelXY = pixelXY.view(worldCoords.shape).to(worldMaps.dtype)
    
    localMap = F.grid_sample(worldMaps,pixelXY,align_corners=True,padding_mode='border')
    localMapOriginHeight = F.grid_sample(localMap,torch.zeros(localMap.shape[0],1,1,2,device=localMap.device),align_corners=True)
    localMap -= localMapOriginHeight
    return localMap

def get_relative_state(reference_state, target_state):
    """
    Returns relative state of the target state w.r.t. the reference state. 
    
    Calculates relative position and orientation in the frame of reference of the reference state. 
    
    Inputs:
        reference_state (torch.Tensor): assumed to have (at least) [x, y, orientation]
        target_state (torch.tensor): assumed to have [x, y, orientation]
    
    Returns:
    torch.tensor: The relative state tensor with shape [..., N]. First two components = rotated relative position. 
    Third component = relative orientation (dTheta) normalized within [-pi, pi].
    """
    dx = target_state[..., 0:1] - reference_state[..., 0:1]
    dy = target_state[..., 1:2] - reference_state[..., 1:2]

    # Gets dTheta and keeps within [-pi, pi]
    dTheta = target_state[..., 2:3] - reference_state[..., 2:3]
    dTheta = (dTheta + torch.pi) % (2.0 * torch.pi) - torch.pi
    
    # For applying R^{-1}(theta_ref)
    cos_head = torch.cos(reference_state[..., 2:3])
    sin_head = torch.sin(reference_state[..., 2:3])
    
    relative_state = torch.cat((
        dx * cos_head + dy * sin_head,
        -dx * sin_head + dy * cos_head,
        dTheta,
        target_state[..., 3:]
    ), dim=-1)

    return relative_state
    
def transitionState(currentState,prediction):
    # Getting the cos / sin of theta values
    cos_head = torch.cos(currentState[...,2:3])
    sin_head = torch.sin(currentState[...,2:3])
    dx = prediction[...,0:1]*cos_head - prediction[...,1:2]*sin_head
    dy = prediction[...,0:1]*sin_head + prediction[...,1:2]*cos_head
    newTheta = currentState[...,2:3]+prediction[...,2:3]
    newTheta = newTheta%(torch.pi*2.0)
    otherDir = newTheta>torch.pi
    newTheta = (newTheta-torch.pi*2.0)*(otherDir)+newTheta*(~otherDir)
    currentState = torch.cat((currentState[...,0:1]+dx,
                                currentState[...,1:2]+dy,
                                newTheta,
                                prediction[...,3:]),dim=-1)
    return currentState


def convert_planar(state, position_only=True):
    """
    Process state vector from quaternions into x, y, and heading OR x, y, heading and velocities.
    """
    
    # Converts state into a tensor
    if type(state) != torch.Tensor:
        state = torch.tensor(state)
    
    oShapePrefix = state.shape[0 : -1]
    
    state = state.view(-1,state.shape[-1])
    
    # Extracting quaternions
    qx = state[:,3:4]
    qy = state[:,4:5]
    qz = state[:,5:6]
    qw = state[:,6:7]
    
    # Extracting cartesian position
    xy = state[:,0:2]
    
    # Getting heading in Euler Angles from quaternion.
    heading = torch.atan2(2*qx*qy+2*qw*qz,qw*qw+qx*qx-qy*qy-qz*qz)
    
    if position_only:
        state = torch.cat((xy, heading),dim=-1)
    else:
        velX = state[:,7:8]*torch.cos(heading)+state[:,8:9]*torch.sin(heading)
        velY = state[:,7:8]*-torch.sin(heading)+state[:,8:9]*torch.cos(heading)
        velTheta = state[:,12:]
        state = torch.cat((xy,heading,velX,velY,velTheta),dim=-1)
    
    return state.view(*oShapePrefix,-1)

class planarRobotState(object):
    def __init__(self,startState,numParticles = 1,terrainMap=None,terrainParams=None):
        self.currentState = startState.repeat_interleave(numParticles,dim=0)
        self.originalDimPrefix = self.currentState.shape[0:-1]
        self.device = self.currentState.device
        if terrainMap is None:
            self.terrainMap = terrainMap
        else:
            self.terrainMap = terrainMap.repeat_interleave(numParticles,dim=0)
        self.terrainParams = terrainParams

    def updateState(self,prediction):
        cos_head = torch.cos(self.currentState[...,2:3])
        sin_head = torch.sin(self.currentState[...,2:3])
        dx = prediction[...,0:1]*cos_head - prediction[...,1:2]*sin_head
        dy = prediction[...,0:1]*sin_head + prediction[...,1:2]*cos_head
        newTheta = self.currentState[...,2:3]+prediction[...,2:3]
        newTheta = newTheta%(torch.pi*2.0)
        otherDir = newTheta>torch.pi
        newTheta = (newTheta-torch.pi*2.0)*(otherDir)+newTheta*(~otherDir)
        self.currentState = torch.cat((self.currentState[...,0:1]+dx,
                                    self.currentState[...,1:2]+dy,
                                    newTheta,
                                    prediction[...,3:]),dim=-1)
        return self.currentState

    def getPredictionInput(self):
        return self.currentState[...,3:]

    def getRelativeState(self,absoluteState):
        return getRelativeState(self.currentState,absoluteState)

    def getHeightMap(self,terrainMap=None,terrainParams=None):
        if terrainMap is None:
            terrainMap = self.terrainMap
        if terrainParams is None:
            terrainParams = self.terrainParams

        while len(terrainMap.shape) <= len(self.currentState.shape):
            terrainMap = terrainMap.unsqueeze(-3)
            terrainMap = terrainMap.repeat_interleave(self.currentState.shape[len(terrainMap.shape)-3],dim=-3)

        # define map pixel locations relative to robot
        pixelXRelRobot=torch.linspace(-terrainParams['senseDim'][0]/2,terrainParams['senseDim'][0]/2,
                                    terrainParams['senseResolution'][0],device=self.device)
        pixelYRelRobot=torch.linspace(-terrainParams['senseDim'][1]/2,terrainParams['senseDim'][1]/2,
                                    terrainParams['senseResolution'][1],device=self.device)
        pixelXRelRobot,pixelYRelRobot=torch.meshgrid(pixelXRelRobot,pixelYRelRobot,indexing='xy')
        pixelXRelRobot,pixelYRelRobot = pixelXRelRobot.unsqueeze(0),pixelYRelRobot.unsqueeze(0)

        # rotate and translate pixel locations to get world position
        posHeading = self.currentState[...,0:3].view(-1,3,1,1)
        cos_head = torch.cos(posHeading[:,2,:])
        sin_head = torch.sin(posHeading[:,2,:])
        worldX = cos_head*pixelXRelRobot - sin_head*pixelYRelRobot + posHeading[:,0,:]
        worldY = sin_head*pixelXRelRobot + cos_head*pixelYRelRobot + posHeading[:,1,:]

        # scale to terrainMapSize
        worldX = worldX/(terrainParams['mapWidth']-1)/terrainParams['mapScale']*2.
        worldY = worldY/(terrainParams['mapLength']-1)/terrainParams['mapScale']*2.
        pixelLocWorld = torch.cat((worldX.unsqueeze(-1),worldY.unsqueeze(-1)),dim=-1)
        terrainMap = terrainMap.view(-1,1,terrainMap.shape[-2],terrainMap.shape[-1])
        heightMaps = torch.nn.functional.grid_sample(terrainMap,pixelLocWorld,align_corners=True)
        robotPos = torch.cat((posHeading[:,0,:].unsqueeze(-1),posHeading[:,1,:].unsqueeze(-1)),dim=-1)
        robotGroundHeight = torch.nn.functional.grid_sample(terrainMap,robotPos,align_corners=True)
        heightMaps = heightMaps - robotGroundHeight
        if useChannel:
            return heightMaps.view(*self.originalDimPrefix,*heightMaps.shape[-3:])
        else:
            return heightMaps.view(*self.originalDimPrefix,*heightMaps.shape[-2:])
