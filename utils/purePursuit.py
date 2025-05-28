import torch
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.planarRobotState import getRelativeState

def wrapAng(ang):
    ang = ang%(2*torch.pi)
    ang[ang > torch.pi] = ang[ang > torch.pi] - 2*torch.pi
    #if ang > torch.pi:
    #    ang -= 2*torch.pi
    return ang

def calcSteerAngle(relX,relY):
    steer_ang = torch.atan2(relX,relY)
    if torch.abs(steer_ang) > torch.pi/2.0:
        steer_ang -= torch.sign(steer_ang)*torch.pi
    return steer_ang

class purePursuit(object):
    def __init__(self,wheelBase,steerScale,throttle_pid,maxErrorSum=1,maxThrottle=1,lookAhead=0):
        self.wheelBase = wheelBase
        self.steerScale = steerScale
        self.throttle_pid = throttle_pid
        self.maxErrorSum = maxErrorSum
        self.maxThrottle = maxThrottle
        self.lookAhead = lookAhead
        self.resetTracker()

    def resetTracker(self):
        self.lastError = 0
        self.errorSum = 0

    def getThrottlePID(self,newError):
        de = self.lastError - newError
        self.lastError = newError
        self.errorSum = torch.clip(self.errorSum + newError, -self.maxErrorSum, self.maxErrorSum)
        throttle = newError*self.throttle_pid[0] + self.errorSum*self.throttle_pid[1] + de*self.throttle_pid[2]
        return torch.clip(throttle,-self.maxThrottle,self.maxThrottle)

    def calcTurnRelative(self,target):
        d = torch.norm(target[:2])
        turnRadius = d/(2*torch.sin(target[2]/2))
        ang = torch.atan2(target[1],target[0]) + (torch.pi/2.0 - target[2]/2.0)
        cen_x = (turnRadius)*torch.cos(ang)
        cen_y = (turnRadius)*torch.sin(ang)

        front_ang = calcSteerAngle(self.wheelBase/2 - cen_x, cen_y)
        rear_ang = -calcSteerAngle(-self.wheelBase/2 - cen_x, cen_y)

        return front_ang,rear_ang,cen_x,cen_y,turnRadius.abs()

    def calcTurnAbsolute(self,start,target):
        relTarget = getRelativeState(start,target)
        front_ang,rear_ang,rel_cen_x,rel_cen_y,R = self.calcTurnRelative(relTarget)
        sinHead = start[2].sin()
        cosHead = start[2].cos()
        cen_x = start[0] + cosHead*rel_cen_x  - sinHead*rel_cen_y
        cen_y = start[1] + sinHead*rel_cen_x  + cosHead*rel_cen_y
        return front_ang,rear_ang,cen_x,cen_y,R

    def trackTraj(self,currentState,refTraj):
        targetIndex = min(refTraj.shape[-2]-1,self.lookAhead-1)
        target = refTraj[targetIndex,:]
        front_ang,rear_ang,cen_x,cen_y,R = self.calcTurnAbsolute(currentState,target)
        front_steer = torch.clip(front_ang/self.steerScale,-1,1)
        rear_steer = torch.clip(rear_ang/self.steerScale,-1,1)

        #startAng = torch.atan2(currentState[1]-cen_y,currentState[0]-cen_x)
        #goalAng = torch.atan2(target[1] - cen_y, target[0] - cen_x)
        #travelAng = wrapAng(goalAng-startAng)
        #travelDist = torch.abs(R)*travelAng*torch.sign(wrapAng(currentState[2]-startAng))
        front_dir_ang = currentState[...,2:] + front_steer
        rear_dir_ang = currentState[...,2:] - rear_steer
        front_dir = torch.cat((front_dir_ang.cos(),front_dir_ang.sin()),dim=-1)
        rear_dir = torch.cat((rear_dir_ang.cos(),rear_dir_ang.sin()),dim=-1)
        com_dir = (front_dir+rear_dir)/2.0
        com_dir = com_dir/torch.norm(com_dir)
        throttleDir = refTraj[0,:2] - currentState[...,:2]
        throttleDist = torch.sum(throttleDir*com_dir)

        #forwardDist = getRelativeState(currentState,refTraj[0,:])[0]
        throttle = self.getThrottlePID(throttleDist)
        return torch.tensor([throttle,front_steer,rear_steer])

    def calcSteerTraj(self, currentState, target, step_size, maxDist=torch.inf, steerLimit=1, forward_only=False):
        if torch.norm(currentState[...,:2] - target[...,:2]) < 0.0001:
            if wrapAng(currentState[...,2] - target[...,2]).abs() < 0.001:
                return currentState.unsqueeze(0), torch.norm(currentState - target)
            else:
                return None, torch.inf

        if target.shape[-1] == 2:
            front_ang,rear_ang,cen_x,cen_y,R = self.calcTurnLoc(currentState,target)
        else:
            front_ang,rear_ang,cen_x,cen_y,R = self.calcTurnAbsolute(currentState,target)

        if torch.abs(front_ang) > self.steerScale*steerLimit or torch.abs(rear_ang) > self.steerScale*steerLimit:
            return None,torch.inf

        R = torch.abs(R)
        startAng = torch.atan2(currentState[1]-cen_y,currentState[0]-cen_x)
        goalAng = torch.atan2(target[1]-cen_y,target[0]-cen_x)
        if forward_only:
            travelDir = torch.sign(wrapAng(currentState[2] - startAng))
            travelAng = wrapAng(goalAng-startAng)
            if travelAng*travelDir < 0:
                travelAng += travelDir*2*torch.pi
        else:
            travelAng = wrapAng(goalAng-startAng)

        if travelAng.abs() > maxDist/R:
            travelAng = maxDist/R*torch.sign(travelAng)
        numSteps = int(torch.ceil(travelAng.abs()/(step_size/R)))

        angs = torch.linspace(0,travelAng,numSteps+1)[1:]

        theta = angs + currentState[2]
        theta = wrapAng(theta)
        x = cen_x + torch.cos(angs + startAng)*R
        y = cen_y + torch.sin(angs + startAng)*R
        traj = torch.cat((x.unsqueeze(-1),
                            y.unsqueeze(-1),
                            theta.unsqueeze(-1)),dim=-1)

        return traj, torch.abs(travelAng*R)

    def calcTurnLoc(self,currentState,target):
        target = torch.cat((target,torch.zeros_like(target[...,-1:])),dim=-1)
        relTarget = getRelativeState(currentState,target)[...,:-1]
        L_square = torch.sum((target[...,:2] - currentState[...,:2])**2,dim=-1)
        R = L_square/2/relTarget[...,1]
        rel_y = R
        rel_x = torch.zeros_like(rel_y)

        front_ang = calcSteerAngle(self.wheelBase/2 - rel_x, rel_y)
        rear_ang = +calcSteerAngle(-self.wheelBase/2 - rel_x, rel_y)
        sinHead = currentState[...,2].sin()
        cosHead = currentState[...,2].cos()
        cen_x = currentState[...,0] + cosHead*rel_x  - sinHead*rel_y
        cen_y = currentState[...,1] + sinHead*rel_x  + cosHead*rel_y
        return front_ang,rear_ang,cen_x,cen_y,R.abs()
    
    def calcSteerTraj_forward(self,currentState,target,step_size,maxDist=torch.inf):
        front_ang,rear_ang,cen_x,cen_y,R = self.calcTurnAbsolute(currentState,target)
        if torch.abs(front_ang) > self.steerScale or torch.abs(rear_ang) > self.steerScale:
            return None,torch.inf

        R = torch.abs(R)
        startAng = torch.atan2(currentState[1]-cen_y,currentState[0]-cen_x)
        goalAng = torch.atan2(target[1]-cen_y,target[0]-cen_x)
        travelDir = torch.sign(wrapAng(currentState[2] - startAng))
        travelAng = wrapAng(goalAng-startAng)
        if travelAng*travelDir < 0:
            travelAng += travelDir*2*torch.pi

        if travelAng.abs() > maxDist/R:
            travelAng = maxDist/R*torch.sign(travelAng)
        numSteps = int(torch.ceil(travelAng.abs()/(step_size/R)))

        angs = torch.linspace(0,travelAng,numSteps+1)[1:]

        theta = angs + currentState[2]
        theta = wrapAng(theta)
        x = cen_x + torch.cos(angs + startAng)*R
        y = cen_y + torch.sin(angs + startAng)*R
        traj = torch.cat((x.unsqueeze(-1),
                            y.unsqueeze(-1),
                            theta.unsqueeze(-1)),dim=-1)

        return traj, torch.abs(travelAng*R)


def step(start,goal,front_ang,rear_ang,wheelBase,step_size):
    if torch.norm(front_ang+rear_ang) < 0.0001:
        relGoal = getRelativeState(start,goal)
        relTrans = torch.tensor([torch.cos(front_ang),torch.sin(front_ang),0])*step_size
        return transitionState(start,relTrans)
    tanf = torch.tan(front_ang)
    tanr = torch.tan(rear_ang)
    cen_x = wheelBase/2.0*(tanr-tanf)/(tanr+tanf)
    cen_y = (cen_x - wheelBase/2.0)/tanf
    R = torch.norm(start[:2] - torch.tensor([cen_x,cen_y]))
    startAng = torch.atan2(start[1]-cen_y,start[0]-cen_x)
    goalAng = torch.atan2(goal[1]-cen_y,goal[0]-cen_x)
    angDiff = goalAng - startAng
    dAng = step_size/R*torch.sign(angDiff)
    x = cen_x + torch.cos(startAng+dAng)*R
    y = cen_y + torch.sin(startAng+dAng)*R
    theta = start[-1] + dAng
    return torch.cat((x.unsqueeze(-1),
                    y.unsqueeze(-1),
                    theta.unsqueeze(-1)),dim=-1)

def brute_force_steer(start,goal,speed,steerScale,wheelBase,maxSteps = torch.inf):
    path = [start]
    while torch.norm(path[-1][:2]-goal[:2]) >= speed and len(path) < maxSteps:
        front_ang,rear_ang,cen_x,cen_y,turnRadius = pure_pursuit_turn(path[-1],goal,wheelBase)
        if torch.abs(front_ang) > steerScale or torch.abs(rear_ang) > steerScale:
            front_ang = torch.clip(front_ang,-steerScale,steerScale)
            rear_ang = torch.clip(rear_ang,-steerScale,steerScale)
            path.append(step(path[-1],goal,front_ang,rear_ang,wheelBase,speed))
        else:
            pathEnd = reachGoal(path[-1],goal,cen_x,cen_y,speed)
            path = [item.unsqueeze(0) for item in path]
            path.append(pathEnd)
            path = torch.cat(path,dim=0)
            break
    if type(path) == list:
        path = torch.cat([item.unsqueeze(0) for item in path],dim=0)
    if maxSteps < torch.inf:
        return path[:maxSteps,:]
    return path

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    goalRange = torch.tensor([[-10,10],[-10,10],[-3.14,3.14]])
    pure_pursuit = purePursuit(1,1,1)
    traj = None
    while traj is None:
        start = torch.rand(goalRange.shape[0])*(goalRange[:,1]-goalRange[:,0]) + goalRange[:,0]
        #start = torch.zeros(3)
        target = torch.rand(goalRange.shape[0])*(goalRange[:,1]-goalRange[:,0]) + goalRange[:,0]
        traj = pure_pursuit.calcSteerTraj(start,target,0.1)

    plt.arrow(start[0],start[1],start[2].cos(),start[2].sin(),head_width=0.1,head_length=0.1,fc='green',ec='green')
    plt.arrow(target[0],target[1],target[2].cos(),target[2].sin(),head_width=0.1,head_length=0.1,fc='red',ec='red')
    for i in range(traj.shape[0]):
        plt.arrow(traj[i,0],traj[i,1],traj[i,2].cos()/2.0,traj[i,2].sin()/2.0,head_width=0.1,head_length=0.1,fc='black',ec='black')
    pure_pursuit.trackTraj(start,traj)

    plt.show()

