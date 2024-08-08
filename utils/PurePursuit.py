import torch
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.planarRobotState import get_relative_state, transitionState
from utils.plot_car import plot_car

def wrap_ang(ang):
    """ Wraps the angle between [-pi, pi] """
    ang = ang % (2*torch.pi)
    ang[ang > torch.pi] = ang[ang > torch.pi] - 2*torch.pi
    return ang

def calc_steer_angle(rel_x, rel_y):
    """
    Calculates the angle from the y-axis.

    Inputs:
        rel_x: point's relative x coordinate.
        rel_y: point's relative y coordinate.

    Returns:
        steer_ang: kept within [-pi/2, pi/2]
    """
    # Getting angle from y-axis to point.
    steer_ang = torch.atan2(rel_x, rel_y)
    
    # Keeping result within -pi/2, pi/2
    if torch.abs(steer_ang) > torch.pi / 2.0:
        steer_ang -= torch.sign(steer_ang) * torch.pi
    
    return steer_ang

class PurePursuit:
    def __init__(self, wheel_base, steer_scale, throttle_pid, max_error_sum=1, max_throttle=1, look_ahead=0):
        self.wheelBase = wheel_base
        self.steerScale = steer_scale
        self.throttle_pid = throttle_pid
        self.maxErrorSum = max_error_sum
        self.maxThrottle = max_throttle
        self.lookAhead = look_ahead
        self.lastError = 0
        self.errorSum = 0

    def reset_tracker(self):
        self.lastError = 0
        self.errorSum = 0

    def get_throttle_pid(self, new_error):
        # TODO: huge assumption here that dt = 1.
        de = self.lastError - new_error
        self.lastError = new_error
        self.errorSum = torch.clip(self.errorSum + new_error, -self.maxErrorSum, self.maxErrorSum)
        throttle = new_error * self.throttle_pid[0] + self.errorSum * self.throttle_pid[1] + de * self.throttle_pid[2]

        return torch.clip(throttle, -self.maxThrottle, self.maxThrottle)

    # TODO: if you give this function control over tire limits...
    # TODO: why does Sean only check one side, and not the other? 
    def calc_turn_relative(self, target):
        """
        Calculates turning circle center relative to the vehicle. Calculates steering angles required to 
        track.

        Input:
            target (torch.tensor) - x, y and target orientation [-\pi to \pi]
        """
        # Lookahead distance
        lookahead_dist = torch.norm(target[:2])
        
        # Radius of turning arc (from definition of chord length)
        turn_radius = lookahead_dist / (2 * torch.sin(target[2]/2))

        # Getting angle to compute center of the turning circle.
        ang = torch.atan2(target[1], target[0]) + (torch.pi/2.0 - target[2] / 2.0)
        cen_x = (turn_radius) * torch.cos(ang)
        cen_y = (turn_radius) * torch.sin(ang)
        
        front_ang = calc_steer_angle(self.wheelBase/2 - cen_x, cen_y)
        rear_ang = -calc_steer_angle(-self.wheelBase/2 - cen_x, cen_y)

        return front_ang, rear_ang, cen_x, cen_y, turn_radius.abs()

    def calc_turn_absolute(self, start, target):
        """
        Input:
            start (torch.tensor) - contains n entries of [x, y, theta, possible other state]
            target (torch.tensor) - 
        """
        relative_state = get_relative_state(start, target)

        front_ang, rear_ang, rel_cen_x, rel_cen_y, turn_radius = self.calc_turn_relative(relative_state)
        
        # Transforms turning circle back into world coordinates.
        sinHead = start[2].sin()
        cosHead = start[2].cos()
        cen_x = start[0] + cosHead * rel_cen_x  - sinHead * rel_cen_y
        cen_y = start[1] + sinHead * rel_cen_x  + cosHead * rel_cen_y

        return front_ang, rear_ang, cen_x, cen_y, turn_radius

    def track_traj(self, current_state, ref_traj):
        """ 
        Main entrypoint for the Pure Pursuit algorithm
        """
        target_index = min(ref_traj.shape[-2] - 1, self.lookAhead - 1)
        target = ref_traj[target_index,:]
        
        # TODO: what is the point of this scaling then clipping?
        front_ang, rear_ang, cen_x, cen_y, turn_radius = self.calc_turn_absolute(current_state,target)
        front_steer = torch.clip(front_ang / self.steerScale, -1, 1)
        rear_steer  = torch.clip(rear_ang / self.steerScale, -1, 1)
        plot_car(current_state, ref_traj[-1, :], self.wheelBase, cen_x, cen_y, turn_radius, show_plot=False)

        # Getting normalized direction of COM (in this case an average of front / rear directions)
        front_dir_ang   = current_state[..., 2:] + front_steer
        rear_dir_ang    = current_state[..., 2:] - rear_steer
        front_dir       = torch.cat((front_dir_ang.cos(), front_dir_ang.sin()),dim=-1)
        rear_dir        = torch.cat((rear_dir_ang.cos(), rear_dir_ang.sin()),dim=-1)
        com_dir         = (front_dir + rear_dir)/2.0
        com_dir         = com_dir / torch.norm(com_dir)
        
        throttle_dir    = ref_traj[0, :2] - current_state[...,:2]
        # Projecting throttle_dir to the direction of COM.
        throttle_dist   = torch.sum(throttle_dir * com_dir)

        plt.arrow(current_state[0], current_state[1], front_dir[0], front_dir[1], head_width=0.1, fc='y', ec="y")
        plt.arrow(current_state[0], current_state[1], rear_dir[0], rear_dir[1], head_width=0.1, fc='b', ec="b")
        plt.arrow(current_state[0], current_state[1], com_dir[0], com_dir[1], head_width=0.1, fc='r', ec="r")
        plt.arrow(current_state[0], current_state[1], throttle_dir[0], throttle_dir[1], head_width=0.1, fc='m', ec="m")

        #forwardDist = getRelativeState(current_state,ref_traj[0,:])[0]
        throttle = self.get_throttle_pid(throttle_dist)

        return torch.tensor([throttle, front_steer, rear_steer])

    def gen_traj(self, current_state, target, step_size, maxDist=torch.inf, steerLimit=1, forward_only=False):
        """
        """
        # Proximity check - checks if distance AND orientation is close to target.
        # TODO: If distance is close to target, but orientation is off, this is treated as a failure.
        if torch.norm(current_state[..., :2] - target[..., :2]) < 0.0001:
            if wrap_ang(current_state[..., 2] - target[..., 2]).abs() < 0.001:
                return current_state.unsqueeze(0), torch.norm(current_state - target)
            else:
                return None, torch.inf

        # If orientation is provided in target, uses calc_turn_absolute.
        else:
            front_ang, rear_ang, cen_x, cen_y, turn_radius = self.calc_turn_absolute(current_state, target)
        
        # TODO: move into calc_turn_absolute function, which should have access to these.
        # Angles exceed steer limits
        if torch.abs(front_ang) > self.steerScale * steerLimit or torch.abs(rear_ang) > self.steerScale * steerLimit:
            return None, torch.inf

        turn_radius = torch.abs(turn_radius)
        startAng = torch.atan2(current_state[1] - cen_y, current_state[0] - cen_x)
        goalAng = torch.atan2(target[1] - cen_y, target[0] - cen_x)

        # DEBUG
        # plot_car(current_state, target, self.wheelBase, cen_x, cen_y, turn_radius, show_plot=True)

        if forward_only:
            travelDir = torch.sign(wrap_ang(current_state[2] - startAng))
            travelAng = wrap_ang(goalAng - startAng)
            if travelAng * travelDir < 0:
                travelAng += travelDir * 2 * torch.pi
        else:
            travelAng = wrap_ang(goalAng - startAng)

        if travelAng.abs() > maxDist / turn_radius:
            travelAng = maxDist / turn_radius * torch.sign(travelAng)
        n_steps = int(torch.ceil(travelAng.abs()/(step_size/turn_radius)))

        angs = torch.linspace(0, travelAng, n_steps + 1)[1:]
        theta = angs + current_state[2]
        theta = wrap_ang(theta)
        x = cen_x + torch.cos(angs + startAng) * turn_radius
        y = cen_y + torch.sin(angs + startAng) * turn_radius
        traj = torch.cat((
            x.unsqueeze(-1),
            y.unsqueeze(-1),
            theta.unsqueeze(-1)
        ), dim=-1)
    
        return traj, torch.abs(travelAng * turn_radius)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    goalRange = torch.tensor([[-10,10],[-10,10],[-3.14,3.14]])
    pure_pursuit = PurePursuit(1, 1, [1.5, 0.01, 0.1])
    traj = None
    while traj is None:
        start = torch.rand(goalRange.shape[0])*(goalRange[:,1]-goalRange[:,0]) + goalRange[:,0]
        #start = torch.zeros(3)
        target = torch.rand(goalRange.shape[0])*(goalRange[:,1]-goalRange[:,0]) + goalRange[:,0]
        target[2] = start[2] + torch.pi/2 + torch.pi/4
        # target[2] = start[2] + torch.pi/4
        traj, _ = pure_pursuit.gen_traj(start, target, 0.1)
    
    # plt.arrow(start[0], start[1], start[2].cos(), start[2].sin(), head_width=0.1, head_length=0.1, fc='green', ec='green')
    # plt.arrow(target[0], target[1], target[2].cos(), target[2].sin(), head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    for i in range(traj.shape[0]):
        plt.arrow(
            traj[i,0],
            traj[i,1],
            traj[i,2].cos()/2.0,
            traj[i,2].sin()/2.0,
            head_width=0.1,
            head_length=0.1,
            fc='black',
            ec='black'
        )
    pure_pursuit.track_traj(start, traj)

    plt.show()
