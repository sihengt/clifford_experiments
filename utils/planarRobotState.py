import numpy as np
import torch


def convert_planar_world_frame_with_vel(state, position_only=False):
    """
    Converts a PyBullet base state [pos (3), quat xyzw (4), lin vel (3), ang vel (3), ...] to the planar state
    [x, y, v, yaw, x_dot, y_dot, yaw_dot] in the world frame, where v = |(x_dot, y_dot)|.
    With position_only=True returns np.array([xy, yaw]) instead.
    """
    
    # Converts state into a tensor
    if type(state) != torch.Tensor:
        state = torch.tensor(state)
    oShapePrefix = state.shape[0 : -1]
    
    state = state.view(-1,state.shape[-1])
    
    # Extracting quaternions
    qx = state[:, 3:4]
    qy = state[:, 4:5]
    qz = state[:, 5:6]
    qw = state[:, 6:7]
    
    # Extracting cartesian position
    xy = state[:, 0:2]
    
    # Getting heading in Euler Angles from quaternion.
    heading = torch.atan2(2*qx*qy + 2*qw*qz, qw*qw + qx*qx - qy*qy - qz*qz)
    
    if position_only:
        state = np.array([xy, heading])
    else:
        vel_mag = torch.sqrt(state[:, 7:8] ** 2 + state[:, 8:9] ** 2)
        vel_xy = state[:, 7:9]
        vel_yaw = state[:, 12:]
        state = torch.cat([xy, vel_mag, heading, vel_xy, vel_yaw], dim=-1)
    
    return state.view(*oShapePrefix, -1)
