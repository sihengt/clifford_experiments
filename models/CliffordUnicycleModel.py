from .UnicycleModel import UnicycleModel
import torch

class CliffordUnicycleModel(UnicycleModel):
    def __init__(self, robot_params, wheel_base=0.75, wheel_radius=0.1):
        super().__init__(self)
        self.driveParams = robot_params['drive']
        self.steerParams = robot_params['steer']
        self.wheelRadius = wheel_radius # TODO: HARDCODED TO MATCH URDF
        self.wheelBase = wheel_base

    """
    Clifford's actions are currently given by throttle, front steering, and rear steering. This 
    function takes a Clifford action and maps it into velocity and angular velocity needed by 
    the unicycle model.
    """
    def action_to_U(self, action):
        # Get linear_velocity command

        # These are all scaled between [-1, 1] and have to be mapped.
        throttle, front_steer, rear_steer = action
        
        # Mapping throttle to wheel velocity
        # omega_wheel in rad/s
        omega_wheel = throttle * self.driveParams['scale']
        front_ang = front_steer * self.steerParams['scale']
        rear_ang = rear_steer * self.steerParams['scale']

        # Reference: https://www.ntu.edu.sg/home/edwwang/confpapers/wdwicar01.pdf
        # https://viewer.mathworks.com/?viewer=live_code&url=https%3A%2F%2Fse.mathworks.com%2Fmatlabcentral%2Fmlc-downloads%2Fdownloads%2F33cfee76-0bb0-42ce-a1bc-46cc156d43f7%2F701dee20-fdf5-4012-aea5-b7b75748cfe0%2Ffiles%2Fdoc%2FmrsDocFourWheelSteer.mlx&embed=web
        v_cmd = self.wheelRadius/2 * omega_wheel * (torch.cos(front_ang) + torch.cos(rear_ang))
        omega_cmd = self.wheelRadius/self.wheelBase * omega_wheel * (torch.sin(front_ang) - torch.sin(rear_ang))

        return {"v_cmd": v_cmd, "omega_cmd": omega_cmd}
    