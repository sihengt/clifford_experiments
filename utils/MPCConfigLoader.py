import numpy as np
import casadi as cs
import yaml

from TrajProc.models.DSKBM import action_type

class MPCConfigLoader:
    def __init__(self, yaml_file):
        if type(yaml_file) == str:
            with open(yaml_file, 'r') as f:
                self.params = yaml.safe_load(f)
        else:
            self.params = yaml_file

    def construct_casadi_params(self):
        self.params["X_lb"]  = cs.DM([-cs.inf, -cs.inf, -cs.inf, -cs.inf],)
        self.params["X_ub"]  = cs.DM([cs.inf, cs.inf, cs.inf, cs.inf],)
        if action_type(self.params["model"]) == "speed_target":
            if "maxCmdAcc" not in self.params:
                raise ValueError("maxCmdAcc is required when model.action is speed_target")
            self.params["U_lb"]  = cs.DM([0.0, -self.params["maxSteer"], -self.params["maxSteer"]])
            self.params["U_ub"]  = cs.DM([self.params["maxSpeed"], self.params["maxSteer"], self.params["maxSteer"]])
            self.params["dU_b"]  = cs.DM([self.params["maxCmdAcc"], self.params["maxDSteer"], self.params["maxDSteer"]])
        else:
            self.params["U_lb"]  = cs.DM([-self.params["maxAcc"], -self.params["maxSteer"], -self.params["maxSteer"]])
            self.params["U_ub"]  = cs.DM([self.params["maxAcc"], self.params["maxSteer"], self.params["maxSteer"]])
            self.params["dU_b"]  = cs.DM([self.params["maxDAcc"], self.params["maxDSteer"], self.params["maxDSteer"]])
        self.params["Q"]     = cs.DM(np.diag(self.params["Q"]))
        self.params["Qf"]    = cs.DM(np.diag(self.params["Qf"]))
        self.params["R"]     = cs.DM(np.diag(self.params["R"]))
        self.params["R_"]    = cs.DM(np.diag(self.params["R_"]))
        return self.params