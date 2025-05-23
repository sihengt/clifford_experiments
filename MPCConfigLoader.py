import numpy as np
import casadi as cs
import yaml

class MPCConfigLoader:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            self.params = yaml.safe_load(f)

    def construct_casadi_params(self):
        self.params["X_lb"]  = cs.DM([-cs.inf, -cs.inf, -cs.inf, -cs.inf],)
        self.params["X_ub"]  = cs.DM([cs.inf, cs.inf, cs.inf, cs.inf],)
        self.params["U_lb"]  = cs.DM([-self.params["maxAcc"], -self.params["maxSteer"], -self.params["maxSteer"]])
        self.params["U_ub"]  = cs.DM([self.params["maxAcc"], self.params["maxSteer"], self.params["maxSteer"]])
        self.params["dU_b"]  = cs.DM([self.params["maxDAcc"], self.params["maxDSteer"], self.params["maxDSteer"]])
        self.params["Q"]     = cs.DM(np.diag(self.params["Q"]))
        self.params["Qf"]    = cs.DM(np.diag(self.params["Qf"]))
        self.params["R"]     = cs.DM(np.diag(self.params["R"]))
        self.params["R_"]    = cs.DM(np.diag(self.params["R_"]))
        return self.params