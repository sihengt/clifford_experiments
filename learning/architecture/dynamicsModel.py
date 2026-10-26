import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveDynamicsModel(nn.Module):
    """
    LSTM dynamics model. Reads a window of (state features, action) pairs and predicts, for the last step,
    a Gaussian over a 4-dim output (x_dot residual or full x_dot, see ModelTrainer.TARGETS).

    Input per step: network.xdotDim state features ([v, yaw]) + network.actionDim actions.
    forward() returns the mean (B, 4) and the Cholesky factor L (B, 4, 4) of the covariance L L^T.
    """
    def __init__(self, networkParams, controlParams, mpcParams):
        super(AdaptiveDynamicsModel, self).__init__()
        self.networkParams = networkParams
        dynamics_params = networkParams['dynamicsModel']

        input_size = networkParams['xdotDim'] + networkParams['actionDim']
        self.lstm = nn.LSTM(input_size,
                            dynamics_params['hidden_size'],
                            dynamics_params['num_layers'],
                            dropout=dynamics_params['dropout'],
                            batch_first=True)

        self.meanFC = nn.Linear(dynamics_params['hidden_size'], networkParams['stateTransitionDim'])
        self.LVarFC = nn.Linear(dynamics_params['hidden_size'], networkParams['stateTransitionDim']**2)

        # Lower bound added to the diagonal of L.
        self.varLowBound = dynamics_params['varLowBound']

        # If True, the covariance does not depend on the input.
        self.staticVar = dynamics_params.get('staticVar', False)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x_window, action_window, context=None, hidden=None, returnHidden=False):
        """
        Params:
            x_window: (B, seq_len, xdotDim) state features
            action_window: (B, seq_len, actionDim) actions
            hidden: initial LSTM state (None for zeros)
            returnHidden: also return the final LSTM state
        """
        connected = torch.cat((x_window, action_window), dim=-1)

        connected, hidden = self.lstm(connected, hidden)
        connected = connected[:, -1, :]  # output at the last step of the window

        mean = self.meanFC(connected)  # (B, D)

        if self.staticVar:
            LVar = self.LVarFC(torch.zeros_like(connected))
        else:
            LVar = self.LVarFC(connected)
        LVar = LVar.view(*mean.shape, mean.shape[-1])  # (B, D, D)

        # Cholesky factor L: strictly positive diagonal plus the strict lower triangle.
        diag = (F.softplus(LVar) + self.varLowBound).diagonal(dim1=-2, dim2=-1).diag_embed()
        offDiag = LVar.tril(diagonal=-1)
        LVar = diag + offDiag

        if returnHidden:
            return mean, LVar, hidden
        return mean, LVar
