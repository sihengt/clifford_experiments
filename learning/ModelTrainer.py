import asyncio
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from utils.StatusPrint import StatusPrint
from .architecture.dynamicsModel import AdaptiveDynamicsModel
from .architecture.multiRobotDataset import SampleLoader
from .architecture.tools import gausLogLikelihood

# What the network learns to predict for the last step of each window:
#   'residual': x_dot_sim - x_dot_kbm  (Corrective Residual Network, CRN)
#   'full':     x_dot_sim               (Pure Data-driven Network, PDN)
TARGETS = ('residual', 'full')


class WarmupInverseSqrtDecay(_LRScheduler):
    """
    Learning rate scheduler from the "Attention is All You Need" paper. Starts out with a constant learning rate for
    warmup_steps, then exponentially decays the learning rate.
    """
    def __init__(self, optimizer, warmup_steps, use_decay=True, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.use_decay = use_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1

        # Linear warmup
        if current_epoch <= self.warmup_steps:
            lr = [base_lr * current_epoch / self.warmup_steps for base_lr in self.base_lrs]

        # Inverse square root decay
        elif self.use_decay:
            lr = [base_lr * (self.warmup_steps ** 0.5) * (current_epoch ** -0.5) for base_lr in self.base_lrs]

        else:
            lr = [base_lr for base_lr in self.base_lrs]

        return lr


def kbm_nominal(state, action, l_f, l_r):
    """Torch version of the double-steer KBM (same equations as TrajProc csDSKBM)."""
    x, y, v, theta = torch.moveaxis(state , -1, 0)
    a, d_f, d_r    = torch.moveaxis(action, -1, 0)

    beta = torch.atan( (l_r * torch.tan(d_f) + l_f * torch.tan(d_r)) / (l_r + l_f))

    vx      = v * torch.cos(beta + theta)
    vy      = v * torch.sin(beta + theta)
    vdot    = a
    thdot   = (v * torch.cos(beta)) / (l_f + l_r) * (torch.tan(d_f) - torch.tan(d_r))

    return torch.stack((vx, vy, vdot, thdot), dim=-1)


def first(value):
    """train.yaml lists some settings per training phase. Only the first phase is used."""
    return value[0] if isinstance(value, (list, tuple)) else value


def sample_training_window(sample, seq_len, target, dynamics):
    """
    Draws one training example uniformly from a robot's collected data.

    Params:
        sample: [states (N, 4), actions (M, 3), xdot (N, 4)] as saved by the data collection client.
            states = [x, y, v, yaw], actions = [a, delta_f, delta_r], xdot = [x_dot, y_dot, v_dot, yaw_dot].
            Each trajectory ends with an inf action row.
        seq_len: window length (train.trainPredSeqLen)
        target: 'residual' or 'full' (see TARGETS)
        dynamics: nominal model f(state, action) -> x_dot, used for the residual target

    Returns:
        dict with
            stateWindow  (seq_len, 2): [v, yaw] for each step of the window
            actionWindow (seq_len, 3): action taken at each step of the window
            target       (1, 4):       target for the last step of the window
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    states, actions, xdot = (t.to(torch.float32) for t in sample)

    with torch.no_grad():
        # Row index of the inf action that ends each trajectory, with -1 in front so trajectory 0 starts at row 0.
        traj_end = (actions[:, 0] == torch.inf).to('cpu')
        traj_end[-1] = True
        traj_end = torch.cat((torch.tensor([-1]), torch.arange(actions.shape[0])[traj_end]), dim=0)

        # Running total of the number of windows in each trajectory.
        n_windows = torch.clamp(traj_end[1:] - traj_end[:-1] - seq_len, min=0).cumsum(dim=0)

        # Pick a window, then find which trajectory it is in and where it starts.
        choice = torch.randint(1, n_windows[-1], (1,))
        choice_traj = torch.arange(len(n_windows))[n_windows > choice].min()
        choice_index = choice - torch.cat((torch.tensor([0]), n_windows))[choice_traj]
        start = traj_end[choice_traj] + 1 + choice_index
        end = start + seq_len
        last = end - 1

        window_target = xdot[last, :]
        if target == 'residual':
            window_target = window_target - dynamics(states[last, :], actions[last, :])

    return {'stateWindow': states[start:end, 2:],
            'actionWindow': actions[start:end, :],
            'target': window_target}


class ModelTrainer(object):
    """
    Trains the LSTM dynamics model on data that the server collects from simulation clients.
    Saves the weights to <trainDir>/adm.pt every 1000 iterations.
    """
    TENSORS = 0
    ROBOT_KEY = 1

    def __init__(self, server, params, log_dir=None):
        """
        Params:
            server: learning.server.Server (provides the simulation task queue)
            params: dict with trainDir, dataDir and the loaded configs (train, network, controls, mpc)
            log_dir: TensorBoard dir. Default: runs/<time>_<host><trainDir>
        """
        self.server = server
        self.params = params
        train_params = params['train']

        if train_params.get('target') not in TARGETS:
            raise ValueError(f"Set 'target' in {params['trainDir']}/config/train.yaml to 'residual' (CRN) "
                             f"or 'full' (PDN). Got {train_params.get('target')!r}.")
        self.target = train_params['target']

        l_f, l_r = params['mpc']['model']['l_f'], params['mpc']['model']['l_r']
        self.dynamics = lambda state, action: kbm_nominal(state, action, l_f, l_r)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.adaptiveDynamicsModel = AdaptiveDynamicsModel(params['network'], params['controls'], params['mpc']).to(self.device)
        self.adaptiveDynamicsModel.train()

        self.adm_file = os.path.join(params['trainDir'], 'adm.pt')
        if train_params['useOldModel']:
            self.adaptiveDynamicsModel.load_state_dict(torch.load(self.adm_file))

        self.model_params = list(self.adaptiveDynamicsModel.parameters())
        self.optimizer = Adam(self.model_params, lr=first(train_params['lrs']))
        self.scheduler = WarmupInverseSqrtDecay(self.optimizer,
                                                first(train_params['warmup_steps']),
                                                use_decay=first(train_params['use_decay']))
        self.grad_clip = first(train_params['gradClips'])

        self.writer = SummaryWriter(log_dir=log_dir, comment=params['trainDir'])

    async def run_training(self):
        train_it = 0
        while True:
            # Every batchTrainIts iterations: wait for the simulations, reload data, queue the next batch.
            if train_it % self.params['train']['batchTrainIts'] == 0:
                torch.cuda.empty_cache()
                while not self.server.checkSimFinished():
                    StatusPrint('[ModelTrainer] Pending simTasks {}, runningSims {}'.format(
                        len(self.server.simTaskQueue),
                        len(self.server.runningSims)), isTemp=True)
                    await asyncio.sleep(1)

                StatusPrint('[ModelTrainer] reloading simulation data')
                self.reloadData()
                await self.server.addBatchToSimQueue()

            dataset_sample = self.samples.getSample()
            loss = self.train_step(dataset_sample[self.TENSORS])

            StatusPrint('trainit: ', train_it, ' loss: ', loss, isTemp=True)
            self.writer.add_scalar('Loss/train', loss, train_it)
            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], train_it)

            if train_it % 1000 == 0:
                torch.save(self.adaptiveDynamicsModel.state_dict(), self.adm_file)

            train_it += 1
            await asyncio.sleep(0.01)

    def train_step(self, sample):
        """
        One optimizer step on one window drawn from sample (see sample_training_window).
        The loss is the negative Gaussian log-likelihood of the target. Returns the loss.
        """
        sample = [tensor.to(self.device) for tensor in sample]
        window = sample_training_window(sample, self.params['train']['trainPredSeqLen'], self.target, self.dynamics)
        pred_mean, pred_L = self.adaptiveDynamicsModel(window['stateWindow'].unsqueeze(0),
                                                       window['actionWindow'].unsqueeze(0))
        loss = -gausLogLikelihood(pred_mean, pred_L, window['target'].unsqueeze(0)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if any(param.grad is not None and param.grad.isnan().any() for param in self.model_params):
            raise FloatingPointError(f'NaN gradient in the dynamics model (loss {loss.item()})')

        torch.nn.utils.clip_grad_norm_(self.model_params, self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def reloadData(self):
        """
        Updates self.samples with sampleLoader
        """
        if hasattr(self,'samples'):
            self.samples.reloadData()
        else:
            self.samples = SampleLoader(self.params['dataDir'], device=self.device)
