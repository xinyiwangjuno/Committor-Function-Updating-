from torch import nn
import torch
from Muller import Mueller_System
# Models taken from 
# https://github.com/arturml/pytorch-wgan-gp/blob/master/wgangp.py


class CommittorNet(nn.Module):
    def __init__(self, d, n, unit=torch.relu, thresh=None, init_mode="meanfield"):
        super(CommittorNet, self).__init__()
        self.n = n
        self.d = d
        self.lin1 = nn.Linear(d, n, bias=False)
        self.unit = unit
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.lin2 = nn.Linear(n, 1, bias=False)
        self.thresh = thresh
        self.initialize(mode=init_mode)
        self.mus = Mueller_System(self.d)

    def initialize(self, mode="meanfield"):
        if mode == "meanfield":
            self.lin2.weight.data = torch.randn(
                self.lin2.weight.data.shape) / self.n
            self.renormalize()

    def forward(self, x):
        x = x.view(-1, self.d)
        q = x
        q = self.lin1(q.float())
        q = self.lin2(q.float())
        # q = self.unit(q.float())
        q = self.tanh1(q.float())
        q = self.tanh2(q.float())
        # q = self.tanh3(q.float())


        # dA2 = (x[:, 0] - self.mus.A[0]) ** 2 + (x[:, 1] - self.mus.A[1]) ** 2
        # dB2 = (x[:, 0] - self.mus.B[0]) ** 2 + (x[:, 1] - self.mus.B[1]) ** 2
        # IA = 0.5 - 0.5 * torch.tanh(1000 * (dA2 - self.mus.R ** 2))
        # IB = 0.5 - 0.5 * torch.tanh(1000 * (dB2 - self.mus.R ** 2))
        # q = (1 - IA) * (1 - IB) * torch.reshape(q, [-1]) + (1 - IA) * IB
        if self.thresh is not None:
            return self.thresh(torch.reshape(q, [-1]).float())
        else:
            return torch.reshape(q, [-1]).float()

    def renormalize(self):
        self.lin1.weight.data /= torch.norm(self.lin1.weight.data,
                                            dim=1).reshape(self.n, 1)



class Discriminator(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, n),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

    def forward(self, x):
        outputs = self.net(x.float())
        outputs = torch.reshape(outputs, [-1])
        return outputs.float()

