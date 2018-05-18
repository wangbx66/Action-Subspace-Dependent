import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal

from utils.math import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', scale_cov=1):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.raw_cov = nn.Parameter(torch.randn(action_dim, action_dim) * scale_cov)
        
        #import pdb; pdb.set_trace()

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_cov = torch.mm(self.raw_cov.t(), self.raw_cov) + 0.00001 * torch.eye(action_mean.size()[1])
        action_cov = action_cov.expand((action_mean.size()[0], action_mean.size()[1], action_mean.size()[1]))
        
        return action_mean, None, action_cov

    def select_action(self, x):
        action_mean, _, action_cov = self.forward(x)
        action_dist = MultivariateNormal(action_mean, action_cov)
        
        return action_dist.sample((1, ))[0] 

    '''
    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    '''

    def get_log_prob(self, x, actions, wi_list=None):
        action_mean, _, action_cov = self.forward(x)
        return multi_normal_log_density(actions, action_mean, action_cov, wi_list)

    '''
    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}
    '''

