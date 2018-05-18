import os
import sys
sys.path.append(os.path.expanduser('~/Action-Subspace-Dependent'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
torch.set_default_tensor_type('torch.DoubleTensor')

from models.mlp_policy_full import Policy
from utils.math import *

'''
[[ 63.18096649  22.92372827  62.53395827  89.64787352]
 [ 22.92372827  32.00559271  44.57377904  48.34455617]
 [ 62.53395827  44.57377904  97.73739999 102.9361416 ]
 [ 89.64787352  48.34455617 102.9361416  139.55102926]]
[[33.18428169 22.30913655]
 [22.30913655 68.04360642]]
'''

def T2(advantage_net, state, action, action_prime, wi):
    a00 = action
    a01 = action * wi + action_prime * (1-wi)
    a10 = action * (1-wi) + action_prime * wi
    a11 = action_prime
    T = advantage_net(state, a00) + advantage_net(state, a01) - advantage_net(state, a10) - advantage_net(state, a11)
    return (T**2).mean()

B = 10000
n = 6
policy_net = torch.load('policy_net')
advantage_net = torch.load('advantage_net')
state = torch.zeros(10000, n)
action = policy_net.select_action(state)
action_prime = policy_net.select_action(state)
l = [1,] + [2,]*(n-1)
wi_cube = np.ndindex(*l)
for wi in wi_cube:
    wi = torch.Tensor(wi).unsqueeze(0)
    T2_value = T2(advantage_net, state, action, action_prime, wi).data.cpu().numpy()
    print(wi.data.cpu().numpy(), T2_value, T2_value / (wi.sum()*(n-wi.sum())).numpy())
