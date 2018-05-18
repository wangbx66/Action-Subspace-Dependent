import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal

import os
import sys
sys.path.append(os.path.expanduser('~/Action-Subspace-Dependent'))
from models.mlp_policy_full import Policy
from utils.math import *

p = Policy(6, 3)
s = torch.ones(5, 6)
a = p.select_action(s)
mean, _, cov = p.forward(s)
wi_list = [torch.Tensor([1,1,0]), torch.Tensor([0,0,1])]
d = multi_normal_log_density(a, mean, cov, wi_list)
for param in p.parameters():
    print(param.grad)
