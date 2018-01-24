import math
import torch
from torch.autograd import Variable
import numpy as np
from utils import use_gpu

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std, wi_list=None):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    if wi_list is None:
        return log_density.sum(1, keepdim=True)
    else:
        results = []
        for wi in wi_list:
            wi = Variable(wi.unsqueeze(1))
            results.append(torch.mm(log_density, wi))
        return torch.cat(results, dim=1)
