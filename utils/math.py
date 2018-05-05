import math
import torch
from torch.autograd import Variable
import numpy as np
from utils import use_gpu
from torch.distributions import MultivariateNormal

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
            results.append(sity, wi)
        return torch.cat(results, dim=1)

def multi_normal_log_density(x, mean, cov, wi_list=None):
    #import pdb; pdb.set_trace()
    if wi_list is None:
        dist = MultivariateNormal(mean, cov)
        return dist.log_prob(x)
    else:
        results = []
        for wi in wi_list:
            idx = np.argwhere(wi).squeeze(0)
            meani = mean[:,idx]
            covi = cov[:,idx,:][:,:,idx]
            xi = x[:,idx]
            if use_gpu:
                meani, covi, xi = meani.cpu(), covi.cpu(), xi.cpu()
            dist = MultivariateNormal(meani, covi)
            lp = dist.log_prob(xi)
            results.append(lp.unsqueeze(1))
        log_prob = torch.cat(results, dim=1)
        if use_gpu:
            return log_prob.cuda()
        else:
            return log_prob
