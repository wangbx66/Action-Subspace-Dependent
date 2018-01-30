import torch
from torch.autograd import Variable, grad
import numpy as np
from utils import use_gpu


def ppo_step(policy_net, value_net, advantage_net, optimizer_policy, optimizer_value, optimizer_advantage, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, lr_mult, lr, clip_epsilon, l2_reg, wi_list):
    A2C = True
    SE = not A2C
    SGR = False
    decay = True
    
    optimizer_policy.lr = lr * lr_mult
    optimizer_value.lr = lr * lr_mult
    optimizer_advantage.lr = lr * lr_mult
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = Variable(returns)
    for _ in range(optim_value_iternum):
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()
        if decay:
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update advantage"""
    advantages_target = Variable(advantages)
    for _ in range(optim_value_iternum):
        actions_var = Variable(actions, requires_grad=True)
        advantages_pred = advantage_net(Variable(states), actions_var)
        advantage_loss = (advantages_pred - advantages_target).pow(2).mean()
        if decay:
            if advantage_net.fm:
                for name, param in advantage_net.named_parameters():
                    if ('VN' in name) or ('AN' in name):
                        advantage_loss += param.pow(2).sum() * l2_reg
            else:
                for name, param in advantage_net.named_parameters():
                    if ('VN' in name) or ('AN' in name):
                        advantage_loss += param.pow(2).sum() * l2_reg
        optimizer_advantage.zero_grad()
        advantage_loss.backward()
        optimizer_advantage.step()

    """update policy"""
    advantages_var = Variable(advantages)
    action_bar = actions.mean(dim=0).unsqueeze(0).expand(actions.size()[0], -1)
    log_probs = policy_net.get_log_prob(Variable(states), Variable(actions), wi_list)
    surr1_components = []
    surr2_components = []
    surr_min = []
    if A2C:
        for cluster, wi in enumerate(wi_list):
            wi = wi.unsqueeze(0).expand(actions.size()[0], -1)
            action_i = Variable(action_bar * wi + actions * (1 - wi))
            advantages_baseline_i = advantage_net(Variable(states), action_i)
            ratio_i = torch.exp(log_probs[:, cluster].unsqueeze(1) - Variable(fixed_log_probs[:, cluster].unsqueeze(1)))
            s1 = ratio_i * (advantages_var - advantages_baseline_i)
            s2 = torch.clamp(ratio_i, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * (advantages_var - advantages_baseline_i)
            if SGR:
                surr_min.append(torch.min(s1, s2))
            else:
                surr1_components.append(s1)
                surr2_components.append(s2)
    elif SE:
        #First calculate \xi = (a - \mu(s)) / \sigma(s)
        #Then calculate \nabla_\theta\mu(s) + \nabla_\theta(\xi\sigma(s))
        va1 = Variable(actions, requires_grad=False)
        mean, logstd, std = policy_net(Variable(states))
        xi = Variable(((va1 - mean) / std).data, requires_grad=False)
        f = (mean + xi*std).unsqueeze(1)

        #g = [param.grad for param in policy_net.parameters()]
        va2 = Variable(actions, requires_grad=True)
        advantages_psi = advantage_net(Variable(states), va2)
        gaa = grad(advantages_psi.sum(), va2)
        gaa = Variable(gaa[0].data, requires_grad=False).unsqueeze(2)
        fta = f.matmul(gaa).squeeze(2)
        for cluster, wi in enumerate(wi_list):
            wi = wi.unsqueeze(0).expand(actions.size()[0], -1)
            action_i = Variable(actions * wi)
            action_negi = Variable(actions * (1 - wi), requires_grad=False)
            advantages_baseline_i = advantage_net(Variable(states), action_i+action_negi)
            ratio_i = torch.exp(log_probs[:, cluster].unsqueeze(1) - Variable(fixed_log_probs[:, cluster].unsqueeze(1)))
            s1 = ratio_i * (advantages_var - advantages_baseline_i)
            s2 = torch.clamp(ratio_i, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * (advantages_var - advantages_baseline_i)
            if SGR:
                surr_min.append(torch.min(s1, s2))
            else:
                surr1_components.append(s1)
                surr2_components.append(s2)
    if SGR:
        policy_surr = -(sum(surr_min)+fta).mean() if SE else -(sum(surr_min)).mean()
    else:
        surr1 = sum(surr1_components)
        surr2 = sum(surr2_components)
        surr = torch.min(surr1, surr2)
        policy_surr = -(surr+fta*surr/(surr1+1e-8)).mean() if SE else -torch.min(surr1, surr2).mean()
    
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()

    """calculate hessian"""
    actions_var = Variable(actions, requires_grad=True)
    g2 = advantage_net.so(Variable(states)).data
    '''
    advantages_pred = advantage_net(Variable(states), actions_var)
    g1 = grad(advantages_pred.sum(), actions_var, create_graph=True)[0]
    ag = g1.sum(dim=0)
    g2 = torch.zeros(g1.size()[0], ag.size()[0], ag.size()[0])
    for idx in range(ag.size()[0]):
        g2[:, idx, :] = grad(ag[idx], actions_var, retain_graph=True)[0].data
    '''
    
    g2m = g2.abs().mean(dim=0)
    return g2m
