import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import debug, use_gpu


class Advantage(nn.Module):
    def __init__(self, sna_dim, hidden_size=(128, 128, 128), activation='tanh'):
        hidden_size = None
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        hidden_size_S1 = (128, 128)
        hidden_size_S2 = (128, 128)
        hidden_size_SA1 = 128
        hidden_size_SAN = (128, 128)
        
        self.fm = True
        self.fo = False
        
        if self.fm is False:
            self.k = sna_dim[1]
        elif sna_dim[1] >= 15:
            self.k = 4
        elif sna_dim[1] >= 12:
            self.k = 3
        else:
            self.k = sna_dim[1]
            self.fm = False
        # with tril(-1) the quadratic term excludes the square entries.
        self.lower = torch.ones(sna_dim[1], sna_dim[1]).tril(-1).nonzero().t()
        if use_gpu:
            self.lower = self.lower.cuda()

        self.affine_layers_V1 = nn.ModuleList()
        self.affine_layers_V2 = nn.ModuleList()
        self.affine_layers_VN = nn.ModuleList()

        last_dim = sna_dim[0]
        for nh in hidden_size_S1:
            self.affine_layers_V1.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.A1 = nn.Linear(last_dim, sna_dim[1])
        self.A1.weight.data.mul_(0.1)
        self.A1.bias.data.mul_(0.0)

        last_dim = sna_dim[0]
        for nh in hidden_size_S2:
            self.affine_layers_V2.append(nn.Linear(last_dim, nh))
            last_dim = nh
        if not self.fm:
            self.A2 = nn.Linear(last_dim, int(sna_dim[1] * (sna_dim[1]-1) / 2))
        else:
            self.A2 = nn.Linear(last_dim, sna_dim[1] * self.k)
        self.A2.weight.data.mul_(0.1)
        self.A2.bias.data.mul_(0.0)

        self.affine_layer_VNs = nn.Linear(sna_dim[0], hidden_size_SA1)
        if self.fo:
            self.affine_layer_VNa = nn.Linear(int(sna_dim[1] * (sna_dim[1]-1) / 2) + sna_dim[1], hidden_size_SA1)
        else:
            self.affine_layer_VNa = nn.Linear(int(sna_dim[1] * (sna_dim[1]-1) / 2), hidden_size_SA1)
        last_dim = hidden_size_SA1
        for nh in hidden_size_SAN:
            self.affine_layers_VN.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.AN = nn.Linear(last_dim, 1)
        self.AN.weight.data.mul_(0.1)
        self.AN.bias.data.mul_(0.0)
        
        self.W = nn.Linear(3, 1)

    def forward(self, s, a, verbose=False):
        #debug()
        s1 = s
        for affine in self.affine_layers_V1:
            s1 = self.activation(affine(s1))
        Advantage_1 = (self.A1(s1) * a).sum(dim=1, keepdim=True)

        s2 = s
        for affine in self.affine_layers_V2:
            s2 = self.activation(affine(s2))
        s2 = self.A2(s2)
        a2 = torch.matmul(a.unsqueeze(2), a.unsqueeze(1))[:, self.lower[0], self.lower[1]]
        if self.fm:
            v2 = torch.matmul(s2.view(s.size()[0], -1, self.k), s2.view(s.size()[0], -1, self.k).transpose(1,2))[:,self.lower[0],self.lower[1]]
            Advantage_2 = (a2*v2).view(s.size()[0], -1).sum(dim=1, keepdim=True)
        else:
            #v2 = s2.view(a.size()[0], a.size()[1], a.size()[1])
            #Advantage_2 = (a2*v2).view(s.size()[0], -1).sum(dim=1, keepdim=True)
            Advantage_2 = (a2*s2).sum(dim=1, keepdim=True)

        sn = self.affine_layer_VNs(s)
        #an = a2.view(s.size()[0], -1)
        an = a2
        if self.fo:
            an = self.affine_layer_VNa(torch.cat([a, an]))
        else:
            an = self.affine_layer_VNa(an)
        # add first or activate first
        xn = self.activation(sn) + self.activation(an)
        for affine in self.affine_layers_VN:
            xn = self.activation(affine(xn))
        Advantage_N = self.AN(xn)

        advantage = self.W(torch.cat([Advantage_1, Advantage_2, Advantage_N], dim=1))
        if verbose:
            advantage_1share = self.W(torch.cat([Advantage_1, Advantage_2-Advantage_2, Advantage_N-Advantage_N], dim=1)).abs().mean()
            advantage_2share = self.W(torch.cat([Advantage_1-Advantage_1, Advantage_2, Advantage_N-Advantage_N], dim=1)).abs().mean()
            advantage_Nshare = self.W(torch.cat([Advantage_1-Advantage_1, Advantage_2-Advantage_2, Advantage_N], dim=1)).abs().mean()
            advantage_bias = self.W(torch.cat([Advantage_1-Advantage_1, Advantage_2-Advantage_2, Advantage_N-Advantage_N], dim=1)).mean()
            advantage_1share = advantage_1share if use_gpu else advantage_1share.cpu()
            advantage_2share = advantage_2share if use_gpu else advantage_2share.cpu()
            advantage_Nshare = advantage_Nshare if use_gpu else advantage_Nshare.cpu()
            advantage_bias = advantage_bias if use_gpu else advantage_bias.cpu()
            print('Shares: {} {} {} {}'.format(advantage_1share.data.numpy().squeeze(), advantage_2share.data.numpy().squeeze(), advantage_Nshare.data.numpy().squeeze(), advantage_bias.data.numpy().squeeze()))
        
        return advantage

    def so(self, s):
        s2 = s
        for affine in self.affine_layers_V2:
            s2 = self.activation(affine(s2))
        s2 = self.A2(s2).data
        if self.fm:
            v2 = torch.matmul(s2.view(s.size()[0], -1, self.k), s2.view(s.size()[0], -1, self.k).transpose(1,2))
        else:
            v2 = torch.zeros(s.size()[0], self.k, self.k)
            v2[:,self.lower[0],self.lower[1]] = s2
            #v2 = s2.view(s.size()[0], self.k, self.k)
        return v2
