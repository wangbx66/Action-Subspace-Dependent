import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.k = 4
        self.fm = True
        self.fo = False

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
            self.A2 = nn.Linear(last_dim, sna_dim[1]**2)
        else:
            self.A2 = nn.Linear(last_dim, sna_dim[1] * self.k)
        self.A2.weight.data.mul_(0.1)
        self.A2.bias.data.mul_(0.0)

        self.affine_layer_state = nn.Linear(sna_dim[0], hidden_size_SA1)
        if self.fo:
            self.affine_layer_action = nn.Linear(sna_dim[1] + sna_dim[1]**2, hidden_size_SA1)
        else:
            self.affine_layer_action = nn.Linear(sna_dim[1]**2, hidden_size_SA1)
        last_dim = hidden_size_SA1
        for nh in hidden_size_SAN:
            self.affine_layers_VN.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.AN = nn.Linear(last_dim, 1)
        self.AN.weight.data.mul_(0.1)
        self.AN.bias.data.mul_(0.0)
        
        self.W = nn.Linear(3, 1)

    def forward(self, s, a):
        s1 = s
        for affine in self.affine_layers_V1:
            s1 = self.activation(affine(s1))
        Advantage_1 = (self.A1(s1) * a).sum(dim=1, keepdim=True)

        s2 = s
        for affine in self.affine_layers_V2:
            s2 = self.activation(affine(s2))
        s2 = self.A2(s2)
        a2 = torch.matmul(a.unsqueeze(2), a.unsqueeze(1))
        if self.fm:
            v2 = torch.matmul(s2.view(s.size()[0], -1, self.k), s2.view(s.size()[0], -1, self.k).transpose(1,2))
            Advantage_2 = (a2*v2).view(s.size()[0], -1).sum(dim=1, keepdim=True)
        else:
            v2 = s2.view(a.size()[0], a.size()[1], a.size()[1])
            Advantage_2 = (a2*v2).view(s.size()[0], -1).sum(dim=1, keepdim=True)

        sn = self.affine_layer_state(s)
        an = a2.view(s.size()[0], -1)
        if self.fo:
            an = self.affine_layer_action(torch.cat([a, an]))
        else:
            an = self.affine_layer_action(an)
        # add first or activate first
        xn = self.activation(sn) + self.activation(an)
        for affine in self.affine_layers_VN:
            xn = self.activation(affine(xn))
        Advantage_N = self.AN(xn)

        advantage = self.W(torch.cat([Advantage_1, Advantage_2, Advantage_N], dim=1))
        return advantage

    def so(self, s):
        s2 = s
        for affine in self.affine_layers_V2:
            s2 = self.activation(affine(s2))
        s2 = self.A2(s2)
        if self.fm:
            v2 = torch.matmul(s2.view(s.size()[0], -1, self.k), s2.view(s.size()[0], -1, self.k).transpose(1,2))
        else:
            v2 = s2.view(a.size()[0], a.size()[1], a.size()[1])
        return v2
