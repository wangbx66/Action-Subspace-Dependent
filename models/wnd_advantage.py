import torch.nn as nn
import torch.nn.functional as F


class Advantage(nn.Module):
    def __init__(self, sna_dim, hidden_size=(128, 128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers_V1 = nn.ModuleList()
        self.affine_layers_V2 = nn.ModuleList()
        self.affine_layers_VN = nn.ModuleList()

        last_dim = sna_dim[0]
        for nh in hidden_size[1:]:
            self.affine_layers_V1.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.A1 = nn.Linear(last_dim, sna_dim[1])
        self.A1.weight.data.mul_(0.1)
        self.A1.bias.data.mul_(0.0)

        last_dim = sna_dim[0]
        for nh in hidden_size[1:]:
            self.affine_layers_V1.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.A2 = nn.Linear(last_dim, sna_dim[1] ** 2)
        self.A2.weight.data.mul_(0.1)
        self.A2.bias.data.mul_(0.0)

        self.affine_layer_state = nn.Linear(sna_dim[0], hidden_size[0])
        self.affine_layer_action = nn.Linear(sna_dim[1] + sna_dim[1]**2, hidden_size[0])
        last_dim = hidden_size[0]        
        for nh in hidden_size[1:]:
            self.affine_layers_VN.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.AN = nn.Linear(last_dim, 1)
        self.AN.weight.data.mul_(0.1)
        self.AN.bias.data.mul_(0.0)
        
        self.W = nn.Linear(3, 1)

    def forward(self, s, a):
        s1 = s
        for affine in self.affine_layers_VN:
            s1 = self.activation(affine(s1))
        Advantage_1 = (self.A1(s1) * a).sum()

        s2 = s
        for affine in self.affine_layers_VN:
            s2 = self.activation(affine(s2))
        a2 = torch.mm(a.unsqueeze(1), a.unsqueeze(1).t()).view(sna_dim[1] ** 2)
        Advantage_2 = (self.A2(s2) * a2).sum()

        sn = self.affine_layer_state(s)
        an = self.affine_layer_action(torch.cat([a, a2]))
        xn = self.activation(sn + an)
        for affine in self.affine_layers_VN:
            xn = self.activation(affine(xn))
        Advantage_N = self.AN(xn)

        advantage = self.W(torch.cat([Advantage_1, Advantage_2, Advantage_N]))
        return advantage

