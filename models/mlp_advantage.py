import torch.nn as nn
import torch.nn.functional as F


class Advantage(nn.Module):
    def __init__(self, sna_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers_afterwards = nn.ModuleList()

        last_dim = sna_dim[0]
        self.affine_layer_state = nn.Linear(last_dim, hidden_size[0])
        last_dim = sna_dim[1]
        self.affine_layer_action = nn.Linear(last_dim, hidden_size[0])
        last_dim = hidden_size[0]
        for nh in hidden_size[1:]:
            self.affine_layers_afterwards.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.advantage_head = nn.Linear(last_dim, 1)
        self.advantage_head.weight.data.mul_(0.1)
        self.advantage_head.bias.data.mul_(0.0)


    def forward(self, s, a):
        s = self.affine_layer_state(s)
        a = self.affine_layer_action(a)
        x = self.activation(s + a)
        for affine in self.affine_layers_afterwards:
            x = self.activation(affine(x))
        advantage = self.advantage_head(x)
        return advantage

