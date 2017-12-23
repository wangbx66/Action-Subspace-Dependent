import torch
from torch.autograd import Variable, grad

x = Variable(torch.ones((2,2)), requires_grad=True)
y = torch.sum(x**2)
g1 = grad(y, x, create_graph=True)[0]

g2 = torch.zeros(g1.size())
for idx in np.ndindex(g1.size()):
    g2[idx] = grad(g1[idx], x)
