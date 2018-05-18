'''
In this module we're trying to acquire the domain partition information through the Hessian.
Suppose that we treat each dimension of the action space as a vertex in the graph, and the absolute value of the elements in the Hessian as the weight of the edge, it reduces to a min-cut problem.
Note that our min-cut problem, specifically, 
 - is a minimum k-cut
 - is weighted
 - is for an undirected graph
 - involves noise where almost all elements of Hessian are positive.

We start from the most simply min-cut solution: Karger's algorithm.
Karger's algorithm is for two partitions and unweighted, undirected graphs.
It randomly and repeatedly merges two nodes until there's only two nodes left and regards those two nodes as the bi-partition.
Conduct that multiple times and select the results with the minimum number of edges between the final two nodes.
For minimum k-cut, a simple greedy approach is with an approximation of 2-2/k.
For weighted graphs, Stoer-Wagner algorithm works as follows.
Suppose that for some nodes s and t a s-t min-cut is found.
Then either the partition is the solution of the original min-cut, or the solution could be found by merging s and t and solving the reduced problem with one less node.
To find a s-t min-cut for some s and t, it starts with a random single-node set A={a}, and repeatedly adds the node that has the largest combined weight with A to A.
Denote the last two nodes added to A as s and t, (t, V-t) is proofed to be an s-t min-cut.
'''

import numpy as np
import networkx as nx

import torch
torch.set_default_tensor_type('torch.DoubleTensor')


def min_k_cut(H, K, force_k=False):
    '''
    1. Respect only the lower triangle of Hessian H, which is the default option of networkx.from_numpy().
    2. The Hessian H(s) is associated with the state and we have the access of the H(s) function.
    3. The Hessian is with noise both from the FM system, the neural network system and the noise of RL.
    '''
    G = nx.from_numpy_matrix(H)
    if not force_k:
        partition = np.zeros(H.shape[0], dtype=np.int64)
        for idx, component in enumerate(nx.connected_components(G)):
            partition[list(component)] = idx
    else:
        subs = []
        for idx, component in enumerate(nx.connected_components(G)):
            subs.append(component)
        K -= len(subs)
        if K > 0:
            for idx, component in enumerate(subs):
                subs[idx] = nx.stoer_wagner(G.subgraph(component)) + (component,)
        while K > 0:
            K -= 1
            #handle cases where len(component)==1
            idx = subs.find(min(subs))
            subs.append(nx.stoer_wagner(G.subgraph(subs[idx][1][0])) + (subs[idx][1][0], ))
            subs.append(nx.stoer_wagner(G.subgraph(subs[idx][1][1])) + (subs[idx][1][1], ))
            del subs[idx]
            
    return partition


def k_connect_components(H, K):
    n = H.shape[0]
    c = np.arange(n)
    if K == n:
        return c
    v = set(range(n))
    tril = np.tril_indices(n, k=-1)
    e = H[np.tril_indices(n, k=-1)]
    for idx in np.argsort(e)[::-1]:
        s = tril[0][idx]
        t = tril[1][idx]
        if not c[s] == c[t]:
            cold = max(c[s], c[t])
            cnew = min(c[s], c[t])
            for node, partition in enumerate(c):
                if partition == cold:
                    c[node] = cnew
            #print(s, t, v)
            v.remove(cold)
        if len(v) <= K:
            z = np.zeros(n)
            for idx, element in enumerate(np.unique(c)):
                z[element] = idx
            for idx, element in enumerate(c):
                c[idx] = z[c[idx]]
            return(c)

def T2(advantage_net, state, action, action_prime, wi):
    a00 = action
    a01 = action * wi + action_prime * (1-wi)
    a10 = action * (1-wi) + action_prime * wi
    a11 = action_prime
    T = advantage_net(state, a00) + advantage_net(state, a11) - advantage_net(state, a10) - advantage_net(state, a01)
    return (T**2).mean()

def Tij(advantage_net, state, action, action_prime, i, j):
    a00 = action
    a01 = action.clone()
    a01[:, i] = action_prime[:, i]
    a10 = action.clone()
    a10[:, j] = action_prime[:, j]
    a11 = action.clone()
    a11[:, (i,j)] = action_prime[:, (i,j)]
    T = advantage_net(state, a00) + advantage_net(state, a11) - advantage_net(state, a10) - advantage_net(state, a01)
    return (T**2).mean()

def combinatorial(advantage_net, state, action, action_prime):
    n = action.shape[1]
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            H[i, j] = Tij(advantage_net, state, action, action_prime, i, j)
    return H

def submodular(advantage_net, state, action, action_prime):
    n = action.shape[1]
    l = [1,] + [2,]*(n-1)
    wi_cube = np.ndindex(*l)
    next(wi_cube) # remove all zero wi
    best_wi = None
    best_T2 = None
    for wi in wi_cube:
        wi = torch.Tensor(wi).unsqueeze(0)
        T2_value = T2(advantage_net, state, action, action_prime, wi)
        T2_value = T2_value / (wi.sum()*(n-wi.sum()))
        if best_T2 is None or T2_value < best_T2:
            best_T2 = T2_value
            best_wi = wi
    return best_wi.squeeze(0)    
    
if __name__ == '__main__':
    n = 6
    H = np.abs(np.random.randn(n, n))
    #H = (H + H.T) / 2
    #H = H - np.diag(np.diag(H))
    H = np.tril(H, k=-1)
    partition = min_k_cut(H, -1)
    print(partition)

#graphs = [nx.from_numpy_matrix(H)]
#cp = [ for G in graphs]






