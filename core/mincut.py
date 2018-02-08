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


def min_k_cut(H, K):
    '''
    1. Respect only the lower triangle of Hessian H, which is the default option of networkx.from_numpy().
    2. The Hassian H(s) is associated with the state and we have the access of the H(s) function.
    3. The Hessian is with noise both from the FM system, the neural network system and the noise of RL.
    '''
    th = np.median(H[H>0])
    T = H.copy()
    T[T<3*th] = 0
    G = nx.from_numpy_matrix(T)
    partition = np.zeros(H.shape[0], dtype=np.int64)
    for idx, component in enumerate(nx.connected_components(G)):
        partition[list(component)] = idx
    return partition

if __name__ == '__main__':
    n = 6
    H = np.abs(np.random.randn(n, n))
    #H = (H + H.T) / 2
    #H = H - np.diag(np.diag(H))
    H = np.tril(H, k=-1)
    partition = min_k_cut(H, -1)
    print(partition)

#graphs = [nx.from_numpy_matrix(H)]
#cp = [nx.stoer_wagner(G) for G in graphs]






