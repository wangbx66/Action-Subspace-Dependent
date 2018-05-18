from cvxopt import matrix, solvers, spdiag, log
import numpy as np

solvers.options['maxiters'] = 200


n = 10
G = matrix(np.concatenate([-np.eye(n).astype(np.float), np.eye(n).astype(np.float)]))
h = matrix(np.concatenate([np.zeros(n).astype(np.float), np.ones(n).astype(np.float)]))
A = matrix(np.ones(n).astype(np.float), (1,n))
b = matrix(1.0)
z = np.array([1.0]*(n-1)+[100.0])
z = z/sum(z)
x0 = matrix(z, (n,1))

'''
def F(x=None, z=None):
    # F(x) = x.Tx
    if x is None: 
        return 0, matrix(np.zeros(n).astype(np.float), (n,1))
    elif min(x) <= 0.0:
        return None
    else:
        f = sum(x**2)
        Df = (x*2).T
        if z is None:
            return f, Df
        else:
            H = z[0]*2*matrix(np.eye(n).astype(np.float))
            return f, Df, H
'''

'''
scipy.optimize.minimize(F, x0, method='L-BFGS-B', jac=DF, hess=H, bounds=bounds)
'''

def acent(A, b, e):
    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, x0
        if min(x) <= 0.0: return None
        f = -sum(log(x)) + e*np.random.normal()
        
        Df = -(x**-1).T + e*matrix(np.random.normal(n), (1,n))
        if z is None: return f, Df
        H = matrix(np.eye(n).astype(np.float))
        return f, Df, H
    sol = solvers.cp(F, A=A, b=b)
    print(sol['x'])
    return sol

def lovasz(G, h, e):
    '''
    You will have to fix two points with partition 0 and 1 respectively.
    '''
    m, n = A.size
    def F(wi):
        z = int(n / 2)
        s1 = wi[:z].sum()
        s2 = wi[z:].sum()
        return s1*(z-s1) + s2*(z-s2)
    def FL(x=None, z=None):
        if x is None: return 0, 0.5*matrix(np.ones(n), (n,1))
        if min(x) < 0.0: return None
        if max(x) > 1.0: return None
        x = np.array(x).flatten()
        order = np.argsort(x)
        sort = np.insert(np.insert(np.sort(x),0,0),n+1,1)
        weight = sort[1:] - sort[:-1]
        fwi = np.zeros(n+1)
        wi = np.zeros(n)
        Df = np.zeros(n)
        f = 0
        for idx, arg in enumerate(order):
            fwi[idx] = F(wi)
            f += fwi[idx] * weight[idx]
            wi[arg] = 1
        fwi[n] = F(wi)
        f += fwi[n] * weight[n]
        clean = np.sort(x)
        for idx, ele in enumerate(x):
            left = np.searchsorted(clean,x[idx])
            right = np.searchsorted(clean,x[idx],side='right')
            base = 0
            grad = 0
            if left >= 1:
                leftgrad = fwi[left] - fwi[left-1]
                grad += leftgrad
                base += 1
            if right <= n:
                rightgrad = fwi[right] - fwi[right-1]
                grad += rightgrad
                base += 1
            Df[idx] = grad / base
        f += e*np.random.normal()
        Df += e*np.random.normal(n)
        Df = matrix(Df, (1,n))
        if z is None: return f, Df
        H = matrix(np.eye(n).astype(np.float))
        return f, Df, H
    sol = solvers.cp(FL, G=G, h=h)
    print(sol['x'])
    return sol


e = 1
#with open('log.txt', 'a') as fw:
#    fw.write('Noise level {}'.format(str(e)))
#    fw.write('\n')
#sol = acent(A,b,e)
sol = lovasz(G,h,e)

result = []
with open('log.txt') as fp:
    for line in fp:
        x = line.strip().split()
        f = np.log(np.array(x, dtype=np.float)).sum()
        result.append(f)

import os
os.remove('log.txt')

import matplotlib.pyplot as plt
plt.plot(result)
plt.title('Error level {}'.format(e))
plt.show()


