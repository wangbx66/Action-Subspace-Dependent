import matplotlib.pyplot as plt
import numpy as np

printmin = False
printmax = False
printavg = True
minr = []
maxr = []
avgr = []
capidx = 6000
with open('figure') as fr:
    for idx, line in enumerate(fr):
        if idx > capidx:
            break
        ll = line.split('\t')
        if printmin:
            minr.append(float(ll[-3].split(' ')[1]))
        if printmax:
            maxr.append(float(ll[-2].split(' ')[1]))
        if printavg:
            avgr.append(float(ll[-1].split(' ')[1]))
if printmin:
    plt.plot(minr)
if printmax:
    plt.plot(maxr)
if printavg:
    plt.plot(avgr)
plt.show()
