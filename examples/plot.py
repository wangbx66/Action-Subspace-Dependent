import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import os
import seaborn as sns

class color:
    def __init__(self):
        self.cl = ['g', 'green', 'r', 'red', 'b', 'blue', 'c', 'cyan', 'm', 'magenta', 'y', 'yellow', 'k', 'black', 'w', 'white']
        self.idx = -1
    def draw(self):    
        self.idx += 1
        return self.cl[self.idx]

cc = color()
sns.set_style("ticks") # style must be one of white, dark, whitegrid, darkgrid, ticks
color_list = sns.color_palette("muted")
plt.rcParams["figure.figsize"] = (8,5.3)
capidx = 1e7
m = 100

def draw(name, k, s, v=1):
    t = []
    avgr = []
    stdr = []
    steps = 0
    with open('log{}-k{}s{}'.format(name, k, s) if v==1 else 'log{}-k{}s{}v{}'.format(name, k, s, v)) as fr:
        for idx, line in enumerate(fr):
            if steps > capidx:
                break
            try: 
                timestamp, i_iter, steps, sample_time, update_time, min_reward, max_reward, std_reward, mean_reward = line.split('\t')
            except:
                timestamp, i_iter, steps, _, sample_time, update_time, min_reward, max_reward, std_reward, mean_reward = line.split('\t')
            steps = int(steps)
            t.append(steps)
            avgr.append(float(std_reward.split(' ')[1]))
            stdr.append(float(mean_reward.split(' ')[1]))
    data = np.array([stdr, avgr]).T
    ss = pd.DataFrame(index=t, data=data)
    ssm = ss.rolling(m).mean()
    ssm[2] = (ssm[0].rolling(m).var() + (ssm[1]**2).rolling(m).mean()) ** 0.5

    plt.plot(ssm.index,ssm[0],c=cc.draw(),linewidth=3.0,label='K{}S{}V{}'.format(k, s, v))
    plt.fill_between(ssm.index, ssm[0]-ssm[2], ssm[0]+ssm[2], alpha=0.2, facecolor=cc.draw())

draw('walker', 1, 1, 1)
draw('walker', 6, 1, 2)
draw('walker', 6, 2, 2)
draw('walker', 6, 1, 1)
plt.legend(loc='upper left')
plt.title('Walker-V1')
plt.show()
