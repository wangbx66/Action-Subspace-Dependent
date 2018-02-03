import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pylab
import os
import seaborn as sns

class color:
    def __init__(self):
        #self.cl = ['orange', 'orange', 'orchid', 'orchid', 'g', 'green', 'c', 'cyan', 'b', 'blue', 'm', 'magenta', 'k', 'black']
        self.cl = ['seagreen', 'seagreen', 'blue', 'blue', 'red','red',  'c', 'cyan',  'orchid', 'orchid', 'g', 'green', 'c', 'cyan', 'b', 'blue', 'm', 'magenta', 'k', 'black']
        self.idx = -1
    def draw(self):
        self.idx += 1
        return self.cl[self.idx]

cc = color()
sns.set_style("ticks") # style must be one of white, dark, whitegrid, darkgrid, ticks
color_list = sns.color_palette("muted")
plt.rcParams["figure.figsize"] = (8,5.3)

env = 'hopper'
if env == 'quadraticm4k2version':
    m = 20
if env == 'quadraticm40k4':
    m = 100
if env == 'quadraticm20k4':
    m = 20
if env == 'quadraticm10k2':
    m = 20
if env == 'quadraticm4k2':
    m = 20
if env == 'quadratic':
    m = 20
if env == 'ant':
    m = 50
if env == 'halfcheetah':
    m = 50
if env == 'walker':
    m = 100
if env == 'hopper':
    m = 50
if env == 'doublehopper':
    m = 100

def draw(name, k, s, v=1):
    t = []
    avgr = []
    stdr = []
    steps = 0
    with open('log{}-k{}s{}v{}'.format(name, k, s, v)) as fr:
        for idx, line in enumerate(fr):
            if steps > 4200000:
                break
            try: 
                timestamp, i_iter, steps, sample_time, update_time, min_reward, max_reward, std_reward, mean_reward = line.split('\t')
            except:
                timestamp, i_iter, steps, _, sample_time, update_time, min_reward, max_reward, std_reward, mean_reward = line.split('\t')
            steps = int(steps)
            if steps < 0:
                continue
            t.append(steps)
            avgr.append(float(std_reward.split(' ')[1]))
            stdr.append(float(mean_reward.split(' ')[1]))
    data = np.array([stdr, avgr]).T
    ss = pd.DataFrame(index=t, data=data)
    ssm = ss.rolling(m).mean()
    ssm[2] = (ssm[0].rolling(m).var() + (ssm[1]**2).rolling(m).mean()) ** 0.5
    
    if k == 3:
        label = 'ADFB'
    elif k == 1:
        label = 'GADB'
    else:
        label = 'ASDG_{}'.format(k)
        
    plt.plot(ssm.index,ssm[0],c=cc.draw(),linewidth=4.0,label=label)
    #plt.plot(ssm.index,ssm[0],c=cc.draw(),linewidth=4.0,label='K{}S{}V{}'.format(k, s, v))
    #cc.draw()
    plt.fill_between(ssm.index, ssm[0]-ssm[1], ssm[0]+ssm[1], alpha=0.2, facecolor=cc.draw())

#plt.xlabel('#Steps', fontsize=18)
#plt.ylabel('Score', fontsize=16)
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: ('%s') % (str(int(x * 1e-3)) + 'k')))

#draw(env, 4, 1, 1)
#draw(env, 40, 1, 1)
#draw(env, 1, 1, 1)s
#draw(env, 1, 1, 1)
#draw(env, 3, 1, 1)
#draw(env, 4, 1, 1)


#draw(env, 10, 1, 1)
#draw(env, 1, 1, 1)
#draw(env, 2, 1, 1)
#draw(env, 2, 1, 1)
draw(env, 3, 1, 1)
#draw(env, 5, 1, 1)
draw(env, 1, 1, 1)
draw(env, 2, 1, 1)
#draw(env, 2, 1, 1)
#draw(env, 3, 1, 1)
#draw(env, 2, 1, 1)
plt.legend(loc='upper left', fontsize=14)
plt.title('Hopper-V1'.format(env), fontsize=20)
plt.savefig('Hopper-V1.pdf')
#plt.title('HalfCheetah-V1'.format(env), fontsize=20)
#plt.savefig('HalfCheetah-V1.pdf')
plt.show()
