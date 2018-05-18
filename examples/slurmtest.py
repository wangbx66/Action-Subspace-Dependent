import os
import glob
import shutil

def wait_mutex():
    import time
    while True:
        try:
            sign = open(os.path.expanduser('~/.mujoco/mutex')).readline()
            if not sign == 'available':
                print('Mutex locked')
                time.sleep(1)
            else:
                return
        except FileNotFoundError:
            print('Mutex not found')
            time.sleep(1)
    
def hold_mutex():
    with open(os.path.expanduser('~/.mujoco/mutex'), 'w') as fw:
        fw.write('occupied')


command_prefix =\
"""#!/bin/bash
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
hostname
whoami\n"""

sweep_directory = 'testdir'
repeats = 20
gamma_vals = [0.99]
lmbda_vals = [0.8]
mu_vals = [0.0]
alpha_vals = [0.015625]
updates = ['mixed']
games = ['freeway']
configs = [[gamma,lmbda,mu,alpha,update,game] for gamma in gamma_vals for lmbda in lmbda_vals for mu in mu_vals for alpha in alpha_vals for update in updates for game in games]

index = 0
for _ in range(repeats):
    for config in configs:
        file_name = 'job_'+str(index)+'.sh'
        command = 'python skytest.py -i {}'.format(index)
        file_string = command_prefix+command+'\n'
        with open(file_name, 'w') as f:
            f.write(file_string)
        index += 1
        wait_mutex()
        hold_mutex()
        os.system("sbatch {}".format(file_name))
        os.remove(file_name)
