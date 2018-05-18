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
#SBATCH --cpus-per-task=1
hostname
whoami\n"""

env_list = ['HalfCheetah-v1']
seed_list = list(range(3))
k_list = [2]
#delta_list = [1., 2., 3., 4., 5., ]
method_list = ['posa']
config_list = [env_list, seed_list, k_list, method_list]
import itertools
configs = itertools.product(*config_list)

index = 0
for config in configs:
    file_name = 'job_'+str(index)+'.sh'
    command = 'python rb_ppo_gym.py --env-name {env} --seed {seed} --learning-rate {lr} --max-iter-num 10 --logger-name log/{env}-k{k}s{seed}-{method} --number-subspace {k}'.format(env=config[0], seed=config[1], k=config[2], method=config[3], lr='3e-3' if config[0].startswith('Quadratic') else '3e-4')
    file_string = command_prefix+command+'\n'
    with open(file_name, 'w') as f:
        f.write(file_string)
    index += 1
    wait_mutex()
    hold_mutex()
    os.system("sbatch {}".format(file_name))
    os.remove(file_name)
