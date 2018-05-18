import os

env_list = ['HalfCheetah-v1']
seed_list = list(range(5))
k_list = [0]
delta_list = [1., 2., 3., 4., 5., ]
config_list = [env_list, seed_list, k_list, delta_list]

import itertools
for config in itertools.product(*config_list):
    command = 'python rb_ppo_gym.py --env-name {env} --seed {seed} --learning-rate 3e-4 --max-iter-num 10000 --logger-name {env}-k{k}s{seed}d{delta} --number-subspace {k} --noise-mult {delta}'.format(env=config[0], seed=config[1], k=config[2], delta=config[3])
    #os.system(command)
    print(command)

