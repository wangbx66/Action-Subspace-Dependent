import os

env_list = ['HalfCheetah-v1']
seed_list = list(range(10))
k_list = [0]
delta_list = [1., 2., 3., 4., 5., ]

for env, seed, k, delta in zip(env_list, seed_list, k_list, delta_list):
    command = 'python rb_ppo_gym.py --env-name {env} --seed {seed} --learning-rate 3e-4 --max-iter-num 10000 --logger-name {env}-k{k}s{seed}d{delta} --number-subspace {k} --noise-mult {delta}'.format(env=env, seed=seed, k=k, delta=delta)
    os.system(command)

