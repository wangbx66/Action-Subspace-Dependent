import os

def hold_mutex():
    with open(os.path.expanduser('~/.mujoco/mutex'), 'w') as fw:
        fw.write('occupied')

def release_mutex():
    with open(os.path.expanduser('~/.mujoco/mutex'), 'w') as fw:
        fw.write('available')

hold_mutex()

import socket
host = socket.gethostname()
assert(host[:7] == 'compute') and (len(host) == 10)
computer = host[-3:]
mjkey = os.path.expanduser('~/.mujoco/mjkey{}.txt'.format(computer))

from shutil import copyfile
copyfile(mjkey, os.path.expanduser('~/.mujoco/mjkey.txt'))

import gym
env = gym.make('Humanoid-v1')
env.reset()

release_mutex()

import time
time.sleep(30)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=0)
args = parser.parse_args()
print(args.index)
