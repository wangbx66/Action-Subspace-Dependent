# action-dependence
Policy gradient research on action-dependent environment

##  pytorch-rl

### OpenAI Gym

```
pipi35 gym
piopi35 gym[atari]
```

Install Mujoco by getting license, downloading v1.50 binaries [here](https://www.roboti.us/download/mjpro150_linux.zip), unzipping and moving to ~/.mujoco/mjpro150, and move the key file to ~/.mujoco/mjkey.txt

```
pipi35 'mujoco-py<1.50.2,>=1.50.1'
pipi35 http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
pipi35 torchvision
```

Test
```
python35 examples/ppo_gym.py --env-name CartPole-v0
```

Mujoco, remove any esisting mujoco first
```
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pipi35 -e .
sudo pacman -S glfw-x11
git clone https://github.com/lobachevzky/gym
cd gym
pipi35 -e .[all]
```

For speed
```
export OMP_NUM_THREADS=1
```

## Evolutionary Clustering

```
K, the number of clusters or action subspaces
alpha, between 0 and 1, the rate of decay of cumulated Hessian matrixs
eta, the importance ratio of the k-means loss of the last frame
```
