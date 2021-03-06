# Action-Subspace-Dependent

```
ssh tangruiming@115.28.175.202 -p 2129
df65w30G
ssh wwliu@gpu16.cse.cuhk.edu.hk
x2xl2voj
ssh shuaili@hpc7.cse.cuhk.edu.hk
t147258-
dl@172.22.22.212
admin
```

##  pytorch-rl

```
python35 rb_ppo_gym.py --env-name Walker2d-v1 --seed 1 --learning-rate 3e-4 --max-iter-num 10000 --logger-name walker-k1s1v1 --number-subspace 1
```

```
source ~/.bashrc
setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:/uac/gds/wwliu/.mujoco/mjpro150/bin
setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:/uac/gds/wwliu/usr/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/uac/gds/shuaili/.mujoco/mjpro150/bin:/research/syzhang/shuaili/usr/lib64
setenv CPATH /uac/gds/wwliu/usr/include
export CPATH=/research/syzhang/shuaili/usr/include
export LD_LIBRARY_PATH=/home/brandon/.mujoco/mjpro150/bin:/usr/lib
wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/g/glfw-devel-3.2.1-2.el7.x86_64.rpm
rpm2cpio glfw-devel-3.2.1-2.el7.x86_64.rpm | cpio -idv
wget https://rpmfind.net/linux/centos/7.4.1708/os/x86_64/Packages/mesa-libOSMesa-17.0.1-6.20170307.el7.x86_64.rpm
rpm2cpio libOSMesa-devel-17.0.5-176.1.x86_64.rpm | cpio -idv
setenv CUDA_VISIBLE_DEVICES 0
setenv OMP_NUM_THREADS 1
```

### Mujoco

```
wget https://www.roboti.us/getid/getid_linux
chmod 777 getid_linux
./getid_linux
wget https://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip
mkdir .mujoco
mv mjpro150 .mujoco
vim .mujoco/mjkey.txt (and paste the key)
(aws) sudo apt-get install libosmesa6-dev
(centos) wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
(centos) bash Anaconda3-5.0.1-Linux-x86_64.sh
(centos) conda install python=3.5
(centos) sudo yum install mesa-libOSMesa-devel.x86_64
(centos) sudo yum install glfw-devel.x86_64
(ubuntu) sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
(ubuntu) sudo chmod +x /usr/local/bin/patchelf

git clone https://github.com/openai/mujoco-py
cd mujoco-py
LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pipi35 -e .
(arch) sudo pacman -S glfw-x11
(aws) sudo apt-get install swig3.0
git clone https://github.com/lobachevzky/gym
cd gym
pipi35 -e .[all]
pipi35 http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
pipi35 torchvision
```

Test
```
python35 examples/ppo_gym.py --env-name CartPole-v0
```

For speed
```
export OMP_NUM_THREADS=1
```

## List of Envs Used (Rvs Chrnon)

| |Hopper-V1|InvertDoublePendulum-V1|InvertPendulum-V1|Reacher-V1|Swimmer-V1|Walker2d-V1|HalfCheetah-V1|BlindPegInsertion-V1|CommunicateTarget-V1|Ant-V1|DoorOpening-V1|Cartpool-V1|Humannoid-V1|HumanoidStandup-V1|
|-|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|Sample Efficient|x| | | | |x|x| | |x| | |x|x|
|Cathy Wu        |x| | | | | |x|x|x|x|x| | | |
|PPO             |x|x|x|x|x|x|x| | | | | | | |
|TRPO            |x| | | |x|x| | | | | |x| | |
|Q-Prop          |x| | |x|x|x|x| | |x| | |x| |
|GAE             | | | | | |x| | | | | |x| |x|
|SVPG            | |x| | | | | | | | | |x| | |
|IPG             | | | | | |x|x| | |x| | |x| |


## Evolutionary Clustering

```
K 
alpha, between 0 and 1, the rate of decay of cumulated Hessian matrixs
beta, between 0 and 1, the importance ratio (compared with overall) of temporal similarity
eta, the importance ratio of the k-means loss of the last frame
```
