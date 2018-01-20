import argparse
import gym
import os
import sys
import pickle
import time
import datetime
sys.path.append('/home/bxwang/action-dependence/pytorch-rl')

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_advantage import Advantage
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=8, metavar='N',
                    help='number of threads for agent (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=100000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()
#args.env_name = 'Humanoid-v1'
args.env_name = 'DoubleHopper-v1'
args.render = False

def env_factory(thread_id):
    if args.env_name == 'Quadratic':
        env = Quadratic()
    elif args.env_name.startswith('Double'):
        env = Double(args.env_name[6:])
    elif args.env_name.startswith('Trible'):
        env = Double(args.env_name[6:], replicate=3)
    elif args.env_name.startswith('Quadruple'):
        env = Double(args.env_name[9:], replicate=4)
    else:
        env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)
state_dim = env_dummy.observation_space.shape[0]
action_dim = env_dummy.action_space.shape[0]
is_disc_action = len(env_dummy.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
size = (64, 64)
policy_size = size#(8, 8)
critic_size = size#(8, 8)
advantage_size = size#(8, 8)
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
    else:
        policy_net = Policy(state_dim, env_dummy.action_space.shape[0], hidden_size=policy_size, log_std=args.log_std)
    value_net = Value(state_dim, hidden_size=critic_size)
    advantage_net = Advantage((state_dim, action_dim), hidden_size=advantage_size)
else:
    policy_net, value_net, advantage_net, running_state = pickle.load(open(args.model_path, "rb"))
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    advantage_net = advantage_net.cuda()
del env_dummy

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_advantage = torch.optim.Adam(advantage_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 4096

"""create agent"""
agent = Agent(env_factory, policy_net, running_state=running_state, render=args.render, num_threads=args.num_threads)

class eclustering_dummy:
    def step(self, H):
        #return np.array([0,0,0] * 2, dtype=np.int64)
        return np.array([0,0,0] + [1,1,1], dtype=np.int64)

def update_params(batch, i_iter, partition, ecluster):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    #advantage_inputs = torch.cat((states, actions), dim=1)
    #advantages_symbol = advantage_net(Variable(states, volatile=True))
    #advantage = advantages_symbol.data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions), partition).data

    """get advantage estimation from the trajectories"""
    advantages, returns, advantages_unbiased = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    list_H = []

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            H = ppo_step(policy_net, value_net, advantage_net, optimizer_policy, optimizer_value, optimizer_advantage, 1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg, partition)
            list_H.append(H.unsqueeze(0))
    H_hat = torch.cat(list_H, dim=0).mean(dim=0).numpy().astype(np.float64)
    with open('loghdoublehopper01', 'a') as fa:
        fa.write(str(H_hat) + '\n')
    partition = ecluster.step(H_hat)
    return partition

partition = np.array([0,] * action_dim, dtype=np.int64)
for i_iter in range(args.max_iter_num):
    """
    generate multiple trajectories that reach the minimum batch_size
    batch: size optim_epochs tuple, then size 
    """    
    batch, log = agent.collect_samples(args.min_batch_size)
    t0 = time.time()
    partition = update_params(batch, i_iter, partition, eclustering_dummy())
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        msg = '{}\t{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}\n'.format(
            datetime.datetime.now().strftime('%d %H:%M:%S'), i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward'])
        print(msg)
        with open('logdoublehopper01', 'a') as fa:
            fa.write(msg)

    if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
        if use_gpu:
            policy_net.cpu(), value_net.cpu()
        pickle.dump((policy_net, value_net, running_state),
                    open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
        if use_gpu:
            policy_net.cuda(), value_net.cuda()

