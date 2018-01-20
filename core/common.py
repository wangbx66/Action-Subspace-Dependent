

def estimate_advantages(rewards, masks, values, gamma, tau, use_gpu):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()
    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)
    advantages_unbiased = tensor_type(rewards.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    prev_advantage_unbiased = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        advantages_unbiased[i] = deltas[i] + gamma * prev_advantage_unbiased * masks[i]

        prev_return = returns[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
        prev_advantage_unbiased = advantages_unbiased[i, 0]
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_unbiased = (advantages_unbiased - advantages_unbiased.mean()) / advantages_unbiased.std()

    if use_gpu:
        advantages, returns, advantages_unbiased = advantages.cuda(), returns.cuda(), advantages_unbiased.cuda()
    return advantages, returns, advantages_unbiased
