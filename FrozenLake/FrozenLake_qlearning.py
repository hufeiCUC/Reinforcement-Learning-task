#用Q-Learning完成FrozenLake-v0
import gym
import collections
from tensorboardX import SummaryWriter

my_env = "FrozenLake-v0"
gamma = 0.9
alpha = 0.2
test_num = 20
iter_no = 0
best_reward = 0.0

env = gym.make(my_env)
state = env.reset()
test_env = gym.make(my_env)
values = collections.defaultdict(float)
writer = SummaryWriter(comment="-FrozenLake-q-learning")

def greedy_value_action(env,state):
    greedy_value, greedy_action = None, None
    for action in range(env.action_space.n):
        action_value = values[(state, action)]
        if greedy_value is None or greedy_value < action_value:
            greedy_value = action_value
            greedy_action = action
    return greedy_value, greedy_action

def test_process(env,show = False):
    total_reward = 0.0
    state = env.reset()
    while True:
        if show:
            env.render()
        _, action = greedy_value_action(env,state)
        new_state, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done:
            break
        state = new_state
    return total_reward

while True:
    iter_no += 1
    action = env.action_space.sample()
    old_state = state
    new_state, reward, is_done, _ = env.step(action)
    state = env.reset() if is_done else new_state
    greedy_v, _ = greedy_value_action(env,new_state)
    new_v = reward + gamma * greedy_v
    old_v = values[(old_state, action)]
    values[(old_state, action)] = old_v * (1-alpha) + new_v * alpha
    reward = 0.0
    for i in range(test_num):	
        reward += test_process(test_env)
    reward /= test_num
    writer.add_scalar("reward", reward, iter_no)
    if reward > best_reward:
        print("Best reward updated from %.3f to %.3f" % (best_reward, reward))
        best_reward = reward
    if reward > 0.90:
        print("Solved in %d iterations!" % iter_no)
        break
writer.close()
test_reward = test_process(test_env,show = True)
print('test reward = %.1f' % (test_reward))
