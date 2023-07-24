# MarketMakingViaReinforcementLearning
import gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make('CartPole-v1')

upper_bounds = env.observation_space.high[2:4]
lower_bounds = env.observation_space.low[2:4]
no_actions = env.action_space.n
n_bins = (30, 30)
est = KBinsDiscretizer(n_bins = n_bins, encode='ordinal', strategy='uniform')
est.fit([lower_bounds, upper_bounds])


#learning rate
lr  = 0.1
#discount factor
df = 1
#epsilon-greedy
epsilon = 0.2
n_episodes = 15000


Q_table = np.random.uniform(low=0, high=1, size= n_bins + (no_actions,))

def convert_state(state):
    if isinstance(state, tuple):
        return list(state[0])[2:4]
    elif len(state) == 4:
        return list(state)[2:4]
    else:
        return state
def get_index_by_state(state):
    return tuple(est.transform([state])[0].astype(int))

def get_index_by_state_action(state, action):
    return tuple(est.transform([state])[0].astype(int)) + (action,)

def test_policy(state, index):
    state_index = get_index_by_state(state)
    return np.random.choice(np.where(Q_table[state_index]==np.max(Q_table[state_index]))[0])
def train_policy(state, index):
    global epsilon
    if index<500:
        print("random action")
        return np.random.choice(no_actions)

    if index>7000:
        epsilon=0.999* epsilon
    if np.random.random() < epsilon:
        print("random action")
        return np.random.choice(no_actions)
    else:
        state_index = get_index_by_state(state)
        return np.random.choice(np.where(Q_table[state_index]==np.max(Q_table[state_index]))[0])
def magic_policy(state):
  theta,omega=list(state)[2:4]
  if abs(theta) < 0.03:
    return 0 if omega < 0 else 1
  else:
    return 0 if theta < 0 else 1

rewards = []
for index in range(n_episodes):
  rewards_episode = []
  current_state = env.reset()
  current_state = convert_state(current_state)

  terminated = False
  while not terminated:
      # action = test_policy(current_state, index)
      action = train_policy(current_state, index)
      # action = magic_policy(current_state)
      (new_state, reward, terminated, _, _) = env.step(action)
      new_state = convert_state(new_state)
      # print("state: ", current_state, " action: ", "left" if action == 0 else "right", " reward: ", reward)
      rewards_episode.append(reward)
      max_Q = np.max(Q_table[get_index_by_state(new_state)])
      q_index = get_index_by_state_action(current_state, action)
      if not terminated:
          error = reward + df * max_Q - Q_table[q_index]
          Q_table[q_index] = Q_table[q_index] + lr * error
      else:
          error = reward - Q_table[q_index]
          Q_table[q_index] = Q_table[q_index] + lr * error
      current_state = new_state
  total_reward = np.sum(rewards_episode)
  print("loop: ", index, " total reward: ", total_reward, " state: ", current_state)
  rewards.append(total_reward)


plt.plot(rewards)
plt.show()
