import gym
import numpy as np

env = gym.make("Taxi-v3")
env.reset()

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
V = np.zeros([NUM_STATES])
pi = np.zeros([NUM_STATES], dtype=int)

gamma = 0.9 # Discount factor
significant_improvement = 0.01

def best_action_value(s):
    best_action = None
    best_value = float('-inf')

    for action in range(NUM_ACTIONS):
        env.env.s = s

        new_s, reward, done, info = env.step(action)

        v = reward + gamma * V[new_s]
        if v > best_value:
            best_action = action
            best_value = v
    return best_action


iteration = 0

while True:
    biggest_change = 0

    for s in range(NUM_STATES):
        old_v = V[s]
        action = best_action_value(s)
        env.env.s = s

        new_s, reward, done, info = env.step(action)

        V[s] = reward + gamma * V[new_s]
        pi[s] = action

        biggest_change = max(biggest_change, abs(old_v - V[s]))

    iteration += 1

    if biggest_change < significant_improvement:
        print(f"{iteration} iterations done before perfecting!\n")
        break


total_reward = 0
observation = env.reset()
env.render()
done = False
while not done:
    action = pi[observation]
    observation, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print(f"Reward: {total_reward}")
