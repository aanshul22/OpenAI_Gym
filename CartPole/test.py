import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
num_games = 10

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss = 'categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


model = neural_network_model(4)
model.load('cartpole-v1')

def test(model, num_games):
    scores = []
    choices = []
    for each_game in range(num_games):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append(new_observation)
            score += reward
            if done:
                break

        scores.append(score)
    return (scores, choices)


t = test(model, num_games)
print('Average Score', mean(t[0]))
print('Choice 1: {}, Choice 0: {}'.format(t[1].count(1)/len(t[1]), t[1].count(0)/len(t[1])))
