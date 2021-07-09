# OpenAI_Gym

Both bots have been developed using reinforcement learning.

"Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Pinball."

Link : https://gym.openai.com/

CartPole : https://gym.openai.com/envs/CartPole-v1/

Taxi : https://gym.openai.com/envs/Taxi-v3/

## Installation
This application uses the **OpenAI gym** API.
```
pip install gym
```
Check out the documentation for the API [here](https://gym.openai.com/docs/).
External Python libraries **tflearn** and **numpy** were also used.
```
pip install tflearn
pip install numpy
```
## Cartpole
Link: https://gym.openai.com/envs/CartPole-v1/
- Run the **cartpole_v1.py** to train the agent of random trials using Convolutional Neural Networks (CNNs).
- Run the **test.py** to render 10 games played by the agent and output the average score out of 500 (index, meta, and data files are required to be in the same folder).

## Taxi
Link : https://gym.openai.com/envs/Taxi-v2/
- The program uses *Reinforcement Learning* to train the agent.
- At the end of training, one simulation is run.
