import gym
import numpy as np
import sys
from IPython.display import clear_output
from time import sleep

env = gym.make('Pendulum-v0')
env.reset()  # reset environment to a new, random state
env.render()


print(f"Action Space {env.action_space}")
print(f"State Space {env.observation_space}")

env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = []  # for animation

done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    env.render()
    epochs += 1

env.env.close()
print(f"Timesteps taken: {epochs}")
print(f"Penalties incurred: {penalties}")
