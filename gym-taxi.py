# /usr/bin/python3

import gym
import numpy as np
import sys
from IPython.display import clear_output
from time import sleep

env = gym.make('Taxi-v2')
env.reset()  # reset environment to a new, random state
env.render()


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


print("Action Space {env.action_space}")
print("State Space {env.observation_space}")

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

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

# print_frames(frames)     uncomment to printout all frames to console

print(f"Timesteps taken: {epochs}")
print(f"Penalties incurred: {penalties}")
