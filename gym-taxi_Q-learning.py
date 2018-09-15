# /usr/bin/python3

import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print("Timestep: {}".format(i + 1))
        print("State: {}".format(frame['state']))
        print("Action: {}".format(frame['action']))
        print("Reward: {}".format(frame['reward']))
        sleep(.1)



env = gym.make('Taxi-v2')
env.reset()  # reset environment to a new, random state
env.render()

print("Action Space ".format(env.action_space))
print("State Space ".format(env.observation_space))

q_table = np.zeros([env.observation_space.n, env.action_space.n])


'''Q learning Train agent '''


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: {}".format(i))

print("Training finished.\n")

'''finished agent training '''


'''eval agent  '''
frames = []  # for animation
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
# print_frames(frames)     uncomment to printout all frames to console
print("Results after {} episodes:" .format(episodes))
print("Average timesteps per episode: {}".format(total_epochs / episodes))
print("Average penalties per episode: {}".format(total_penalties / episodes))
'''end eval agent  '''
