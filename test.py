#!/usr/bin/env python3

import gym
env = gym.make('gym_dyad_slider:DyadSlider-v0')
env.reset()

for i in range(1000):
     observation, reward, done = env.step(env.action_space.sample())
     env.render()

     if done:
         env.reset()

env.close()
