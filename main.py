import numpy as np
import matplotlib.pyplot as plt

import gym

from agent import TDAgent


if __name__ == "__main__":

    env = gym.make('gym_dyad_slider:DyadSlider-v0')
    p1 = TDAgent(perspective = 0)
    p2 = TDAgent(perspective = 1)

    env_state = env.reset()

    for i in range(1000):
        env_state, env_reward, done = env.step([p1.get_force(env_state),
                                                p2.get_force(env_state)])

        env.render()

        p1.give_reward(env_state, env_reward, done)
        p2.give_reward(env_state, env_reward, done)

        if done:
            env.reset()

        env.close()
