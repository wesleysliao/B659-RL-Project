import numpy as np
import matplotlib.pyplot as plt

import gym

from agent import TDAgent, FixedAgent


if __name__ == "__main__":

    env = gym.make('gym_dyad_slider:DyadSlider-v0')
    p1 = TDAgent(perspective = 0)
    p2 = FixedAgent(perspective = 1)

    episodes = 4000
    max_steps = 300

    rewards = np.zeros((episodes,))

    for episode in range(episodes):
        env_state = env.reset()

        episode_reward_total = 0.0

        for step in range(max_steps):
             env_state, env_reward, done = env.step([p1.get_force(env_state),
                                                     p2.get_force(env_state)])
             episode_reward_total += env_reward

             #env.render()

             p1.give_reward(env_state, env_reward, done)
             p2.give_reward(env_reward, done)

             if done:
                  env.reset()
                  break

        rewards[episode] = episode_reward_total

    env.close()
   
    plt.figure()
    plt.plot(np.arange(episodes), rewards)
    plt.ylabel("Total Reward")
    plt.xlabel("Learning Episodes")

    plt.show()
