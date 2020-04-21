import numpy as np
import matplotlib.pyplot as plt

import gym

from agent import TDAgent, FixedAgent


def dyad_slider_prep(seed = 1234):

    random_offset = 0.0 #np.random.random() * 1000.0

    def reference_fn(t):
        offset_t = t + random_offset

        return 0.01 * (   (2 * np.sin(0.1 * offset_t))
                        + (5 * np.sin(0.3 * offset_t))
                        + (1 * np.sin(0.5 * offset_t))
                        + (3 * np.sin(0.8 * offset_t))
                       )

    env = gym.make('gym_dyad_slider:DyadSlider-v0',
                   reference_trajectory_fn = reference_fn,

                   agent_force_min = -100.0, #N
                   agent_force_max = 100.0, #N

                   slider_mass = 3.0, #kg
                   slider_limits = np.array([-0.125, 0.125]), #m

                   episode_length_s = 20.0,
                   )

    return env




if __name__ == "__main__":

    env = dyad_slider_prep()
    p1 = FixedAgent(perspective = 0,
                    force_max = env.agent_force_max,
                    force_min = env.agent_force_min,
                    c_effort = 0.0,
                    )
    p2 = TDAgent(perspective = 1,
                    force_max = env.agent_force_max,
                    force_min = env.agent_force_min,
                    c_effort = 0.0,
                    )

    episodes = 1000
    max_steps = 2000

    rewards = np.zeros((episodes,))

    for episode in range(episodes):
        print(episode)
        env_state = env.reset()

        episode_reward_total = 0.0

        for step in range(max_steps):
             env_state, env_reward, done = env.step([p1.get_force(env_state),
                                                     p2.get_force(env_state)])
             episode_reward_total += env_reward

             if episode == episodes - 1:
                 env.render()

             #p1.give_reward(env_state, env_reward, done)
             p2.give_reward(env_state, env_reward, done)

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
