import numpy as np
import matplotlib.pyplot as plt

import gym

from agents.base import FixedAgent
from agents.tdagent import TDAgent


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
                   simulation_freq_Hz = 500,
                   action_freq_Hz = 50,

                   episode_length_s = 20.0,

                   agent_force_min = 0.0, #N
                   agent_force_max = 100.0, #N

                   slider_mass = 3.0, #kg
                   slider_limits = np.array([-0.125, 0.125]), #m

                   reference_trajectory_fn = reference_fn,

                   #integration = "rk45",
                   )

    return env


def empirical_eval(dyad_slider_env, agents, number_of_episodes):
    total_reward = 0.0

    env_state = dyad_slider_env.reset()
    for episode in range(number_of_episodes):
        for step in range(dyad_slider_env.max_episode_steps):
             env_state, env_reward, done = dyad_slider_env.step([agents[0].get_force(env_state, record_history = False),
                                                                 agents[1].get_force(env_state, record_history = False)])
             total_reward += env_reward

             if done:
                  env.reset()
                  break

    return total_reward



if __name__ == "__main__":

    env = dyad_slider_prep()
    p1 = FixedAgent(perspective = 0,
                    force_max = env.agent_force_max/10,
                    force_min = env.agent_force_min,
                    c_effort = 0.0,
                    )
    p2 = TDAgent(perspective = 1,
                    force_max = env.agent_force_max,
                    force_min = env.agent_force_min,
                    c_effort = 0.0,
                    )

    episodes = 1000
    eval_resolution = 100
    eval_period = episodes / eval_resolution
    eval_episodes = 10

    eval_reward = np.zeros((eval_resolution,))

    for episode in range(episodes):
        print(episode)
        env_state = env.reset()

        episode_reward_total = 0.0

        for step in range(env.max_episode_steps):
             env_state, env_reward, done = env.step([p1.get_force(env_state),
                                                     p2.get_force(env_state)])
             episode_reward_total += env_reward

             #if episode == episodes - 1:
             env.render()

             p1.give_reward(env_reward, done, next_environment_state = env_state)
             p2.give_reward(env_reward, done, next_environment_state = env_state)

             if done:
                  env.reset()
                  break

        if episode % eval_period == 0:
            eval_reward[int(episode / eval_period)] = empirical_eval(env, [p1, p2], eval_episodes)

    env.close()
   
    plt.figure()
    plt.plot(np.arange(eval_resolution) * eval_period, eval_reward)
    plt.ylabel("Quality")
    plt.xlabel("Learning Episodes")

    plt.show()
