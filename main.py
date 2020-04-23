import time
import datetime

import numpy as np
import matplotlib.pyplot as plt

import gym

from agents.base import PIDAgent
from agents.tdagent import TDAgent


def reference_fn_generator():
    random_offset = np.random.random() * 2 * np.pi

    def reference_fn(t):
        offset_t = t + random_offset

        return 0.1 * (  (0.2 * np.sin(0.4 * offset_t))
                      + (0.4 * np.sin(0.8 * offset_t))
                      + (0.1 * np.sin(1.6 * offset_t))
                      + (0.3 * np.sin(3.2 * offset_t))
                     )

    return reference_fn


def dyad_slider_prep(seed = 1234):

    env = gym.make('gym_dyad_slider:DyadSlider-v0',
                   simulation_freq_Hz = 500,
                   action_freq_Hz = 50,

                   episode_length_s = 20.0,

                   agent_force_min = 0.0, #N
                   agent_force_max = 100.0, #N

                   slider_mass = 3.0, #kg
                   slider_limits = np.array([-0.125, 0.125]), #m

                   reference_generator = reference_fn_generator,

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
                 agents[0].reset()
                 agents[1].reset()
                 env.reset()
                 break

    return total_reward



if __name__ == "__main__":

    env = dyad_slider_prep()
    p1 = PIDAgent(-30, 0.0, -30,
                  perspective = 0,
                  force_max = env.agent_force_max,
                  force_min = env.agent_force_min,
                  c_effort = 0.0,
                  )


    p2 =  TDAgent(
                  perspective = 1,
                  force_max = env.agent_force_max,
                  force_min = env.agent_force_min,
                  c_effort = 0.0,
                  Q_init = 300.0,
                  force_delta_max = 0.1,
                  epsilon_start = 0.5,
                  epsilon_decay = 0.0001,
                  epsilon_final = 0.01,
                  )

    number_of_episodes = 5000
    eval_resolution = 200
    eval_period = number_of_episodes / eval_resolution
    eval_episodes = 10

    eval_reward = np.zeros((eval_resolution,))

    period_time = time.perf_counter()
    time_per_episode_s = 0.0

    for episode_ndx in range(number_of_episodes):
        remaining_time_s = time_per_episode_s * (number_of_episodes - episode_ndx)
        print("{:2.2%} estimated time remaining:".format(episode_ndx/number_of_episodes),
              time.strftime("%H:%M:%S", time.gmtime(remaining_time_s)), end="\r")
        env_state = env.reset()

        episode_reward_total = 0.0

        if episode_ndx == (number_of_episodes / 4):
            p1 = TDAgent(
                         perspective = 0,
                         force_max = env.agent_force_max,
                         force_min = env.agent_force_min,
                         c_effort = 0.0,
                         Q_init = 300.0,
                         force_delta_max = 0.1,
                         epsilon_start = 0.5,
                         epsilon_decay = 0.0001,
                         epsilon_final = 0.01,
                        )

        for step in range(env.max_episode_steps):
             env_state, env_reward, done = env.step([p1.get_force(env_state),
                                                     p2.get_force(env_state)])
             episode_reward_total += env_reward

             if episode_ndx % eval_period == 0:
                 env.render()

             p1.give_reward(env_reward, done, next_environment_state = env_state)
             if episode_ndx < (number_of_episodes / 4) or episode_ndx > ((3 * number_of_episodes) / 4):
                 p2.give_reward(env_reward, done, next_environment_state = env_state)

             if done:
                 p1.reset()
                 p2.reset()
                 env.reset()
                 break

        if episode_ndx % eval_period == 0:
            eval_reward[int(episode_ndx / eval_period)] = empirical_eval(env, [p1, p2], eval_episodes)
            time_per_episode_s = (time.perf_counter() - period_time) / (eval_period + eval_episodes)
            period_time = time.perf_counter()

    print('')
    print("complete")


    env.close()
   
    plt.figure()
    plt.plot(np.arange(eval_resolution) * eval_period, eval_reward)
    plt.ylabel("Score")
    plt.xlabel("Learning Episodes")

    plt.show()
