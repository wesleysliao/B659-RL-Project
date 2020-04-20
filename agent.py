from collections import deque

import numpy as np
from scipy.special import expit as sigmoid



class DyadSliderAgent(object):

    def __init__(self,
                 force_max = 1.0,
                 force_min = -1.0,
                 c_error = 1.0,
                 c_effort = 1.0,
                 perspective = 0,
                 history_length = 2):

        self.perspective = perspective

        self.force = 0.0
        self.force_max = force_max
        self.force_min = force_min

        self.observation_history = deque(maxlen = history_length)
        self.action_history = deque(maxlen = history_length)

        self.c_error = c_error
        self.c_effort = c_effort


    def get_force(self, environment_state):
        observation = self.observe(environment_state)

        action = self.action_policy(observation)

        self.observation_history.append(observation)
        self.action_history.append(action)

        self.force = self.action_to_force(action)

        return self.force


    def give_reward(self, reward, is_terminal):
        subj_reward = self.subjective_reward(reward)

        if is_terminal:
            pass
        else:
            pass


    def observe(self, environment_state):
        x, x_dot, r, r_dot, \
        force_interaction, force_interaction_dot = environment_state

        if self.perspective % 2 == 0:
            error = x - r
            error_dot = x_dot - r_dot

        else:
            error = r - x
            error_dot = r_dot - x_dot

        observation = np.array([error, error_dot,
                                force_interaction, force_interaction_dot])

        return observation


    def action_policy(self, observation):
        return 0


    def action_to_force(self, action):
        force = action
        return force


    def effort(self, action):
        return np.sum(action)


    def subjective_reward(self, environment_reward):
        last_action = self.action_history[-1]

        reward = ((self.c_error * environment_reward)
                  + (self.c_effort * self.effort(last_action)))

        return reward



   
class RandomAgent(DyadSliderAgent):

    def __init__(self, action_space):
        self.action_space = action_space

    def action_policy(self, observation):
        return self.action_space.sample()



class TDAgent(DyadSliderAgent):

    def __init__(self,
                 Q_init = None,
                 discounting_coeff = 1.0,
                 learning_rate = 0.01,
                 epsilon = 0.01,
                 force_delta_max = 10.0,
                 seed_=None, **kwargs):

        self.nS = 10000
        self.nA = 3

        self.gamma = discounting_coeff
        self.alpha = learning_rate
        self.epsilon = epsilon
       

        if Q_init is None:
            self.Q_table = np.zeros((self.nS, self.nA))
        else:
            self.Q_table = Q_init

        self.force_delta_max = force_delta_max


        super().__init__(**kwargs)


    def give_reward(self, next_environment_state, reward, is_terminal):

        subj_reward = self.subjective_reward(reward)

        next_observation = self.observe(next_environment_state)

        s_old = self.observation_history[-1]
        s_new = next_observation

        a_old = self.action_history[-1]
        a_new = self.action_policy(s_new)

        print(s_old, s_new, a_old, a_new, subj_reward)

        if is_terminal:
            self.Q_table[s_old, a_old] += self.alpha * (subj_reward - self.Q_table[s_old, a_old])
        else:
            self.Q_table[s_old, a_old] += self.alpha * (subj_reward
                                                        + self.gamma * self.Q_table[s_new, a_new]
                                                        - self.Q_table[s_old, a_old])


    def observe(self, environment_state):
        error, error_dot, force_interaction, force_interaction_dot \
          = super().observe(environment_state)

        d = np.zeros(4 ,dtype=int)
        d[0] = np.floor(sigmoid(2.2 * error) * 9.99)
        d[1] = np.floor(sigmoid(error_dot) * 9.99)
        d[2] = np.floor(sigmoid(1 * force_interaction) * 9.99)
        d[3] = np.floor(sigmoid(0.3 * force_interaction_dot) * 9.99)
        return int(d[0] + 10 * d[1] + 100 * d[2] + 1000 * d[3])


    def epsilon_greedy(self, choices, epsilon):
        if np.random.rand() > epsilon:
            return np.argmax(choices)
        else:
            return np.random.randint(len(choices))


    def action_policy(self, observation):
        action = self.epsilon_greedy(self.Q_table[observation], self.epsilon)

        return action


    def action_to_force(self, action):

        print(self.force, self.force_delta_max, self.force_max, self.force_min)

        if action == 1:
            return min(self.force + self.force_delta_max, self.force_max)
        elif action == 2:
            return max(self.force - self.force_delta_max, self.force_min)
        else:
            return self.force

       
    def effort(self, action):
        force = self.action_to_force(action)
        effort = (-1) * abs(force / 20)

        return effort
