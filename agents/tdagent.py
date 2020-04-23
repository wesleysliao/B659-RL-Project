
import numpy as np
from scipy.special import expit as sigmoid

from agents.base import DyadSliderAgent

class TDAgent(DyadSliderAgent):

    def __init__(self,
                 Q_init = None,
                 discounting_coeff = 1.0,
                 learning_rate = 0.01,
                 epsilon_start = 0.4,
                 epsilon_decay = 0.01,
                 epsilon_final = 0.01,
                 force_delta_max = 10.0,
                 seed_=None, **kwargs):

        self.nS = 10000
        self.nA = 3

        self.gamma = discounting_coeff
        self.alpha = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final

        if Q_init is None:
            self.Q_table = np.zeros((self.nS, self.nA))
        elif np.isscalar(Q_init):
            self.Q_table = Q_init * np.ones((self.nS, self.nA))
        else:
            self.Q_table = Q_init

        self.force_delta_max = force_delta_max


        self.reset_count = 0
        super().__init__(**kwargs)


    def give_reward(self, reward, is_terminal, next_environment_state = None):

        subj_reward = self.subjective_reward(reward)

        next_observation = self.observe(next_environment_state)

        s_old = self.observation_history[-1]
        s_new = next_observation

        a_old = self.action_history[-1]
        a_new = self.action_policy(s_new)

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
        if action == 1:
            self.force = min(self.force + self.force_delta_max, self.force_max)
        elif action == 2:
            self.force = max(self.force - self.force_delta_max, self.force_min)

        return self.force


    def effort(self, action):
        force = self.action_to_force(action)
        effort = (-1) * abs(force / 20)

        return effort


    def reset(self):
        self.force = 0.0

        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_final:
            self.epsilon = self.epsilon_final
