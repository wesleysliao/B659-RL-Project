import gym
import sys, time
import numpy as np; 
from numpy import asarray as narr
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
import torch
from torch import nn, optim


# Import custom scripts
sys.path.append('configs/')
sys.path.append('agents/')
sys.path.append('util/')
import env_old
from util import buffers, nn_models, torch_trainer
from agents import rl_agent, train_agents, benchmark_agents


BMHyperparams = namedtuple('BMHyperparams',
                ('batch_size','learning_rate', 'buffer_max_size',
                 'experience_sift_tol', 'target_int'))
#Benchmark x-axis related
n_episodes = 150 #recommended: 150
n_intervals = 15 # recommended: 15
n_eval = 10

# Buffer-related
experience_sift_tol = 0.01
buffer_max_size = 500

# Algo hyperparams
learning_rate=0.005
batch_size=128
xaxis_params = (n_episodes, n_intervals, n_eval)


#------- Create the agents
buffer1 = buffers.ReplayMemory(buffer_max_size, tag=experience_sift_tol)
buffer2 = deepcopy(buffer1)

    #------- Create the NN models
# Input:
# r, r', x, x', fn, fndot, f
# Output: q
qnet = nn_models.NetRelu1L1(7, 1)
optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
rl1 = torch_trainer.DQNTrainer(qnet, optimizer, criterion)
rl2 = deepcopy(rl1)

pdcont1 = [0.1, .2]#[0., 0]#[0.3, 1.]#[0.0364, 1.]#[0.01, 0.03]
pdcont2 = [0.2, 0.1]#[0., 0]#[0.3, 1.]#[0.0364, 1.]#[0.01, 0.03]
sigma = 0. #0.3

seed = 1234
env = env_old.PhysicalDyads(seed_=seed, max_freq=0.2)
#___________________________________________
