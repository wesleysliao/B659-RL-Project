import gym
import sys, time
import numpy as np; 
from numpy import asarray as narr
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
import torch
from torch import nn, optim
import pickle

# Import custom scripts
sys.path.append('configs/')
sys.path.append('agents/')
sys.path.append('util/')
import env_old
from util import buffers, nn_models, torch_trainer
from agents import rl_agent, train_agents, benchmark_agents

# Constants that don't change across experiments. Be careful! lots of global vars.
# this import is run before each experiment, which serves the prupose of resetting 
# those parameters they've been changes.

from configs.fixed_exp_constants import * 

exp_name = 'exp1.3'
target_int = 16
c_effort = 0.

hyperparams = BMHyperparams(batch_size, learning_rate, 
                buffer_max_size, experience_sift_tol, target_int)

agent1 = rl_agent.DQNAgent(rl1, pdcont1, buffer1, perspective=0, sigma=sigma, 
                 hyperparams=hyperparams, c_error=1., c_effort=c_effort,
                 force_rms=1.)
agent2 = rl_agent.DQNAgent(rl2, pdcont2, buffer2, perspective=1, sigma=sigma, 
                 hyperparams=hyperparams, c_error=1., c_effort=c_effort,
                 force_rms=1.)
rl1.set_agent(agent1); rl2.set_agent(agent2)

agent1.set_train_hyperparams(hyperparams)
agent2.set_train_hyperparams(hyperparams)

algo = train_agents.train_dyad
x_fpx, y_fpx, agent1_fpx, agent2_fpx = benchmark_agents.benchmark(algo, hyperparams, env, 
                                       agent1, agent2, xaxis_params)

pickle.dump((x_fpx, y_fpx, agent1_fpx, agent2_fpx), open( "data/"+exp_name+"results.p", "wb" ) )