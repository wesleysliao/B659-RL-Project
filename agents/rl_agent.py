# Agent class
import numpy as np
# sys.path.append('agents/')
# sys.path.append('lib/')
from agents.base import DyadSliderAgent
import torch
from torch import FloatTensor as tarr
from scipy.special import softmax
import random

class DQNAgent(DyadSliderAgent):
    # imports: numpy as np
    # softmax from scipy.special, 
    # FloatTensor as tarr from torch
    
    #__________________________ Interface methods
    def __init__(self, rl, pdcoef, buffer, perspective, sigma,
                 hyperparams=None, force_rms=1., **kwargs):
        super().__init__(force_max=20., force_min=-20.,
                       perspective=perspective, **kwargs)
        self.rl = rl
        self.pd = pdcoef
        self.buffer = buffer
        self.sigma = sigma
        self.force_rms = force_rms
        self.hp = hyperparams
    
    def set_train_hyperparams(self, hyperparams):
        self.hp = hyperparams
        
    def add_experience(self, trans_tuple):
        self.buffer.push(*trans_tuple)
    
    def train_step(self):
        if len(self.buffer)>1:
            # Run one step of SGD using the whole batch data
            self.rl.step(self.buffer)
    
    def get_force(self, env_state, eps=1., role=None, verbose=False):
        
        # role=0: PID controller
        # role=1: do nothing.
        
        if role is None:
            force, qvals = self._qbased_force(env_state, eps=eps)
            if verbose is True:
                return force, qvals
            return force
        elif role == 1:
            return 0.
        elif role == 0:
            return self._apply_pid(env_state) #case: role=0
        else:
            raise ValueError('role should be in {0,1,None}')
        
    def update_target_qnet(self):
        self.rl.update_target_qnet()
    
    
    #__________________________ Methods used in train_step() by the rl torch model
    def get_force_batch(self, state_batch):
        force0_batch = np.zeros((len(state_batch),1)); force1_batch = np.zeros((len(state_batch),1))
        i=0;
        for sample_s in state_batch:
            force0_batch[i] = self._apply_pid(sample_s)
            force1_batch[i] = self._addnoise_cap(0.)
        
        return (force0_batch, force1_batch)
        
    
    def compute_utility(self, reward, force):
        return self.c_error*reward - self._compute_effort(force)
    
    #__________________________ Methods used by the agent itself
    
    def _compute_effort(self, force):
        return self.c_effort* abs(force/self.force_rms)
    
    
    def _qbased_force(self, env_state, eps=0.):
        # returns the appropriate role given the q
        
        force0 = self._apply_pid(env_state)
        force1 = self._addnoise_cap(0.)
        
        # Assuming the input to qnet is [state, action]
        q0 = self.rl.net(torch.cat( (tarr(env_state), tarr([force0])) )) 
        q1 = self.rl.net(torch.cat( (tarr(env_state), tarr([force1])) ))
        
        q0 = float(q0); q1 = float(q1)
        qvals = [q0, q1]
        
        if eps>0. and random.random()<eps:
            probs = softmax([q0,q1])
            if random.random()< probs[0]:
                return force0, qvals
            else:
                return force1, qvals
        else:
            if q0>q1:
                return force0, qvals
            else:
                return force1, qvals
    
    
    def _addnoise_cap(self, force):
        mlt_noise = force*np.random.normal(0, self.sigma)
        add_noise = np.random.normal(0, self.sigma)
        noisy_force = force+ mlt_noise+add_noise
        force_capped = self.force_max*np.tanh((2./self.force_max) *noisy_force)
        return force_capped
        
        
    def _apply_pid(self, env_state):
        # env_state can be any 1dimensional indexable sequence
        
        e = env_state[0]-env_state[2]
        ed = env_state[1]-env_state[3]
        if self.perspective !=0:
            e = -e; ed=-ed
        force = np.dot(self.pd, [e,ed])
        capped_noisy_force = self._addnoise_cap(force)
        return capped_noisy_force