# Agent class
import numpy as np
# sys.path.append('agents/')
# sys.path.append('lib/')
import base #from base import DyadSliderAgent
import torch
from torch import FloatTensor as tarr
from scipy.special import softmax
import random

class RolePDAgent(base.DyadSliderAgent):
    
    def __init__(self, pdcoef, sigma, perspective, force_rms=1., **kwargs):
        super().__init__(force_max=20., force_min=-20.,
                       perspective=perspective, **kwargs)
        self.pd = pdcoef
        self.sigma = sigma
        self.force_rms = force_rms
    
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
        
        
    def get_qvals(self, state_batch, qnet):
        # This method is public, meant to be used by q-learning or TD update.
        # state_batch is a batch of env states, formed into a numpy arr.
        # returns a torch tensor of qvals, one element for each action.
        
        # Create the action batch
        force0_batch = np.zeros((len(state_batch),1)); force1_batch = np.zeros((len(state_batch),1))
        i=0;
        for sample_s in state_batch:
            force0_batch[i] = self._apply_pid(sample_s)
            force1_batch[i] = self._addnoise_cap(0.)
        
        sa_batch0 = np.concatenate( (state_batch, force0_batch),axis=1)
        sa_batch1 = np.concatenate( (state_batch, force1_batch),axis=1)
        
        # Assuming the input to qnet is [state, action]
        q0 = qnet(tarr(sa_batch0))
        q1 = qnet(tarr(sa_batch1))
        
        return torch.cat((q0, q1))
                  
        
    def _qbased_force(self, env_state, qnet, eps=1.):
        # returns the appropriate role given the q
        
        force0 = self._apply_pid(env_state)
        force1 = self._addnoise_cap(0.)
        
        # Assuming the input to qnet is [state, action]
        q0 = qnet(torch.cat( (tarr(env_state), tarr([force0])) )) 
        q1 = qnet(torch.cat( (tarr(env_state), tarr([force1])) ))
        
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
        
        
    def get_force(self, env_state, role=0, qnet=None, eps=1, verbose=False):
        # role=0: PID controller
        # role=1: do nothing.
        
        if role == 1:
            return 0.
        elif qnet is not None:
            force, qvals = self._qbased_force(env_state, qnet, eps=eps)
            if verbose is True:
                return force, qvals
            return force
        else:
            return self._apply_pid(env_state) #case: role=0
        
        
    
    def effort(self, force):
        return self.c_effort* abs(force/self.force_rms)
    
    def compute_utility(self, reward, force):
        return self.c_error*reward - self.effort(force)