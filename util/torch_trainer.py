from copy import deepcopy
import buffers
import numpy as np
from numpy import asarray as narr
import torch
from torch import FloatTensor as tarr

class TorchTrainer():
    # Base class for agent.rl, agent.predictive
    # Attributes: net, optim, criterion
    # imports: none
            
    def __init__(self, net, optim, criterion, agent):
        self.net = net
        self.optim = optim
        self.criterion = criterion
        self.agent = agent
    
    def set_agent(self, agent):
        self.agent = agent
        
    def _prepare_pred_target(self, buffer):
        # Prepare a batch of predictions and a batch of targets, using the buffer and the actions in the agent.
        # For RL: input: agent.buffer, agent.get_f_for_all_actions(state)
        # For supervised: input: agent.buffer, 
        # pred, target should be returned as torch variables with gradients
        #return pred, target
        raise NotImplementedError
        
    def step(self, buffer):
        prediction, target = self._prepare_pred_target(buffer)
        self.optim.zero_grad() #zero all of the gradients
        loss = self.criterion(prediction, target)
        loss.backward()# Backward pass: compute gradient of the loss with respect to model parameters.
        self.optim.step()# Update model parameters

class DQNTrainer(TorchTrainer):
    # imports: Transition, 
    # deepcopy from copy, numpy as np, np.asarray as narr
    # torch.FloatTensor as tarr
    
    def __init__(self, net, optim, criterion, agent=None):
        super().__init__(net, optim, criterion, agent)
        self.target_qnet = deepcopy(net)
    
    
    def _prepare_pred_target(self, buffer):
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        batch = buffers.Transition(*zip(*transitions))
        # batch.state is a tuple. each entry is one sample.
        # each sample is a list of the feature vars.
        # For batch.action, batch.reward, the sample is a float.
        sa_batch = np.concatenate( (narr(batch.state), 
                                    narr(batch.action)[:,np.newaxis]),axis=1)
        sa_batch = tarr(sa_batch)
        q_pred = self.net(sa_batch).view(-1)
        
        #calculate target qvals
        reward_batch = tarr(narr(batch.reward))
        next_s_batch_narr = narr(batch.next_state)
        force0_batch_narr, force1_batch_narr = self.agent.get_force_batch(next_s_batch_narr)
        next_sa0_batch_narr = np.concatenate( (next_s_batch_narr, force0_batch_narr),axis=1)
        next_sa1_batch_narr = np.concatenate( (next_s_batch_narr, force1_batch_narr),axis=1)
        q0 = self.net(tarr(next_sa0_batch_narr))
        q1 = self.net(tarr(next_sa1_batch_narr))
        target_qvals = reward_batch+ torch.cat((q0, q1)).max()
            
        return q_pred, target_qvals
    
    
    def update_target_qnet(self):
        self.target_qnet.load_state_dict(self.net.state_dict())
        self.target_qnet.eval()
    
class SupervisedTrainer(TorchTrainer):
    def _prepare_pred_target(self, buffer):
        pass