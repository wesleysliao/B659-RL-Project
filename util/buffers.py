# Format of saving experiences:
# Since memory is not an issue here, let's allow redundancy and have th einput and output of all.
#state, action, reward, next_state


import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    # Adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    # For using as batch, follow this recipe:
        #Make sure Transition is defined:
#         Transition = namedtuple('Transition',
#                                 ('state', 'action', 'next_state', 'reward'))
#         In training loop
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
    
    def __init__(self, capacity, tag=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.tag = tag # The tag is not used within this class. The main purpose is keeping an attribute about the data that is allowed in memory. 

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
        
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

    
    
# class SupervisionBuffer():
#     def __init__(self, maxsize, tol=0.01):
#         self.maxsize = maxsize
#         self.data = deque([], maxlen=maxsize)
#         self.target = deque([], maxlen=maxsize)
#         self.tolerance = tol
# #         self.size = 0
            
#     def append(self, data, target):
#         self.data.appendleft(data)
#         self.target.appendleft(target)
# #         self.size =len(self.data)
        
#     def sample_batch(self, size):
#         self_size = len(self.data)
#         if size>self_size:
#             sample_size = self_size
#         else:
#             sample_size = size
#         mask = np.random.randint(self_size, size=sample_size)
#         return narr(self.data)[mask], narr(self.target)[mask]
    
#     def __len__(self):
#         return len(self.data)
