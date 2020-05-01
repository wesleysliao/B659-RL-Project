import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

    

class NetL1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, nout, bias=False)
        
    def forward(self, x):
        out = self.fc1(x)
        return out
    

class NetL1b(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, nout, bias=True)
        
    def forward(self, x):
        out = self.fc1(x)
        return out
    
    
class NetRelu1L1(nn.Module):

    def __init__(self, nin, nout, n_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(nin, n_hidden)
#         self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(n_hidden, nout)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, nout)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))#self.bn1(F.relu(self.fc1(x)))
        out = self.fc2(z1)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        return out


class NetRelu3L1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
#     def __init__(self, nin, nout):
#         super(PredictiveNet, self).__init__()
#         self.fc1 = nn.Linear(nin, 20)
#         self.fc2 = nn.Linear(20, nout)
        
#     def forward(self, x):
#         z1 = F.relu(self.fc1(x))
#         out = self.fc2(z1)
#         return out

    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, nout)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        out = self.fc4(z3)
        return out
    
    

class NetGlu2L1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, 128)
        self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, nout)
        
    def forward(self, x):
        z1 = F.glu(self.fc1(x))
        z2 = F.glu(self.fc2(z1))
        out = self.fc3(z2)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        return out
