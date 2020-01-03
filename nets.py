# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:15:57 2020

@author: lazar
"""
import torch
import torch.nn as nn
from torch.nn.functional import relu


def FFNN2net(indput_size,l1, l2):
    class Net(nn.Module):
         def __init__(self):
             super(Net, self).__init__()
             
     
             self.dropout = nn.Dropout(p=0.5)
     
     
             self.l1 = nn.Linear(in_features = indput_size,
                                 out_features = l1)
             
             self.bn1 = torch.nn.BatchNorm1d(l1)
             
             self.l2 = nn.Linear(in_features = l1,
                                 out_features = l2)
     
             self.bn2 = torch.nn.BatchNorm1d(l2)
             
             self.l_out = nn.Linear(in_features=l2,
                                 out_features=1,
                                 bias=False)
             
         def forward(self, x):
     
             # Output layer
             x = self.l1(x)
             x = self.bn1(x)
             x = self.dropout(x)
             x = relu(x)
             
             x = self.l2(x)
             x = self.bn2(x)
             x = self.dropout(x)
             x = relu(x)
             
             x = self.l_out(x)
             x = torch.sigmoid(x)
             
             return x
        
    return Net()
