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


def CNNnet(kernel_size1, out_channels1, stride1, kernel_size2, out_channels2, stride2):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
    
            self.dropout = nn.Dropout(p=0.5)
    
            self.conv1 = nn.Conv1d(in_channels = 1,
                                   out_channels = out_channels1,
                                   kernel_size = kernel_size1,
                                   stride = stride1)
            
            self.pool1 = nn.MaxPool1d(4, stride = 2)
            
            self.bn1 = nn.BatchNorm1d(20)
            
            self.conv2 = nn.Conv1d(in_channels = out_channels1,
                                   out_channels = out_channels2,
                                   kernel_size = kernel_size2,
                                   stride = stride2)
            
            self.bn2 = nn.BatchNorm1d(30)
            
            self.pool2 = nn.MaxPool1d(3, stride = 3)
            
            self.l1 = nn.Linear(in_features = out_channels*kernel_size2/stride2+1,
                                out_features = 150)
            
            self.bn3 = nn.BatchNorm1d(150)
            
            self.l2 = nn.Linear(in_features = 130,
                                out_features = 30)
            
            self.l_out = nn.Linear(in_features=150,
                                out_features=output_size,
                                bias=False)
            
        def forward(self, x):
    
            # Output layer
            x = self.conv1(x)
            x = self.dropout(x)
            x = self.bn1(x)
            x = relu(x)
            x = self.pool1(x)
            
    #        x = self.conv2(x)
    #        x = self.dropout(x)
    #        x = self.bn2(x)
    #        x = relu(x)
    #        x = self.pool2(x)
            x = x.view(-1, 20*115)
            
            x = self.l1(x)
            x = self.dropout(x)
            x = self.bn3(x)
            x = relu(x)
            
    #        x = self.l2(x)
    #        x = self.dropout(x)
    #        x = relu(x)
            
            x = self.l_out(x)
            x = torch.sigmoid(x)
            
            return x
    return Net()

def LSTMnet(num_layers, hidden_size):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            
            self.lstm = nn.LSTM(input_size = 1, hidden_size = hidden_size, num_layers = num_layers, bias=True, dropout=0.5)
            
            self.l_out = nn.Linear(in_features=hidden_size,
                                out_features=1,
                                bias=False)
            
        def forward(self, x):
            
            x = x.permute(2,0,1)     
            x, (h, c) = self.lstm(x)
            x = h[1].view(-1, 30)
            x = relu(x)
    
            x = self.l_out(x)
            
            x = torch.sigmoid(x)
            return x
    return Net()

def CNNLSTMstackednet(out_channels,kernel_size,stride, hidden_size, num_layers):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
    
            self.dropout = nn.Dropout(p=0.5)
    
            self.conv1 = nn.Conv1d(in_channels = 1,
                                   out_channels = out_channels,
                                   kernel_size = kernel_size,
                                   stride = stride)
            
            self.pool1 = nn.MaxPool1d(4, stride = 2)
            
            self.bn1 = nn.BatchNorm1d(10)
            
            self.lstm = nn.LSTM(input_size = 10, hidden_size = hidden_size, num_layers = num_layers, bias=True, dropout=0.5)
            
            self.l_out = nn.Linear(in_features=hidden_size,
                                out_features=output_size,
                                bias=False)
            
        def forward(self, x):
    
            # Output layer
            x = self.conv1(x)
            x = self.dropout(x)
            x = self.bn1(x)
            x = relu(x)
            x = self.pool1(x) 
            x = x.permute(2,0,1)        
            x, (h, c) = self.lstm(x)
            x = h[-1].view(-1, 8)
            x = relu(x)
            x = self.l_out(x)
            x = torch.sigmoid(x)
            return x
    return Net()

