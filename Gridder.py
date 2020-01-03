# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:29:55 2020

@author: lazar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataloader2
from torch.utils import data

inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = dataloader2.data_prep(
                                                                                            path=r'D:\Desk\DTU\DeepLearn\DLProj\Fruedal data.xlsx',
                                                                                            day='weekday',
                                                                                            start_time=1,
                                                                                            end_time=6,
                                                                                            num_days_seq = 50,
                                                                                            t_train=0.4,
                                                                                            t_test=0.6,
# %%                                                                                            clean=True)
class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y

training_set = Dataset(np.array(inputs_train), np.array(targets_train))
validation_set = Dataset(np.array(inputs_val), np.array(targets_val))
test_set = Dataset(np.array(inputs_test), np.array(targets_test))

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')

trainloader = torch.utils.data.DataLoader(training_set, batch_size=50,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(validation_set, batch_size=20,
                                         shuffle=True)
train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)

# %%
import torch
import torch.nn as nn
import Trainers as trainers
import nets as nets
import torch.optim as optim
import pickle

use_cuda = torch.cuda.is_available()

indput_size = inputs_train.shape[1]
output_size = 1

# =============================================================================
# lr = [0.001, 0.0005, 0.0001, 0.00005]
# weight_decay = [1e-02, 1e-03, 1e-04, 0]
# l1 = [200, 120, 60]
# l2 = [60, 30, 10]
# num_epochs = [1000, 2000, 3000]
# 
# =============================================================================
lr = [0.001, 0.0005]
weight_decay = [1e-03]
l1 = [120, 60]
l2 = [30, 15]
num_epochs = [2000]

max_f1 = 0
attri = [0,0,0,0,0]
for lr in lr:
    for weight_decay in weight_decay:
        for l1 in l1:
            for l2 in l2:
                for num_epochs in num_epochs:
                    
                    net = nets.FFNN2net(indput_size,l1,l2)
                    net.cuda()
                    
                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    net = trainers.trainer(net, trainloader, optimizer,  criterion, num_epoch)
                    prec, recall, f1 = trainers.evaluator(net, inputs_val, targets_val)
                    
                    if f1[1] > max_f1:
                        torch.save(net, open('FFNN2net.sav', 'wb'))
                        max_f1 = f1[1]
                        attri[0] = lr
                        attri[1] = weight_decay
                        attri[2] = l1
                        attri[3] = l2
                        attri[4] = num_epochs
                    