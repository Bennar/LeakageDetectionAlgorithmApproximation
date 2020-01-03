# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:07:53 2019

@author: lazar
"""
# Deeplearn Proj

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataloader2


inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = dataloader2.data_prep(load=True,
                                                                                            path=r'D:\Desk\DTU\DeepLearn\DLProj\Fruedal data.xlsx',
                                                                                            day='weekday',
                                                                                            start_time=1,
                                                                                            end_time=6,
                                                                                            num_days_seq = 50,
                                                                                            t_train=0.4,
                                                                                            t_test=0.6,
                                                                                            clean=True)

# %%
from torch.utils import data

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
testloader = torch.utils.data.DataLoader(test_set, batch_size=50,
                                         shuffle=True)
train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)

# %%
use_cuda = torch.cuda.is_available()
# print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu


indput_size = inputs_train.shape[1]
output_size = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = 10,
                               kernel_size = 15,
                               stride = 1)
        
        self.pool1 = nn.MaxPool1d(4, stride = 2)
        
        self.bn1 = nn.BatchNorm1d(10)
        
        self.lstm = nn.LSTM(input_size = 10, hidden_size = 8, num_layers = 2, bias=True, dropout=0.5)
        
        self.l_out = nn.Linear(in_features=8,
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
        x = h[1].view(-1, 8)
        x = relu(x)
        x = self.l_out(x)
        x = torch.sigmoid(x)
        return x


# Choose net and convert it to GPU
net = Net()
if use_cuda:
    net.cuda()


criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=0.001)

num_epoch = 4000

# Track loss
training_loss, validation_loss = [], []

for epoch in range(num_epoch):  # loop over the dataset multiple times
    epoch_training_loss = 0
    epoch_validation_loss = 0
    
    net.eval()

    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        
        inputs.unsqueeze_(0)
        inputs = inputs.permute(1,0,2)
        # wrap them in Variable
        inputs, labels = get_variable(Variable(inputs)), get_variable(Variable(labels))


        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        epoch_validation_loss += loss.cpu().detach().numpy()
    
    net.train()
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs.unsqueeze_(0)
        inputs = inputs.permute(1,0,2)

        # wrap them in Variable
        inputs, labels = get_variable(Variable(inputs)), get_variable(Variable(labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_training_loss += loss.cpu().detach().numpy()

    training_loss.append(epoch_training_loss/len(training_set))
    validation_loss.append(epoch_validation_loss/len(validation_set))

print('Finished Training')

torch.save(net, 'CNNLSTM.ph')

# %%
net.eval()
# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()
     
# prediction plot
n = inputs_train.shape[0]
outputs = np.zeros(n)
net = net.cpu()
for i in range(n):
    inputs = torch.Tensor(inputs_train[i].reshape(1,-1))
    inputs.unsqueeze_(0)
    inputs = inputs.permute(1,0,2)
    outputs[i] = net.forward(inputs).data.numpy()
 
outputs = np.round(-outputs)

plt.figure()
plt.title('Trainset predictions')
plt.plot(targets_train[:n], 'b-')
plt.plot(outputs, 'r-')
plt.show()

# prediction plot
n = inputs_test.shape[0]
outputs = np.zeros(n)
for i in range(n):
    inputs = torch.Tensor(inputs_test[i].reshape(1,-1))
    inputs.unsqueeze_(0)
    inputs = inputs.permute(1,0,2)
    outputs[i] = net.forward(inputs).data.numpy()
 
outputs = np.round(-outputs)

plt.figure()
plt.title('validations predictions')
plt.plot(targets_test[:n], 'b-')
plt.plot(outputs, 'r-')
plt.show()

# %%
from sklearn.metrics import classification_report
target_names = ['Bonger ikke ud', 'Bonger ud']

print(classification_report(targets_test, -outputs, target_names=target_names))

# %%

# prediction plot
n = int(inputs_test.shape[0]/2)
outputs = np.zeros(n)
 
for i in range(n):
    inputs = torch.Tensor(inputs_test[i].reshape(1,-1))
    inputs.unsqueeze_(0)
    inputs = inputs.permute(1,0,2)
    outputs[i] = net.forward(inputs).data.numpy()
 
outputs = np.round(-outputs)
 
scale = inputs_val/np.max(inputs_test)
means = np.zeros(inputs_test.shape[0])
for i in range(inputs_val.shape[0]):
    means[i] = np.mean(scale[i,-6:])-np.mean(scale[i, :-6])
   

plt.figure(figsize=(16,10), dpi= 80)
plt.title('CNN + LSTM predictions', fontsize=30)
plt.plot(targets_test[:n], 'b-')
plt.plot(outputs, 'r-')
#plt.plot(means[:n], 'g-')
plt.grid(axis='both', alpha=.3)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.yticks(fontsize=20, alpha=.7)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['targets', 'predictions'], fontsize = 25)
plt.show()