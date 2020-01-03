# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:13:51 2019

@author: Ida Maria Christensen
"""

 
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataloader2
import pickle
from sklearn.metrics import classification_report
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

 
#load data
inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = dataloader2.data_prep(load=True, path =r'C:\Users\irima\OneDrive\Desktop\DTU\04-E19\02456 Deep learning\Projekt\Fruedal.xlsx',
                                                                                            day='weekday',
                                                                                            start_time=1,
                                                                                            end_time=6,
                                                                                            num_days_seq = 60,
                                                                                            t_train=0.4,
                                                                                            t_test = 0.6,
                                                                                            clean=True)
#pickle data
with open('data_w.pickle', 'wb') as f:
    pickle.dump(inputs_train, f)
    pickle.dump(targets_train, f)
    pickle.dump(inputs_val, f)
    pickle.dump(targets_val, f)

    f.close()

# %% load pickled data

with open('data_w.pickle', 'rb') as f:
    inputs_train = pickle.load(f)
    targets_train = pickle.load(f)
    inputs_test = pickle.load(f)
    targets_test = pickle.load(f)

    f.close()

#%%
 
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
test_set = Dataset(np.array(inputs_test), np.array(targets_test))
 
print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(test_set)} samples in the test set.')
 
trainloader = torch.utils.data.DataLoader(training_set, batch_size=200,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=200,
                                         shuffle=True)
train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)
 
# %%
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
 
 
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
 

def one_hot(x):
    """ One-hot encode vector. """

    y = np.zeros(x.shape[0])
    y[x == 1] = 0
    y[x == 0] = 1
    
    y = np.vstack((y,x))
    
    return torch.from_numpy(y)

#%% define network
 
input_size = inputs_train.shape[1]
output_size = 2
 
class Net(nn.Module):
     def __init__(self):
         super(Net, self).__init__()
         
 
         self.dropout = nn.Dropout(p=0.50)
 
 
         self.l1 = nn.Linear(in_features = input_size,
                             out_features = 150)
         
         self.bn1 = torch.nn.BatchNorm1d(150)
         
         self.l2 = nn.Linear(in_features = 150,
                             out_features = 75)
 
         self.bn2 = torch.nn.BatchNorm1d(75)
         
         self.l_out = nn.Linear(in_features=75,
                             out_features=output_size,
                             bias=False)
         
     def forward(self, x):
 
         # Output layer
         x = self.l1(x)
         x = self.bn1(x)
         x = self.dropout(x)
         x = F.relu(x)
         
         x = self.l2(x)
         x = self.bn2(x)
         x = self.dropout(x)
         x = F.relu(x)
         x = self.l_out(x)
         x = torch.sigmoid(x)
         
         return x
 
 
 
 
net = Net()
print(net)
#%% 

#define weights
w = torch.Tensor([1, 30000])

criterion = nn.BCELoss(weight = w)
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-01)

num_epoch = 1000
 
# Track loss
training_loss, test_loss = [], []

#Train loop
for epoch in range(num_epoch):  # loop over the dataset multiple times
    epoch_training_loss = 0
    epoch_test_loss = 0
   
    net.eval()
 
    for i, data_set in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data_set
        
        # one-hot encode targets
        labels = one_hot(labels)
 
        # wrap them in Variable
        inputs, labels = get_variable(Variable(inputs)), get_variable(Variable(labels))
        
        outputs = net(inputs)
       
        loss = criterion(outputs, labels.T.float())
       
        epoch_test_loss += loss.detach().numpy()
    
    net.train()
    
    for i, data_set in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data_set
        
        labels = one_hot(labels)
        
        # wrap them in Variable
        inputs, labels = get_variable(Variable(inputs)), get_variable(Variable(labels))
 
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.T.float())
        loss.backward()
        optimizer.step()
       
        epoch_training_loss += loss.detach().numpy()
 
    training_loss.append(epoch_training_loss/len(training_set))
    test_loss.append(epoch_test_loss/len(test_set))
 
print('Finished Training')
 
# Plot training and test loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, test_loss, 'b', label='Test loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()

#%% Save model
pickle.dump(net, open('weightedFFNN.sav', 'wb'))

#%%
# prediction plot of test data
n = inputs_test.shape[0]
outputs = np.zeros((n,2))
net.eval()
for i in range(n):
    inputs = torch.Tensor(inputs_test[i].reshape(1,-1))
    outputs[i] = net.forward(inputs).data.numpy()
 
outputs = np.round(-outputs)
scale = inputs_test/np.max(inputs_test)
means = np.zeros(inputs_test.shape[0])
for i in range(inputs_test.shape[0]):
    means[i] = np.mean(scale[i,-6:])-np.mean(scale[i, :-6])
 
plt.figure(figsize=(16,10), dpi= 80)
plt.title('FFNN (weighted) predictions', fontsize=30)
plt.plot(targets_test[:int(n)], 'b-')
plt.plot(outputs[:int(n),1], 'r-')
#plt.plot(means, 'g-')
plt.grid(axis='both', alpha=.3)
plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
plt.yticks(fontsize=20, alpha=.7)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['targets', '- predictions'], fontsize = 25)
plt.show()

target_names = ['No leak', 'Leak']

print(classification_report(targets_test, -outputs[:,1], target_names=target_names))

#%% Load saved model
n = inputs_test.shape[0]
loaded_model = pickle.load(open('weightedFFNN.sav', 'rb'))
loaded_model.eval()
for i in range(n):
    inputs = torch.Tensor(inputs_test[i].reshape(1,-1))
    outputs[i] = loaded_model.forward(inputs).data.numpy()

outputs = np.round(outputs)

print(classification_report(targets_test, outputs[:,1], target_names=target_names))
