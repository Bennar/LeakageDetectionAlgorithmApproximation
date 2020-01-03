# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:20:15 2020

@author: lazar
"""

from torch.autograd import Variable
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

use_cuda = torch.cuda.is_available()

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

def trainer(net, trainloader, optimizer,  criterion, num_epoch):
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        
        net.train()
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
    
            # wrap them in Variable
            inputs, labels = get_variable(Variable(inputs)), get_variable(Variable(labels))
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return net


def permutetrainer(net, trainloader, optimizer,  criterion, num_epoch):
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        
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
            
    return net

def evaluator(net, inputs_val, targets_val ):
    net.eval()
    n = inputs_val.shape[0]
    outputs = np.zeros(n)
    net = net.cpu()
    for i in range(n):
        inputs = torch.Tensor(inputs_val[i].reshape(1,-1))
        outputs[i] = net.forward(inputs).data.numpy()
    outputs = np.round(outputs)
    prec, recall, f1, support =  precision_recall_fscore_support(targets_val, outputs)
    return prec, recall, f1



def evaluatorpermute(net, inputs_val, targets_val ):
    net.eval()
    n = inputs_val.shape[0]
    outputs = np.zeros(n)
    net = net.cpu()
    for i in range(n):
        inputs = torch.Tensor(inputs_val[i].reshape(1,-1))
        inputs.unsqueeze_(0)
        inputs = inputs.permute(1,0,2)
        outputs[i] = net.forward(inputs).data.numpy()
    
    prec, recall, f1, support =  precision_recall_fscore_support(targets_val, outputs)
    return prec, recall, f1


    