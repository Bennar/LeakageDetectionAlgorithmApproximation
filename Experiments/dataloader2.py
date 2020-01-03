# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:57:30 2019

@author: lazar
"""

import pandas as pd
import numpy as np
import datetime


def interval_getter(dataset,start_time,end_time):
    dataset = np.delete(dataset, np.s_[0:start_time], axis=1)
    dataset = np.delete(dataset, np.s_[(end_time-start_time):], axis=1)
    return dataset

def data_wrapper(data_set_wrapper,targets,num_days):
    targets = targets[(num_days-1):]
    wrapped_dataset = np.zeros((len(data_set_wrapper)-(num_days-1),num_days*data_set_wrapper.shape[1]))
    for i in range(len(wrapped_dataset)):
        concat = np.array([])
        for j in range(num_days):
            concat = np.concatenate((concat,data_set_wrapper[(i+j)]),axis=None)
        wrapped_dataset[i] = concat
    return wrapped_dataset, targets

def data_load(path):
    data = pd.read_excel(path)
    leak = pd.read_excel(path, sheet_name='Leak')
    return data, leak

def data_prep(path='filler', day='weekday', start_time=0, end_time=23, num_days_seq = 20, t_train=0.8, t_test = 0.1, clean=True):
    # =============================================================================
    #     data_prep is a function to prepare our data to our model. It loads data,
    #     seperates it into 3 categories; saturdays, sundays and weekdays, shortens it
    #     to only contain measurements between a given start and end hour, combines days
    #     into sequences, removes days where there are 0 observations and then splits the data into train, validation and testset
    #     
    #     load: Boolean telling if we should load data
    #     path: string containing path of data
    #     day: string setting which dataset to return (saturday, sunday or weekday)
    #     start_time: integer setting start of daily measurement interval (0 is at 00:00)
    #     end_time: integer setting end of daily time interval (max is 23)
    #     num_seq_days: integer stating how many consecutive days to be put into sequence
    #     t: fraction of data to use for training
    #     t_test: fraction of dataset to use as test data
    #     clean: Boolean stating wheather to remove days with zero or not    
    # =============================================================================
    data, leak = data_load(path)

    N = data.shape[0]
    
    # Creates target vectors of boolean if leak detected or not.
    targets = np.zeros(N)
    for i in range(N):
        if data['LogDate'][i] in set(leak['LogDate']):
            targets[i] = leak['Value'][leak['LogDate'] == data['LogDate'][i]]
        
    for i in range(N):
        if targets[i] < 0:
            targets[i] = 1
        if targets[i] > 0:
            targets[i] = 1
        
    
    # We set the data up in a logic structure where the first 3 rows are feature, target and date
    # and the next 3 are boolean for weekday, saturday or sunday.
        
    data_matrix = []
    data_matrix = data[440]
    data_matrix = np.vstack((data_matrix, targets))
    data_matrix = np.vstack((data_matrix, data['LogDate']))
    
    data_matrix = np.vstack((data_matrix, np.zeros((3,N))))
    
    # We remove the first year and 71 data points, and then we scew with 23 datapoints. 
    #The year because it seem like the algorithm wasnt applied in the first year and 
    #the 71 datapoints to make the data length have modulus (7 * 24) = 0. 
    #The scew is to make sure the data starts on monday at 00:00
    
    data_matrix = data_matrix[:,(24*7*52 + 71 - 23 - 1):-24]
    
    n_weeks = int(data_matrix.shape[1]/(7*24))
    
    # We set the boolean part of the matrix accordingly
    for i in range(n_weeks):
        for j in range(24*5):    
            data_matrix[3,(i*7*24 + j)] = 1
        
        for k in range(24):
            data_matrix[4,(i*7*24 + 24*5 + k)] = 1
            
        for l in range(24):
            data_matrix[5, (i*7*24 + 24*5 + 24 + l)] = 1
        
    
    # We load a file containing day, month and year of danish puplic holiday. We
    # sort these days into sundays
    helligdage = pd.read_excel(r'D:\Desk\DTU\DeepLearn\DLProj\Danske_helligdage.xlsx')
    for i in range(data_matrix.shape[1]):
        date_time_obj = datetime.datetime.strptime(data_matrix[2,i], '%Y-%m-%d %H:%M:%S.%f')
        for j in range(helligdage.shape[0]):
            if date_time_obj.day == helligdage['day'][j]:
                if date_time_obj.month == helligdage['month'][j]:
                    if date_time_obj.year == helligdage['year'][j]:
                        data_matrix[5,i] = 1
    
    # we split the data up in 3 sets, one for weekdays, one for saturdays and one for sundays
    data_set_weekday = []
    data_set_saturday = []
    data_set_sunday = []
    for i in range(data_matrix.shape[1]):
        if data_matrix[3,i] == 1 and data_matrix[5,i] == 0:
            data_set_weekday.append(data_matrix[0:2,i])
        if data_matrix[4,i] == 1:
            data_set_saturday.append(data_matrix[0:2,i])
        if data_matrix[5,i] == 1:
            data_set_sunday.append(data_matrix[0:2,i])
    
    # we choose which dataset to use
    if day == 'weekday':
        data_set = data_set_weekday
        
    if day == 'saturday':
        data_set = data_set_saturday
            
    if day == 'sunday':
        data_set = data_set_sunday
    
    data_set = np.array(data_set)
    # The data is sorted and choosen, now we set it up as intended
    # we set our targets to daily values since the algorithm is only set to go off one time a day
    n_days = int(data_set.shape[0]/24)
    targets = np.zeros(n_days)
    for i in range(n_days):
        targets[i] = max(data_set[(i*24):(i*24+24),1])
        
    # we rearrange the data to be of shape (n_days,24)
    inputs = np.zeros((n_days,24))
    for i in range(n_days):
        inputs[i] = data_set[(i*24):(i*24+24),0]
    
    # we cuts the dataset down to the given time interval
    inputs = interval_getter(inputs,start_time,end_time)
        
    # we wrap the given consecutive days together to sequences
    inputs, targets = data_wrapper(inputs,targets,num_days_seq)
        
        
    if clean == True:
        delete_list = []
        for i in range(len(inputs)):
            if 0 in inputs[i]:
                delete_list.append(i)
            if np.isnan(np.sum(inputs[i])):
                delete_list.append(i)
                    
        inputs = np.delete(inputs, delete_list, axis=0)
        targets = np.delete(targets, delete_list)
    
    # we split the dataset into training, validation and test set.    
    # fraction of traindata/data
    N = len(inputs) 
    
    x_train = inputs[:int(N*t_train)].astype('float32')
    y_train = targets[:int(N*t_train)].astype('float32')
    x_test = inputs[int(N*t_train):int(N*(t_train+t_test))].astype('float32')
    y_test = targets[int(N*t_train):int(N*(t_train+t_test))].astype('float32')
    x_val = inputs[int(N*(t_train+t_test)):].astype('float32')
    y_val = targets[int(N*(t_train+t_test)):].astype('float32')
    
    return x_train, y_train, x_test, y_test, x_val, y_val