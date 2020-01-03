# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:57:14 2019

@author: Ida Maria Christensen
"""
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
import dataloader2
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = dataloader2.data_prep(load=True, path =r'C:\Users\irima\OneDrive\Desktop\DTU\04-E19\02456 Deep learning\Projekt\Fruedal.xlsx',
                                                                                            day='weekday',
                                                                                            start_time=1,
                                                                                            end_time=6,
                                                                                            num_days_seq = 60,
                                                                                            t_train=0.4,
                                                                                            t_test = 0.6,
                                                                                            clean=True)
with open('data_w.pickle', 'wb') as f:
    pickle.dump(inputs_train, f)
    pickle.dump(targets_train, f)
    pickle.dump(inputs_val, f)
    pickle.dump(targets_val, f)

    f.close()

#%%

with open('data_w.pickle', 'rb') as f:
    inputs_train = pickle.load(f)
    targets_train = pickle.load(f)
    inputs_test = pickle.load(f)
    targets_test = pickle.load(f)
    f.close()

#%%Decision Tree
# Create Decision Tree classifer object
    w = {0:1, 1:2}
    DTC = DecisionTreeClassifier(criterion='gini', class_weight=w)
    
    # Train Decision Tree Classifer
    DTC = DTC.fit(inputs_train,targets_train)
    
    #Predict the response for test dataset
    targets_pred = DTC.predict(inputs_test)
    target_names = ['No leak', 'Leak']
    print(classification_report(targets_test, targets_pred, target_names=target_names))

pickle.dump(DTC, open('decisiontree.sav', 'wb'))


#%% Random forest

w = {0:1, 1:2}
RFC = RandomForestClassifier(criterion = 'gini',n_estimators=10,class_weight=w)

RFC = RFC.fit(inputs_train, targets_train)

targets_pred = RFC.predict(inputs_test)

print(classification_report(targets_test, targets_pred, target_names=target_names))

pickle.dump(RFC, open('randomforest.sav', 'wb'))


#%% Support vector machine

SVC = svm.SVC(kernel='linear')

SVC = SVC.fit(inputs_train, targets_train)

targets_pred = SVC.predict(inputs_test)

print(classification_report(targets_test, targets_pred, target_names=target_names))

pickle.dump(SVC, open('supportvector.sav', 'wb'))

#%% Loading and plotting a saved model

loaded_model = pickle.load(open('decisiontree.sav', 'rb'))
targets_pred = loaded_model.predict(inputs_test)
print(classification_report(targets_test, targets_pred, target_names=target_names))


plt.figure()
plt.plot(targets_test, '-b')
plt.plot(-targets_pred, '-r')
plt.title("Test predictions")
plt.show()

plt.figure()
plt.plot(targets_train, '-b')
plt.plot(-loaded_model.predict(inputs_train), '-r')
plt.title("Train predictions")
plt.show()