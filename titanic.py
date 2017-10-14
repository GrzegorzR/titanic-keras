#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:51:07 2017

@author: gr
"""

import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier
from  data import prepare_test_data, prepare_train_data
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
    
def prepare_network(features_num):
    hidden_size = 50
    drop_out = 0
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=features_num))
    model.add(Dropout(drop_out))
    model.add(Dense(hidden_size, activation='relu', input_dim=features_num))
    model.add(Dropout(drop_out))
    model.add(Dense(2, activation='softmax'))
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])    
    return model    

    
def train(X, y, network):
    
    network.fit(X, y, nb_epoch =5, batch_size = 20)
    #print network.predict(X)
    
    return network
    
def test_data_result(X, network):    
    dataset = pd.read_csv("test.csv")
    
    results = network.predict(X)
    survived = []
    for result in results:
        survived.append(int(result[0]< result[1]))

    
    df2 = pd.DataFrame({"PassengerId":dataset["PassengerId"],
                        "Survived" : survived
                        })
    return df2

def prepare_cm(y_pred, y_test):
    survived1 = []
    for result in y_pred:
        survived1.append(int(result[0]< result[1]))
        
    survived2 = []
    for result in y_test:
        survived2.append(int(result[0]< result[1]))
    return confusion_matrix(survived1, survived2)   
    
def main():
    X_test = prepare_test_data("test.csv")
    X, y = prepare_train_data("train.csv")
    X, x_test, y, y_test = train_test_split(X, y, test_size = 0.1)
    
    model = prepare_network(14)
    
    #train(X,y,X_test,y_test, model)
    train(X,y, model)
    result = test_data_result(X_test, model)
    
    y_pred = model.predict(x_test)

    cm =   prepare_cm(y_pred, y_test)
    
    return result, cm 
    
res, cm = main()
res.to_csv("result.csv", index = False)




















