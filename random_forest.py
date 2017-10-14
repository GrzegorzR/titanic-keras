#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:18:34 2017

@author: gr
"""

from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

from  data import prepare_test_data, prepare_train_data

# Set random seed
np.random.seed(0)








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


def main():
    X_test = prepare_test_data("test.csv")
    X, y = prepare_train_data("train.csv")
    forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)   
    clf.fit(X, y)
    result = test_data_result(X_test, clf)
    result.to_csv("result.csv", index = False)

main()