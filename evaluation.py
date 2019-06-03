# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:29:51 2019

This file contains evaluation functions
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import preprocessing as prp
import mlp

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def k_fold(manager, k, data, weights=0):
    """
    Runs a k fold validation
    k: Number of folds

    Returns:
        List: One confusion matrix per fold
    """
    cms = []
    histories = []
    test_size = 1 / k
    n = data.count()[0]

    for i in range(k):
        print("Fold " + str(i + 1) + '/' + str(k))
        # Run a complete training
        start_i = n * i // k
        timed_data = prp.split_data(data, test_size=test_size, start_index=start_i, ordered=True)
        train_dataset, manager.scaler = prp.scale_and_format(timed_data[0], timed_data[1], timed_data[2], timed_data[3])
        manager.model, history = mlp.run_training(train_dataset, layers_sizes=manager.params["layers_sizes"],layers_activations=manager.params["layers_activations"],
                                                  epochs_nb=manager.params["epochs_nb"], batch_size=manager.params["batch_size"])
        pred = manager.get_pred(timed_data[1])
        cm = confusion_matrix(timed_data[3]["label"], pred["label"])
        cms.append(cm)
        histories.append(history)
    return cms, histories

def strat_k_fold(manager, k, data, weight=0):
    """
    Runs a k fold validation with folds made by 
    preserving the percentage of samples for each class
    k: Number of folds

    Returns:
        List: One confusion matrix per fold
    """
    cms = []
    histories = []

    X = data.drop('label',axis=1)
    Y = pd.DataFrame(data['label'],columns=['label'])
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state=1)
    skf.get_n_splits(X,Y)
    
    fold = 1
    for train_index, test_index in skf.split(X,Y):
        print("TRAIN:", train_index, "TEST:", test_index)
        print("Fold : ",fold)
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        train_datasets, manager.scaler = prp.scale_and_format(X_train, X_test, Y_train, Y_test)
        weight_list = prp.compute_weights(Y_train)
        manager.model, history = mlp.run_training(train_datasets, layers_sizes=manager.params["layers_sizes"],layers_activations=manager.params["layers_activations"],
                                                  epochs_nb=manager.params["epochs_nb"], batch_size=manager.params["batch_size"],weight_list=weight_list)
        pred = manager.get_pred(X_test)
        cm = confusion_matrix(Y_test["label"], pred["label"])
        cms.append(cm)
        histories.append(history)
    return cms, histories
    
    

def plot_histograms(datas):
    """
    Plots histograms per features
    datas: list of pandas.DataFrame to analize
    """
    n = len(datas)
    for column in datas[0]:
        print(column)
        plt.figure()
        for data in datas:
            plt.hist(data[column], bins=50, alpha=1.0/n)
        plt.legend([str(i) for i in range(n)])
        plt.show()