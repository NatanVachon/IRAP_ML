# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:04:27 2019

This file contains functions used for data preprocessing
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def split_data(raw_data, test_size = 0.3, ordered = False, start_index = 0):
    """
    Getting the train and test sets
    Arguments:
        timed_data : a given pandas.DataFrame() to convert to test and train sets, containing at least a 'label' column
        ordered : defines if the data sets created are shuffled (default: ordered=False) or chronological (ordered=True).
        test_size : proportion of samples included in the test set (default: 0.3)

        nb_class : defines if the problem has to be downgraded to 3 classes instead of 4

    Returns:
        X_train_timed, y_train_timed, X_test_timed, y_test_timed
        (These variables are called '_timed' because they still contain the 'epoch' information of the initial data)
    """
    y = pd.DataFrame()
    y['label'] = raw_data['label']

    X = raw_data.copy().drop("label", axis=1)

    if ordered :
        if start_index == 0:
            split_index = int(test_size*raw_data.count()[0])

            X_test_timed = X.iloc[0:split_index,:]
            y_test_timed = y.iloc[0:split_index,:]

            X_train_timed = X.iloc[split_index:,:]
            y_train_timed = y.iloc[split_index:,:]

        else :
            split_index1 = start_index
            split_index2 = start_index + int(test_size*raw_data.count()[0])

            X_test_timed = X.iloc[split_index1:split_index2,:]
            y_test_timed = y.iloc[split_index1:split_index2,:]
            X_test_timed.index = [i for i in range(X_test_timed.count()[0])]
            y_test_timed.index = [i for i in range(y_test_timed.count()[0])]

            X_train_timed = pd.concat([X.iloc[0:split_index1,:], X.iloc[split_index2:,:]], ignore_index=True)
            y_train_timed = pd.concat([y.iloc[0:split_index1,:], y.iloc[split_index2:,:]], ignore_index=True)

    else:
        X_train_timed, X_test_timed, y_train_timed, y_test_timed = train_test_split(X,y,test_size = test_size)

    return X_train_timed, X_test_timed, y_train_timed, y_test_timed


def scale_and_format(raw_X_train, raw_X_test, raw_y_train, raw_y_test):
    """
    Apply scaling to input data and format test data to vectors
    raw_X_train: Training dataset
    raw_X_test: Test dataset
    raw_y_train: Training test set
    raw_y_test: Test test set

    Returns: tuple of scaled and formated data (length 4) and sklearn.preprocessing.StandardScaler
    """
    X_train = raw_X_train.copy()
    X_test = raw_X_test.copy()

    y_train = one_hot_encode(raw_y_train['label'].tolist())
    y_test =  one_hot_encode(raw_y_test['label'].tolist())

    scaler = StandardScaler().fit(X_train)
    # Scale the train and test set
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test), scaler


def one_hot_encode(labels):
    """
    Useful functions to switch between labels representations
        The 2 different representations are represented here for an example with 3 distinct classes

        labels = [0, 0, 1, 2, 1, 0, 2, 2, 0]

        y = [[1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 1],
             [0, 0, 1],
             [1, 0, 0]]
    labels: List of labels to encode

    Returns: Encoded labels
    """
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded)
    return y