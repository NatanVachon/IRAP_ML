# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:21:34 2019

This is the main file containing the main classes and functions
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import mlp
import preprocessing as prp
import pickle as pkl
import pandas as pd
import evaluation as ev
import numpy as np
from sklearn.metrics import confusion_matrix
import os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   DEFAULT PARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
SAVE_PATH = ""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   CLASSES DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class TrainingManager:
    """
    Class used to store everything about a set of parameters and the related trained neural network
    """
    name = ""         # Name of the training manager
    params = {}       # Dictionnary containing every train parameter
    model = None      # Trained neural network
    scaler = None     # Data scaler used to normalize data
    cm = None         # Confusion matrix
    remarks = ""      # Special remarks about the manager

    def __init__(self, name = "lastTraining", layers_sizes = mlp.LAYERS_SIZES, layers_activations = mlp.LAYERS_ACTIVATIONS,
                 epochs_nb = mlp.EPOCHS_NB, batch_size = mlp.BATCH_SIZE, test_size = mlp.TEST_SIZE):
        """
        CONSTRUCTOR
        name: Name of the training manager
        layers_sizes: Number of neurons in each layer
        layers_activations: Activation functions for each layer
        epochs_nb: Number of epochs for the training
        batch_size: Batch size for the training
        test_size: Proportion of the training dataset used for testing (test set)
        """
        self.name = name
        self.params["layers_sizes"] = layers_sizes
        self.params["layers_activations"] = layers_activations
        self.params["epochs_nb"] = epochs_nb
        self.params["batch_size"] = batch_size
        self.params["test_size"] = test_size

    def __getitem__(self, index):
        return self.params[index]

    def __setitem__(self, index, item):
        self.params[index] = item

    def print_param(self):
        """
        Displays the training manager parameters
        """
        print("Name: " + self.name)
        print("Layers sizes: " + str(self.params["layers_sizes"]))
        print("Layers activations: " + str(self.params["layers_activations"]))
        print("Epochs number: " + str(self.params["epochs_nb"]))
        print("Batch size: " + str(self.params["batch_size"]))
        print("Test size: " + str(self.params["test_size"]))

    def run_training(self, dataset=None, weight=0, loss_function = mlp.jaccard_distance, verbose=1):
        """
        Train a new neural network according to the manager's paremeters
        dataset: DataFrame whose columns are features and lines are train samples

        Returns: Training history
        """
        # Prepare data
        split_dataset = prp.split_data(dataset, test_size=self.params["test_size"])
        train_dataset, self.scaler = prp.scale_and_format(split_dataset[0], split_dataset[1], split_dataset[2], split_dataset[3])
        if weight:
            weight_list = prp.compute_weights(split_dataset[2])
        else:
            weight_list = None
        
        # Train
        self.model, history = mlp.run_training(train_dataset, layers_sizes=self.params["layers_sizes"], layers_activations=self.params["layers_activations"], epochs_nb=self.params["epochs_nb"],
                                               batch_size=self.params["batch_size"], weight_list = weight_list, loss_function=loss_function, verbose=verbose)
        # return history
        return history

    def get_pred(self, data):
        """
        Predicts class
        data: Input list

        Returns: pandas.DataFrame Predicted classes
        """
        # Scale data
        # Remove label if necessary
        if "label" in data.columns:
            scaled_data = self.scaler.transform(data.drop("label", axis=1))
        else:
            scaled_data = self.scaler.transform(data)
        # Predict
        pred_classes = self.model.predict_classes(scaled_data)
        # Return DataFrame
        pred_df = pd.DataFrame()
        pred_df["label"] = pred_classes
        return pred_df

    def get_prob(self, X):
        """
        Get the probability with which the model predicted each class
        X_test: Input data to predict

        Returns: pandas.DataFrame Computed probabilities
        """
        X = self.scaler.transform(X)
        y_prob = self.model.predict(X)

        y_prob_df = pd.DataFrame()
        y_prob_df['prob_ev'] = [y_prob[i][0] for i in range(X.shape[0])]
        y_prob_df['prob_sh'] = [y_prob[i][1] for i in range(X.shape[0])]
        y_prob_df['prob_sw'] = [y_prob[i][2] for i in range(X.shape[0])]
        return y_prob_df

    def test(self, test_data):
        """
        Test step to compute confusion matrix
        tm: Training manager
        test_data: pandas.DataFrame Test dataset

        Returns: Confusion matrix
        """
        true_labels = test_data["label"]
        pred_labels = pd.DataFrame()
        pred_labels["label"] = self.get_pred(test_data.drop("label", axis=1))["label"]
        self.cm = confusion_matrix(true_labels, pred_labels)
        return self.cm

    def save(self, path=SAVE_PATH):
        """
        Saves the training manager
        path: Folder where you want to save the training manager
        """
        if self.model is None:
            print("No model attached. Nothing to save")
            return
        # Construct saved object
        to_save = {"name":self.name, "params":self.params, "scaler":self.scaler, "cm":self.cm}
        # Create folder
        if not os.path.isdir(path + self.name):
            os.mkdir(path + self.name)
        path += self.name + '/'
        # Save data
        f = open(path + self.name + ".pkl", "wb")
        pkl.dump(to_save, f, pkl.HIGHEST_PROTOCOL)
        self.model.save(path + self.name + ".h5")
        f.close()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def loadManager(path):
    """
    Loads a saved TrainingManager and returns it
    path: Path of the training manager folder

    Returns: loaded training manager
    """
    # Get folder name
    filename = path.split('/')[-1]
    # Checks if folder exists
    if not os.path.isdir(path):
        print("Training manager " + filename + " doesn't exists")
        return
    #Load data
    loaded = TrainingManager()
    f = open(path + '/' + filename + ".pkl", "rb")
    loaded_data = pkl.load(f)
    loaded.name = loaded_data["name"]
    loaded.params = loaded_data["params"]
    loaded.scaler = loaded_data["scaler"]
    loaded.cm = loaded_data["cm"]
    loaded.model = mlp.load_model(path + '/' + filename + ".h5")
    f.close()
    return loaded