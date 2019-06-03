# IRAP Multiclass MLP Tutorial
This is a library used for multiclass prediction problems using multilayers perceptrons.
This tutorial will show you how to be able to use the library and how to use it.

# Summary
  - Installation
  - Data format
  - Library explanation

# Installation
Environement: (Optional)
  Use Anaconda to make libraries management very easy:  
  https://www.anaconda.com/
  
  After installing Anaconda create a virtual environement by typing the following commands in the Anaconda prompt:  
  `conda create -n <yourEnvName> python=3.6`  
  
  It's important that the Python's version you are using is 3.6. If you don't use this version, some problems can appear.  
  To enter your new virtual environment, type:  
  `conda activate <yourEnvName>`  
  And to leave it type:  
  `conda deactivate`  
  
  Now, let's install the libraries we need.  
  To install the needed libraries, type the following commands in the anaconda command prompt:  
  ```
  conda install numpy
  conda install pandas
  conda install scikit-learn
  conda install keras
  conda install matplotlib
  pip install pickle
  ```
  (`conda install spyder` : Environnement de développement facultatif)  

  If you have a CUDA compatible gpu, use:  
  `conda install tensorflow-gpu`  
  
  else use:  
  `conda install tensorflow`  
  
  Your virtual environment is now complete.  
  You always have to be in your virtual environment to use the libraries so always be in it when you work.  
  Now download the library and put all of the files in your working directory.  

# Data format
  In this library, data is stored in dataframes which is a class from the Pandas library:  
  https://pandas.pydata.org/pandas-docs/stable/  
  Basically, dataframes are arrays whose columns can be strings.
  
  Your dataframes must follow these rules:  
  - Each line represents a sample
  - Each column represents a feature
  - One of the columns isn't a feature, it's name is "label" and it represents the sample's class
  
# Library explanation
## Initialization
  All of this library is based on the TrainingManager class. This class stores everything about trainings, tests, predctions ...  
  
  To understand how this works, lets train a mlp.  
  First we need to import the TrainingManagment file and create a training manager:  
  ```Python
  import TrainingManagment as tm
  manager = tm.TrainingManager()
  ```
  To set it's name, simply use:
  ```Python
  manager.name = "tutorial"
  ```
  
  You can now modify our parameters, for instance:  
  ```Python
  manager["batch_size"] = 64
  ```  
  The parameters you can change are:  
  - layers_sizes: (List) Each element of the list is the number of neurons in the corresponding layer (including input and output layers).
  - layers_activations: (List) List of strings representing the activation function for each layer (ex: ["relu", "tanh", "softmax"]).
  - epochs_nb: Number of epochs.
  - batch_size: Number of samples in each batch.
  - test_size: Proportion of samples used for test.
  
  To check that everything is ready before training the model, you cas use the print_param() function:
  ```Python
  manager.print_param()
  
  Name: tutorial
  Layers sizes: [8, 8, 3, 3]
  Layers activations: ['relu', 'relu', 'tanh', 'softmax']
  Epochs number: 50
  Batch size: 64
  Test size: 0.2
  ```
  ## Training
  Now that everything is ready, we can launch the training by typing:
  ```Python
  history = manager.run_training(dataset=my_dataframe, loss_function=my_loss_function, verbose=1)
  ```
  Arguments:  
  - dataset: DataFrame in the required format.  
  - loss_function: Loss function used during the training, it uses the Jacquard distance by default.
  - verbose: Amount of training feedback you want (0, 1 or 2). If you don't know what it is, you don't care about it.
  
  Returns:
  - history: Training history.

  ## Predictions 
  You can either predict class or raw probabilities by doing:
  ```Python
  pred_classes = manager.get_pred(data=input_data)
  pred_probas = manager.get_prob(data=input_data)
  ```  
  
