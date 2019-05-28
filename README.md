## IRAP Multiclass MLP Tutorial
This is a library used for multiclass prediction problems using multilayers perceptrons.
This tutorial will show you how to be able to use the library and how to use it.

# Summary
  - Installation
  - Data format
  - Functions explanation

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
  
