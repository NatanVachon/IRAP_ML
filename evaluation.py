# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:29:51 2019

This file contains evaluation functions
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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