# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#libraries to import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
from sklearn import metrics
import collections
import joblib
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.pyplot import specgram
import seaborn as sns
from scipy.stats import norm
# import joypy as jp
from matplotlib import cm
from scipy import signal
import pywt
import matplotlib.patches as mpatches
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
import os

#%%

def Cummulative_plots(Featurespace,classspace,i,ax):
        
    """
    Plot cumulative distribution of features based on categorical labels.

    Args:
        Featurespace (numpy.ndarray): Array of features.
        classspace (numpy.ndarray): Array of categorical labels.
        i (int): Index of the feature to plot.
        ax (matplotlib.axes.Axes): Axes object to plot the cumulative distribution.

    Returns:
        None
    """
    
    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'20%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'40%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'60%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'80%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(4,'100%-Cu')
    df2 = pd.DataFrame(df2)
    print(columns) 
    
    print(i)
    
    Featurespace_1 = Featurespace.transpose()
    data=(Featurespace_1[i])
    data=data.astype(np.float64)
    
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
    df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
    data = pd.concat([df1, df2], axis=1)
    minval = min(data.categorical.value_counts())
    data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
    
    Cummulative_dist_plot(data,i,ax)

#%%
    
def Cummulative_dist_plot(data, i, ax):
    """
    Plot the cumulative distribution of different target values.

    Args:
        data (pandas.DataFrame): The input data containing the target values and feature column.
        i (int): The weight index.
        ax (matplotlib.axes.Axes): The axes object to plot the cumulative distribution.

    Returns:
        None
    """

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == '20%-Cu']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == '40%-Cu']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == '60%-Cu']
    data_3 = data_3.drop(labels='target', axis=1)
    data_4 = data[data.target == '80%-Cu']
    data_4 = data_4.drop(labels='target', axis=1)
    data_5 = data[data.target == '100%-Cu']
    data_5 = data_5.drop(labels='target', axis=1)
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    sns.set(style="white")
    fig=plt.subplots(figsize=(5,3), dpi=800)
    sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="blue",ax=ax)
    sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="green",ax=ax)
    sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="red",ax=ax)
    sns.kdeplot(data_4['Feature'], shade=True,alpha=.5, color="orange",ax=ax)
    sns.kdeplot(data_5['Feature'], shade=True,alpha=.5, color="purple",ax=ax)
    

    ax.set_title("Weight " + str(i+1), y=1.0, pad=-14)
    ax.set_xlabel('Weight distribution') 
    # ax.set_ylabel('Density')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    

#%%

def distribution_plot(data, i, Material, folder_created):
    """
    Plots the distribution of feature values for different target classes.

    Args:
        data (DataFrame): The input data containing feature values and target classes.
        i (int): The weight index.
        Material (str): The material name.
        folder_created (str): The name of the folder where the plot will be saved.

    Returns:
        None
    """

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == '20%-Cu']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == '40%-Cu']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == '60%-Cu']
    data_3 = data_3.drop(labels='target', axis=1)
    data_4 = data[data.target == '80%-Cu']
    data_4 = data_4.drop(labels='target', axis=1)
    data_5 = data[data.target == '100%-Cu']
    data_5 = data_5.drop(labels='target', axis=1)
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    sns.set(style="white")
    fig=plt.subplots(figsize=(5,3), dpi=800)
    fig = sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="blue")
    fig = sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="green")
    fig = sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="red")
    fig = sns.kdeplot(data_4['Feature'], shade=True,alpha=.5, color="orange")
    fig = sns.kdeplot(data_5['Feature'], shade=True,alpha=.5, color="purple")
    
    
    data=pd.concat([data_1,data_2,data_3],axis=1) 
    data=data.to_numpy()
    
    plt.title("Weight " + str(i+1))
    plt.legend(labels=['20%-Cu', '40%-Cu', '60%-Cu', '80%-Cu', '100%-Cu'],bbox_to_anchor=(1.49, 1.05))
    title=folder_created+'/'+str(Material)+'_'+str(i+1)+'_'+'distribution_plot'+'.png'
    # plt.xlim([0.0, np.max(data)])
    # plt.ylim([0.0, 65])
    plt.xlabel('Weight distribution') 
    plt.savefig(title, bbox_inches='tight')
    plt.show()
    
#%%

def plots(Featurespace, classspace, Material, folder_created):
    """
    Plot the Featurespace and classspace data.

    Args:
        Featurespace (numpy.ndarray): The Featurespace data.
        classspace (numpy.ndarray): The classspace data.
        Material (str): The material name.
        folder_created (str): The folder path where the plots will be saved.
    """

    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2 = df2['Categorical'].replace(0, '20%-Cu')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, '40%-Cu')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, '60%-Cu')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(3, '80%-Cu')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(4, '100%-Cu')
    df2 = pd.DataFrame(df2)
    print(columns)
    
    for i in range(columns):
        print(i)
        Featurespace_1 = Featurespace.transpose()
        data = (Featurespace_1[i])
        data = data.astype(np.float64)
        # data = abs(data)
        df1 = pd.DataFrame(data)
        df1.rename(columns={df1.columns[0]: "Feature" }, inplace=True)
        df2.rename(columns={df2.columns[0]: "categorical" }, inplace=True)
        data = pd.concat([df1, df2], axis=1)
        
        minval = min(data.categorical.value_counts())
        data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique()])
        
        distribution_plot(data, i, Material, folder_created)
       
 #%%    

def plots_choice(i, Featurespace, classspace):
    """
    Plot the choice of features and class space.

    Args:
        i (int): The index of the feature space to plot.
        Featurespace (numpy.ndarray): The feature space.
        classspace (numpy.ndarray): The class space.

    Returns:
        None
    """

    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'20%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'40%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'60%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'80%-Cu')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(4,'100%-Cu')
    df2 = pd.DataFrame(df2)
    print(columns)
    
    
    Featurespace = Featurespace.transpose()
    data=(Featurespace[i])
    data=data.astype(np.float64)
    #data= abs(data)
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
    df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
    data = pd.concat([df1, df2], axis=1)
    
    minval = min(data.categorical.value_counts())
    data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
    
    distribution_plot(data,i)
       
 