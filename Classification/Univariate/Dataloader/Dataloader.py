# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#%% Libraries required:
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # implementing train-test-split
import os


class Mechanism(Dataset):
    """
    A custom dataset class for handling sequences and labels.

    Args:
        sequences (list): A list of tuples containing two sequences and a label.

    Returns:
        tuple: A tuple containing the concatenated sequence and the label in the tensor format.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_1, sequence_2, label = self.sequences[idx]
        sequence_1 = torch.Tensor(sequence_1)
        sequence_2 = torch.Tensor(sequence_2)
        sequence1 = sequence_1.view(1, -1)
        sequence2 = sequence_2.view(1, -1)
        sequence = torch.cat((sequence1, sequence2), 0)
        label = torch.tensor(label).long()
        return sequence, label


def dataprocessing(df):
    """
    Preprocesses the input dataframe by standardizing the values.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The preprocessed dataframe with standardized values.
    """
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    return database


def data_pipeline(Material, total_path, windowsize):
    """
    Process the data for a given material.

    Args:
        Material (str): The name of the material.
        total_path (str): The total path to the data files.
        windowsize (int): The window size for processing the data.

    Returns:
        tuple: A tuple containing the processed rawspace and classspace arrays.
    """
    classfile = str(Material) + '_classspace' + '_' + str(windowsize) + '.npy'
    rawfile = str(Material) + '_rawspace' + '_' + str(windowsize) + '.npy'
    classfile = (os.path.join(total_path, classfile))
    rawfile = (os.path.join(total_path, rawfile))

    classspace = np.load(classfile).astype(np.int64)
    classspace = np.expand_dims(classspace, axis=1)
    rawspace = np.load(rawfile).astype(np.float64)
    rawspace = pd.DataFrame(rawspace)
    rawspace = dataprocessing(rawspace)

    rawspace = pd.DataFrame(rawspace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    rawspace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]

    rawspace = rawspace.to_numpy()
    classspace = classspace.to_numpy()

    return rawspace, classspace


def Data_torch(classspace, Featurespace):
    
    
    """
    Preprocesses the data and creates train and test sets for PyTorch DataLoader.

    Args:
        classspace (numpy.ndarray): The categorical labels for the data.
        Featurespace (numpy.ndarray): The first set of features.
        
    Returns:
        tuple: A tuple containing the train and test sets as PyTorch DataLoader objects.
    """

    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']
    df2 = df2['Categorical'].replace(1, 0)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, 1)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(3, 2)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(4, 3)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(5, 4)
    df2 = pd.DataFrame(df2)

    uniq = list(set(df2['Categorical']))
    uniq = np.sort(uniq)
    classspace = df2.to_numpy().astype(float)

    sequences = []
    for i in range(len(classspace)):
        # print(i)
        sequence_features = Featurespace[i]
        label = classspace[i]
        sequences.append((sequence_features, label))

    sequences = Mechanism(sequences)

    return sequences




