# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#libraries to import

from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split  # implementing train-test-split


class Mechanism(Dataset):
    """
    This class represents a dataset for a specific mechanism.

    Args:
        sequences (list): A list of tuples containing two sequences and a label.

    Attributes:
        sequences (list): A list of tuples containing two sequences and a label.

    Returns:
        tuple: A tuple containing the sequence and its corresponding label.
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
        # sequence,label
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
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database


def Data_torch(classspace, Featurespace_1, Featurespace_2):
    """
    Preprocesses the input data and creates train and test datasets for torch.

    Args:
        classspace (numpy.ndarray): The class labels for each data sample.
        Featurespace_1 (pandas.DataFrame): The first feature space data.
        Featurespace_2 (pandas.DataFrame): The second feature space data.

    Returns:
        tuple: A tuple containing the train and test datasets.
            trainset (torch.utils.data.DataLoader): The train dataset.
            testset (torch.utils.data.DataLoader): The test dataset.
    """

    Featurespace_1 = pd.DataFrame(Featurespace_1)
    Featurespace_1 = dataprocessing(Featurespace_1)

    Featurespace_2 = pd.DataFrame(Featurespace_2)
    Featurespace_2 = dataprocessing(Featurespace_2)

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

    Featurespace_1 = Featurespace_1.to_numpy().astype(float)
    Featurespace_2 = Featurespace_2.to_numpy().astype(float)

    sequences = []
    for i in range(len(classspace)):
        # print(i)
        sequence_features_1 = Featurespace_1[i]
        sequence_features_2 = Featurespace_2[i]
        label = classspace[i]
        sequences.append((sequence_features_1, sequence_features_2, label))

    sequences = Mechanism(sequences)

    train, test = train_test_split(sequences, test_size=0.2)
    trainset = torch.utils.data.DataLoader(train, batch_size=100, num_workers=0,
                                           shuffle=True)

    testset = torch.utils.data.DataLoader(test, batch_size=100, num_workers=0,
                                          shuffle=True)

    return trainset, testset
