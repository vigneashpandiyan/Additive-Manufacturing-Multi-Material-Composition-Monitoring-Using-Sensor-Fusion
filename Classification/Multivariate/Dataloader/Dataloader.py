# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:33:51 2023

@author: srpv
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # implementing train-test-split
import os


class Mechanism(Dataset):

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
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database


def data_pipeline(Material, total_path):
    windowsize = 5000
    classfile = str(Material)+'_classspace'+'_' + str(windowsize)+'.npy'
    rawfile = str(Material)+'_rawspace'+'_' + str(windowsize)+'.npy'
    classfile = (os.path.join(total_path, classfile))
    rawfile = (os.path.join(total_path, rawfile))

    classspace = np.load(classfile).astype(np.int64)
    classspace = np.expand_dims(classspace, axis=1)
    rawspace = np.load(rawfile).astype(np.float64)
    rawspace = pd.DataFrame(rawspace)
    rawspace = dataprocessing(rawspace)
    # rawspace =rawspace.to_numpy()

    rawspace = pd.DataFrame(rawspace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    rawspace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]

    rawspace = rawspace.to_numpy()
    classspace = classspace.to_numpy()

    return rawspace, classspace


def Data_torch(classspace, Featurespace_1, Featurespace_2):

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
