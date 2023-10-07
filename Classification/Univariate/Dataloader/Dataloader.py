# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:33:51 2023

@author: srpv
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class Mechanism(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        sequence, label = self.sequences[idx]
        sequence = torch.Tensor(sequence)
        sequence = sequence.view(1, -1)
        label = torch.tensor(label).long()
        sequence, label
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

    print("respective windows", data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())

    print("windows of the class: ", minval)

    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])
    print("The dataset is well balanced: ", data.Categorical.value_counts())

    rawspace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]

    rawspace = rawspace.to_numpy()
    classspace = classspace.to_numpy()

    return rawspace, classspace


def Data_torch(classspace, Featurespace):

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
