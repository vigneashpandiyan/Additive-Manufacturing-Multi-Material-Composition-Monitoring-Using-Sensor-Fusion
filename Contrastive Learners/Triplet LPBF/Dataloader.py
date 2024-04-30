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
import torch
from torchvision import transforms
import os
import pandas as pd

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

    def __init__(self,sequences):
        self.sequences = sequences
    
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self,idx):
        sequence,label =  self.sequences [idx]
        sequence=torch.Tensor(sequence)
        sequence = sequence.view(1, -1)
        label=torch.tensor(label).long()
        sequence,label
        return sequence,label
    
def dataprocessing(df):
    """
    Preprocesses the input dataframe by standardizing its values.

    Args:
        df (pandas.DataFrame): The input dataframe to be processed.

    Returns:
        pandas.DataFrame: The processed dataframe with standardized values.
    """
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database


def data_pipeline(Material, total_path, windowsize):
    """
    Process the data for a given material.

    Args:
        Material (str): The name of the material.
        total_path (str): The total path of the data files.
        windowsize (int): The size of the window.

    Returns:
        tuple: A tuple containing the processed rawspace and classspace data.
            rawspace (ndarray): The processed rawspace data.
            classspace (ndarray): The classspace data.

    """
    # Function implementation goes here

    
    classfile = str(Material)+'_classspace'+'_' + str(windowsize)+'.npy'
    rawfile = str(Material)+'_rawspace'+'_' + str(windowsize)+'.npy'
    classfile = (os.path.join(total_path, classfile))
    rawfile = (os.path.join(total_path, rawfile))

    classspace = np.load(classfile).astype(np.int64)
    classspace = np.expand_dims(classspace, axis=1)
    rawspace = np.load(rawfile).astype(np.float64)

    rawspace = pd.DataFrame(rawspace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    rawspace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]
    classspace = classspace.to_numpy()

    rawspace = dataprocessing(rawspace)
    rawspace = rawspace.to_numpy()

    return rawspace, classspace


class Triplet_dataloader(Dataset):
    """
    Dataset class for triplet data loading.

    Args:
        data (list): List of input data.
        label (list): List of corresponding labels.
        df (pandas.DataFrame): DataFrame containing the labels.
        train (bool): Flag indicating whether the dataset is for training or not.
        transform (callable): Optional transform to be applied to the input data.

    Returns:
        tuple: A tuple containing the anchor image, positive image, negative image, and anchor label.
    """

    def __init__(self, data, label, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform

        if self.is_train:
            self.images = data
            self.labels = label
            self.index = pd.DataFrame(label).index.values
        else:
            self.images = data
            self.labels = label
            self.index = pd.DataFrame(label).index.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item]

        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.index !=
                                       item][self.labels[self.index != item] == anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item]

            negative_list = self.index[self.index !=
                                       item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item]

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            label = self.labels[item]
            return anchor_img, label
