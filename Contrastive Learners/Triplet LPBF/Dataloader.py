# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:27:06 2021

@author: srpv
"""


from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from torchvision import transforms
import os
import pandas as pd


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
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    rawspace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]
    classspace = classspace.to_numpy()

    rawspace = dataprocessing(rawspace)
    rawspace = rawspace.to_numpy()

    return rawspace, classspace


class MNIST(Dataset):
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
        #anchor_img = self.images[item].reshape(28, 28, 1)
        anchor_img = self.images[item]

        # print(item)

        if self.is_train:
            anchor_label = self.labels[item]
            # print(anchor_label)

            positive_list = self.index[self.index !=
                                       item][self.labels[self.index != item] == anchor_label]
            # print(positive_list)

            positive_item = random.choice(positive_list)
            #positive_img = self.images[positive_item].reshape(28, 28, 1)
            positive_img = self.images[positive_item]

            negative_list = self.index[self.index !=
                                       item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            #negative_img = self.images[negative_item].reshape(28, 28, 1)
            negative_img = self.images[negative_item]

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            # if self.transform:
            #     anchor_img = self.transform(self.to_pil(anchor_img))
            label = self.labels[item]
            return anchor_img, label
