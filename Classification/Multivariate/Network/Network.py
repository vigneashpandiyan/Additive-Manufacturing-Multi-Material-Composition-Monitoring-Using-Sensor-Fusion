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
from torch import nn, optim
from torch.nn import functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) class for classification tasks.

    Args:
        classes (int): Number of classes for classification.
        embedding_dims (int): Dimensionality of the embedding layer.
        dropout_rate (float): Dropout rate for regularization.

    Attributes:
        dropout_rate (float): Dropout rate for regularization.
        conv1 (nn.Conv1d): First convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer after the first convolutional layer.
        conv2 (nn.Conv1d): Second convolutional layer.
        bn2 (nn.BatchNorm1d): Batch normalization layer after the second convolutional layer.
        conv3 (nn.Conv1d): Third convolutional layer.
        bn3 (nn.BatchNorm1d): Batch normalization layer after the third convolutional layer.
        conv4 (nn.Conv1d): Fourth convolutional layer.
        bn4 (nn.BatchNorm1d): Batch normalization layer after the fourth convolutional layer.
        conv5 (nn.Conv1d): Fifth convolutional layer.
        bn5 (nn.BatchNorm1d): Batch normalization layer after the fifth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        pool (nn.MaxPool1d): Max pooling layer.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(x): Forward pass of the CNN.
    """
    
    def __init__(self, classes, embedding_dims, dropout_rate):
        super(CNN, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=16)
        self.bn1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16)
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(16, 32, kernel_size=16)
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(832, embedding_dims)
        self.fc2 = nn.Linear(embedding_dims, embedding_dims)
        self.fc3 = nn.Linear(embedding_dims, classes)

        self.pool = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        x = x.view(-1, 832)

        x = self.fc1(x)

        x = self.fc2(x)
        x = self.fc3(x)

        return x
