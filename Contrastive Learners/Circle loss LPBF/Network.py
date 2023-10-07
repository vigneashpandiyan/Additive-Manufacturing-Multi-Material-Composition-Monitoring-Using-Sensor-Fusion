# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:19:48 2021

@author: srpv
"""
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch.nn as nn
import torch


class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class Network(nn.Module):
    def __init__(self, droupout, emb_dim):
        super(Network, self).__init__()
        #torch.Size([100, 1, 5000])
        self.dropout = droupout

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=16),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

        )

        self.fc = nn.Sequential(
            nn.Linear(64*13, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
            # nn.Linear(32, emb_dim),
            # PrintLayer(),

        )

    def forward(self, x):
        PrintLayer(),
        # x = x.permute(0, 2, 1)
        PrintLayer(),
        x = self.conv(x)
        x = x.view(-1, 832)
        x = self.fc(x)

        return nn.functional.normalize(x)
