# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#libraries to import
import torch.nn as nn
import torch
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """
    TripletLoss is a custom loss function used for training triplet networks.
    
    Args:
        margin (float): The margin value for the triplet loss. Default is 1.0.
    
    Inputs:
        anchor (torch.Tensor): The anchor samples.
        positive (torch.Tensor): The positive samples.
        negative (torch.Tensor): The negative samples.
    
    Returns:
        torch.Tensor: The computed triplet loss value.
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
