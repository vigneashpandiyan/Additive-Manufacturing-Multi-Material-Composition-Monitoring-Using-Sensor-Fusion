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
from typing import Tuple
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    
    """
    Converts the label tensor into similarity matrices for positive and negative pairs.

    Args:
        normed_feature (Tensor): The normalized feature tensor.
        label (Tensor): The label tensor.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the similarity matrices for positive and negative pairs.
    """

    # Calculate the similarity matrix
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)

    # Create a label matrix
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    # Create positive and negative matrices
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    # Flatten the similarity and label matrices
    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    # Return the similarity matrices for positive and negative pairs
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    
class CircleLoss(nn.Module):
    """
    CircleLoss is a custom loss function used for contrastive learning.
    
    Args:
        m (float): Margin parameter for the loss function.
        gamma (float): Scaling parameter for the loss function.
    
    Inputs:
        sp (Tensor): Positive similarity tensor.
        sn (Tensor): Negative similarity tensor.
    
    Returns:
        Tensor: The computed loss value.
    """
    
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
