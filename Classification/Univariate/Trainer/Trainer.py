
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:33:51 2023

@author: srpv
"""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']


def Network_trainer(net, trainset, testset, device, epoch):

    costFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    Loss_value = []
    Train_loss = []
    Iteration_count = 0
    iteration = []
    Epoch_count = 0
    Total_Epoch = []
    Accuracy = []
    Learning_rate = []

    Training_loss_mean = []
    Training_loss_std = []

    for epoch in range(epoch):
        epoch_smoothing = []
        learingrate_value = get_lr(optimizer)
        Learning_rate.append(learingrate_value)
        closs = 0
        scheduler.step()

        for i, batch in enumerate(trainset, 0):

            data, output = batch
            data, output = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
            prediction = net(data)
            loss = costFunc(prediction, output.squeeze())  # torch.Size([100, 3]),#torch.Size([100])
            closs += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_smoothing.append(loss.item())

            if i % 10 == 0:
                print('[%d  %d] loss: %.4f' % (epoch+1, i+1, loss))

        loss_train = closs / len(trainset)
        Loss_value.append(loss_train)

        Training_loss_mean.append(np.mean(epoch_smoothing))
        Training_loss_std.append(np.std(epoch_smoothing))

        correctHits = 0
        total = 0

        for batches in testset:

            data, output = batches
            data, output = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
            output = output.squeeze()
            prediction = net(data)
            prediction = torch.argmax(prediction, dim=1)
            total += output.size(0)
            correctHits += (prediction == output).sum().item()

        Epoch_count = epoch+1
        Total_Epoch.append(Epoch_count)
        Epoch_accuracy = (correctHits/total)*100
        Accuracy.append(Epoch_accuracy)
        print('Accuracy on epoch [%d] :  %.3f' % (epoch+1, Epoch_accuracy))

    iteration = np.array(iteration)
    Loss_value = np.array(Loss_value)
    Total_Epoch = np.array(Total_Epoch)
    Accuracy = np.array(Accuracy)
    Learning_rate = np.array(Learning_rate)
    Training_loss_mean = np.array(Training_loss_mean)
    Training_loss_std = np.array(Training_loss_std)

    return net, iteration, Loss_value, Total_Epoch, Accuracy, Learning_rate, Training_loss_mean, Training_loss_std
