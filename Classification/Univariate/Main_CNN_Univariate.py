# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
from Trainer.Trainer import *
from Dataloader.Dataloader import *
from Network.Network import *
from Utils.Utils import *
from sklearn.model_selection import train_test_split  # implementing train-test-split
from tqdm.auto import tqdm
from prettytable import PrettyTable
from torchsummary import summary
import os
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Clearing the cache
torch.cuda.empty_cache()

# %%
# Defining the Hyperparameters for training the network
batch_size = 256
epoch = 300
windowsize = 5000
Sensor_stream = ["D1", "D2"]
embedding_dims = 32

for Material in Sensor_stream:

    # %%
    # Checking the availability of the GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()

    # %%
    # Defining the path for the data  ---> Folder path
    total_path = r"C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion\Data"

    rawspace, classspace = data_pipeline(Material, total_path, windowsize)
    sequences = Data_torch(classspace, rawspace)

    train, test = train_test_split(sequences, test_size=0.2)
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=0,
                                           shuffle=True)

    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=0,
                                          shuffle=True)

    # %%
    # Defining the network and if the GPU device is avaialble then the network will be loaded and trained on the GPU
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = CNN(len(np.unique(classspace)), embedding_dims, dropout_rate=0.1)

    net.to(device)
    summary(net, (1, 5000))

    # %%
    # Training the network
    model, iteration, Loss_value, Total_Epoch, Accuracy, Learning_rate, Training_loss_mean, Training_loss_std = Network_trainer(
        net, trainset, testset, device, epoch=epoch)

    PATH = './CNN_Univariate '+str(Material)+'.pth'
    torch.save(model.state_dict(), PATH)
    torch.save(model, PATH)
    model = torch.load(PATH)

    correctHits = 0
    total = 0
    for batches in testset:
        data, output = batches
        data, output = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
        output = output.squeeze()
        prediction = model(data)
        prediction = torch.argmax(prediction, dim=1)
        total += output.size(0)
        correctHits += (prediction == output).sum().item()

    print('Accuracy = '+str((correctHits/total)*100))
    print('Finished Training')

    # %%
    # Plotting the confusion matrix and the training loss, learning rate and accuracy
    folder_created = os.path.join('Figures/', str(Material))
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    plots(iteration, Loss_value, Total_Epoch, Accuracy, Learning_rate,
          Training_loss_mean, Training_loss_std, folder_created)
    count_parameters(net)

    classes = ('1', '2', '3', '4', '5')  # Change the classes based on the dataset
    plotname = str(Material)+' CNN_Univariate'+'_CF'+'.png'
    plot_confusion_matrix(model, testset, classes, device, plotname)
