"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# %% Libraries required:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(model, testset, classes, device, plotname):
    """
    Plots the confusion matrix for a given model on a test set.

    Args:
        model (torch.nn.Module): The trained model.
        testset (torch.utils.data.Dataset): The test set.
        classes (list): The list of class labels.
        device (torch.device): The device to run the model on.
        plotname (str): The name of the plot file to save.

    Returns:
        None
    """

    y_pred = []
    y_true = []

    # iterate over test data
    for batches in testset:
        data, output = batches
        data, output = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
        output = output.squeeze()
        prediction = model(data)
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.data.cpu().numpy()
        output = output.data.cpu().numpy()
        y_true.extend(output)  # Save Truth
        y_pred.extend(prediction)  # Save Prediction

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = cmn*100
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.set(font_scale=3)
    b = sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes, cmap="coolwarm",
                    linewidths=0.1, annot_kws={"size": 25}, cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts:
        b.set_text(b.get_text() + " %")
    plt.ylabel('Actual', fontsize=25)
    plt.xlabel('Predicted', fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center", fontsize=20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname = str(plotname)
    plt.savefig(plotname, bbox_inches='tight')
    plt.show()
    plt.clf()

# %%


def plots(iteration, Loss_value, Total_Epoch, Accuracy, Learning_rate, Training_loss_mean, Training_loss_std, Material):
    """
    Plots the training loss, training accuracy, and learning rate over epochs.

    Args:
        iteration (int): The current iteration.
        Loss_value (numpy.ndarray): The loss values for each epoch.
        Total_Epoch (numpy.ndarray): The total number of epochs.
        Accuracy (numpy.ndarray): The training accuracy for each epoch.
        Learning_rate (numpy.ndarray): The learning rate for each epoch.
        Training_loss_mean (numpy.ndarray): The mean training loss for each epoch.
        Training_loss_std (numpy.ndarray): The standard deviation of training loss for each epoch.
        Material (str): The material name.

    Returns:
        None
    """

    Accuracyfile = str(Material)+'Accuracy'+'.npy'
    Lossfile = str(Material)+'Loss_value'+'.npy'
    np.save(Accuracyfile, Accuracy, allow_pickle=True)
    np.save(Lossfile, Loss_value, allow_pickle=True)

    Training_loss_mean_file = str(Material)+'Training_loss_mean'+'.npy'
    Training_loss_std_file = str(Material)+'Training_loss_std'+'.npy'
    np.save(Training_loss_mean_file, Training_loss_mean, allow_pickle=True)
    np.save(Training_loss_std_file, Training_loss_std, allow_pickle=True)

    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(Loss_value, 'r', linewidth=2.0)
    # ax.fill_between(Loss_value, Training_loss_mean - Training_loss_std, Training_loss_mean + Training_loss_std, alpha=0.9)
    plt.title('Epochs vs Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plot_1 = str(Material)+' Loss_value_' + '.png'
    plt.savefig(plot_1, dpi=600, bbox_inches='tight')
    plt.show()
    plt.clf()

    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(Total_Epoch, Accuracy, 'g', linewidth=2.0)
    plt.title('Training Epoch vs Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plot_2 = str(Material)+' Accuracy_'+'.png'
    plt.savefig(plot_2, dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(Total_Epoch, Learning_rate, 'b', linewidth=2.0)
    plt.title('Training Epoch  vs Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plot_3 = str(Material)+' Learning_rate_' + '.png'
    plt.savefig(plot_3, dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    Loss_value = pd.DataFrame(Loss_value)
    Training_loss_mean = pd.DataFrame(Training_loss_mean)
    Training_loss_std = pd.DataFrame(Training_loss_std)

    under_line = (Training_loss_mean - Training_loss_std)[0]
    over_line = (Training_loss_mean + Training_loss_std)[0]
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(Loss_value, 'r', linewidth=2.0)
    plt.fill_between(
        Training_loss_std.index,
        under_line,
        over_line,
        alpha=.650
    )
    plt.title('Epochs vs Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plot_1 = str(Material)+' Average_Loss_value' + '.png'
    plt.savefig(plot_1, dpi=600, bbox_inches='tight')

    plt.show()
    plt.clf()


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a given model.

    Args:
        model (torch.nn.Module): The model for which the parameters need to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
