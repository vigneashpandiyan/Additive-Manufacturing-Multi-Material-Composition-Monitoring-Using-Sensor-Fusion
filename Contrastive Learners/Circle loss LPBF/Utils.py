# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#libraries to import
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from Dataloader import *
from torchvision import transforms

#%%
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
marker = ["*", ">", "X", "o", "s"]
colors = ['blue', 'g', 'red', 'orange', 'purple']
mnist_classes = ['20%-Cu', '40%-Cu', '60%-Cu', '80%-Cu', '100%-Cu']

graph_title = "Feature space distribution"

#%%
def init_weights(m):
    """
    Initializes the weights of a convolutional layer using the Kaiming normal initialization.

    Args:
        m (nn.Conv1d): The convolutional layer to initialize.

    Returns:
        None
    """
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)

#%%
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

#%%
def plot_embeddings(embeddings, targets, graph_name_2D, xlim=None, ylim=None):
    """
    Plots the embeddings in a 2D graph.

    Args:
        embeddings (numpy array): The embeddings to be plotted.
        targets (numpy array): The corresponding targets for each embedding.
        graph_name_2D (string): The name of the output graph file.
        xlim (tuple, optional): The x-axis limits for the plot.
        ylim (tuple, optional): The y-axis limits for the plot.

    Returns:
        None
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    count = 0
    for i in np.unique(targets):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.7,
                    color=colors[count], marker=marker[count], s=100)
        count = count+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes, bbox_to_anchor=(1.42, 1.05))
    plt.xlabel('Weights_1', labelpad=10)
    plt.ylabel('Weights_2', labelpad=10)
    plt.title(str(graph_title), fontsize=15)
    plt.savefig(graph_name_2D, bbox_inches='tight', dpi=600)
    plt.show()

#%%
def plot_embeddings_reduced(embeddings, targets, graph_name_2D, test_size, xlim=None, ylim=None):
    """
    Plots the reduced embeddings in a 2D graph.

    Parameters:
    - embeddings (array-like): The embeddings to be plotted.
    - targets (array-like): The corresponding targets for each embedding.
    - graph_name_2D (str): The name of the output graph file.
    - test_size (float): The proportion of data to be used for testing.
    - xlim (tuple, optional): The limits for the x-axis.
    - ylim (tuple, optional): The limits for the y-axis.

    Returns:
    None
    """
    embeddings, _, targets, _ = train_test_split(
        embeddings, targets, test_size=test_size, random_state=66)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    count = 0
    for i in np.unique(targets):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.7,
                    color=colors[count], marker=marker[count], s=100)
        count = count+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes, bbox_to_anchor=(1.42, 1.05))
    plt.xlabel('Weights_1', labelpad=10)
    plt.ylabel('Weights_2', labelpad=10)
    plt.title(str(graph_title), fontsize=15)
    plt.savefig(graph_name_2D, bbox_inches='tight', dpi=600)
    plt.show()

#%%
def TSNEplot(z_run, test_labels, graph_name, test_size, ang, perplexity):
    """
    Plot the lower dimension representation of the latent space using t-SNE algorithm.

    Parameters:
    z_run (array-like): Array of latent space features fed row-wise.
    test_labels (array-like): Groundtruth variable.
    graph_name (str): Name of the output graph file.
    test_size (float): Proportion of the dataset to include in the test split.
    ang (float): Azimuthal angle for the 3D plot.
    perplexity (float): Perplexity parameter for the t-SNE algorithm.

    Returns:
    ax (matplotlib.axes.Axes): The 3D plot axes.
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    output = z_run
    # array of latent space, features fed rowise

    target = test_labels
    # groundtruth variable
    output, _, target, _ = train_test_split(output, target, test_size=test_size, random_state=66)
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ', perplexity)

    group = target
    group = np.ravel(group)

    RS = np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]
    x3 = tsne_fit[:, 2]

    df = pd.DataFrame(dict(x=x1, y=x2, z=x3, label=group))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq = np.sort(uniq)

    print(uniq)

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2

    ax = plt.axes(projection='3d')

    ax.grid(False)
    ax.view_init(azim=ang)  # 115

    marker = ["*", ">", "X", "o", "s"]
    color = ['blue', 'g', 'red', 'orange', 'purple']

    ax.set_facecolor('white')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    graph_title = "Lower dimension representation"

    j = 0
    for i in uniq:
        print(i)
        indx = group == i
        a = x1[indx]
        b = x2[indx]
        c = x3[indx]
        ax.plot(a, b, c, color=color[j], label=uniq[j], marker=marker[j], linestyle='', ms=7)
        j = j+1

    plt.xlabel('Dimension-1', labelpad=10)
    plt.ylabel('Dimension-2', labelpad=10)
    ax.set_zlabel('Dimension-3', labelpad=10)
    plt.title(str(graph_title), fontsize=15)

    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.zticks(fontsize = 25)

    plt.legend(mnist_classes, loc='upper left', frameon=False)
    plt.savefig(graph_name, bbox_inches='tight', dpi=400)
    plt.show()
    return ax, fig

#%%
def compute_embeddings(model, train_df, device, dataset, folder_created):
    """
    Compute embeddings for the given model and training data.

    Args:
        model (torch.nn.Module): The model used for computing embeddings.
        train_df (torch.utils.data.Dataset): The training dataset.
        device (torch.device): The device to run the computation on.
        dataset (str): The name of the dataset.
        folder_created (str): The folder path where the embeddings and labels will be saved.

    Returns:
        tuple: A tuple containing the computed embeddings and corresponding labels.
    """

    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_df, 0):
            data, output = batch
            img, label = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
            label = label.squeeze()

            img = model(img.to(device, dtype=torch.float)).cpu().numpy()
            train_results.append(img)
            labels.append(label.cpu().numpy())

    train_results = np.concatenate(train_results)
    train_labels = np.concatenate(labels)

    train_embeddings = folder_created+'/'+str(dataset)+'_embeddings' + '.npy'
    train_labelsname = folder_created+'/'+str(dataset)+'_labels'+'.npy'
    np.save(train_embeddings, train_results, allow_pickle=True)
    np.save(train_labelsname, train_labels, allow_pickle=True)

    return train_results, train_labels
