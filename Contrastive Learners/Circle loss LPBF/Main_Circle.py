# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#libraries to import

import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn, Tensor
from typing import Tuple
from Utils import *
from Network import *
from Loss import *
from Dataloader import *
from Generalization import *
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split  # implementing train-test-split
import os
from matplotlib import animation
from Classifiers.RF import *
from Classifiers.SVM import *
from Classifiers.NeuralNets import *
from Classifiers.kNN import *
from Classifiers.QDA import *
from Classifiers.NavieBayes import *
from Classifiers.Logistic_regression import *
from Classifiers.XGBoost import *
from Visualization import *
# %%

# Clearing the cache
torch.cuda.empty_cache()
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

#%%
# GPU Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cuda":
    torch.cuda.get_device_name()
#%%
# Hyperparameters for the model training
embedding_dims = 16
batch_size = 256
epochs = 300
windowsize = 5000
Material_1 = "D1"
Material_2 = "D2"

# %%
# Defining the path for the data  ---> Folder path  
total_path = r"C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion\Data"

featurefile_1 = 'D1_rawspace_5000.npy'  # 'AE'+'_'+ 'PSD' +'.npy'
featurefile_2 = 'D2_rawspace_5000.npy'  # 'AE'+'_'+ 'PSD' +'.npy'
classfile = 'D1_classspace_5000.npy'  # 'Classlabel'+'_'+ 'PSD' +'.npy'

featurefile_1 = (os.path.join(total_path, featurefile_1))
featurefile_2 = (os.path.join(total_path, featurefile_2))
classfile = (os.path.join(total_path, classfile))

Featurespace_1 = np.load(featurefile_1).astype(np.float64)
Featurespace_2 = np.load(featurefile_2).astype(np.float64)
classspace = np.load(classfile).astype(np.float64)

# Data_torch function helps to convert the data into torch dataset
trainset, testset = Data_torch(classspace, Featurespace_1, Featurespace_2)

# %%

def get_lr(optimizer):

    """
    Returns the learning rate of the optimizer.

    Parameters:
    optimizer (torch.optim.Optimizer): The optimizer object.

    Returns:
    float: The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        # print('Learning rate =')
        # print(param_group['lr'])
        return param_group['lr']


# %%
# Model definition
model = Network(droupout=0.05, emb_dim=embedding_dims)
# model.apply(init_weights)
# model = torch.jit.script(model).to(device)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=0.25)
# criterion = torch.jit.script(TripletLoss())
criterion = CircleLoss(m=0.25, gamma=25)
model.train()

#%%
# Training the model
Loss_value = []
Learning_rate = []
Training_loss_mean = []
Training_loss_std = []
for epoch in range(epochs):
    epoch_smoothing = []
    learingrate_value = get_lr(optimizer)
    Learning_rate.append(learingrate_value)
    closs = 0
    scheduler.step()

    for i, batch in enumerate(trainset, 0):

        data, output = batch
        data, output = data.to(device, dtype=torch.float), output.to(device, dtype=torch.long)
        output = output.squeeze()
        prediction = model(data)
        loss = criterion(*convert_label_to_similarity(prediction, output))

        closs += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_smoothing.append(loss.item())

        if i % 25 == 0:
            print('[%d  %d] loss: %.4f' % (epoch+1, i+1, loss))

    loss_train = closs / len(trainset)
    Loss_value.append(loss_train.cpu().detach().numpy())
    Training_loss_mean.append(np.mean(epoch_smoothing))
    Training_loss_std.append(np.std(epoch_smoothing))

# print('Accuracy = '+str((correctHits/total)*100))
print('Finished Training')


# %%
# Saving the model
torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict()
            }, "trained_model.pth")


# %%
# Saving the loss values
Loss_value = np.asarray(Loss_value)
Loss_embeddings = (str(Material_1)+str(Material_2))+'Loss_value'+'_Circle' + '.npy'
np.save(Loss_embeddings, Loss_value, allow_pickle=True)

Training_loss_mean = np.asarray(Training_loss_mean)
Training_loss_mean_file = (str(Material_1)+str(Material_2))+'Training_loss_mean'+'_Circle' + '.npy'
np.save(Training_loss_mean_file, Training_loss_mean, allow_pickle=True)

Training_loss_std = np.asarray(Training_loss_std)
Training_loss_std_file = (str(Material_1)+str(Material_2))+'Training_loss_std'+'_Circle' + '.npy'
np.save(Training_loss_std_file, Training_loss_std, allow_pickle=True)

Learning_rate = np.array(Learning_rate)
Learning_ratefile = (str(Material_1)+str(Material_2))+'Learning_rate'+'_Circle' + '.npy'
np.save(Learning_ratefile, Learning_rate, allow_pickle=True)

# %%
# Plotting the loss values
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(6, 3))
plt.plot(Loss_value, 'b', linewidth=2.0)
plt.title('Epochs vs Training value')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.savefig((str(Material_1)+str(Material_2)) +
            'Training loss_Circle.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
# Plotting the loss values with mean and standard deviation
plt.rcParams.update(plt.rcParamsDefault)
plt.figure(figsize=(6, 3))
Loss_value = pd.DataFrame(Loss_value)
Training_loss_mean = pd.DataFrame(Training_loss_mean)
Training_loss_std = pd.DataFrame(Training_loss_std)

under_line = (Training_loss_mean - Training_loss_std)[0]
over_line = (Training_loss_mean + Training_loss_std)[0]
fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(Loss_value, '#F97306', linewidth=2.0, label='Circle loss')
plt.fill_between(
    Training_loss_std.index,
    under_line,
    over_line,
    alpha=.650,
    label='Circle loss Std.',
    color='#FBDD7E'
)
plt.title('Epochs vs Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plot_1 = str(Material_1)+str(Material_2)+' Average_Loss_value_Circle' + '.png'
plt.savefig(plot_1, dpi=600, bbox_inches='tight')
plt.show()
plt.clf()

#%%
# Plotting the learning rate
plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(6, 3))
plt.plot(Learning_rate, 'g', linewidth=2.0)
plt.title('Epochs vs Learning rate')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Learning rate', fontsize=10)
plt.legend(['Learning rate'])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.locator_params(axis='both', nbins=6)
plt.savefig((str(Material_1)+str(Material_2)) +
            ' Learning_rate_Circle.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
# Counting the number of parameters
count_parameters(model)
# %%
# Generalization of the model
X_train, X_test, y_train, y_test = generalization(
    Material_1, Material_2, trainset, testset, total_path, model, device)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

data = pd.concat([X_train, X_test], axis=0)
label = pd.concat([y_train, y_test], axis=0)
label.columns = ['Categorical']
data = pd.concat([data, label], axis=1)


print("respective windows", data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())

print("windows of the class: ", minval)

data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique()])
print("The dataset is well balanced: ", data.Categorical.value_counts())

rawspace = data.iloc[:, :-1].to_numpy()
classspace = data.iloc[:, -1].to_numpy()

X_train, X_test, y_train, y_test, = train_test_split(rawspace, classspace, test_size=0.3)

# %%
# Visualization of the data
folder_created = os.path.join('Figures/', (str(Material_1)+str(Material_2)))
print(folder_created)
try:
    os.makedirs(folder_created, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

plots(X_train, y_train, (str(Material_1)+str(Material_2)), folder_created)
classes = np.unique(y_train)


fig, axs = plt.subplots(
    nrows=4,
    ncols=4,
    sharey=False,
    figsize=(16, 9),
    dpi=1000
)

columns = np.atleast_2d(X_train).shape[1]
graph_name_32D = 'CNN_Latent_32D' + '_Circle'+'.png'
for i in range(columns):
    ax = axs.flat[i]
    Cummulative_plots(X_train, y_train, i, ax)

fig.tight_layout()
fig.savefig(graph_name_32D)
fig.show()


# %%
# Classifiers
RF(X_train, X_test, y_train, y_test, 100, classes, folder_created)
SVM(X_train, X_test, y_train, y_test, classes, folder_created)
NN(X_train, X_test, y_train, y_test, classes, folder_created)
KNN(X_train, X_test, y_train, y_test, classes, 15, 'distance', folder_created)
QDA(X_train, X_test, y_train, y_test, classes, folder_created)
NB(X_train, X_test, y_train, y_test, classes, folder_created)
LR(X_train, X_test, y_train, y_test, classes, folder_created)
XGBoost(X_train, X_test, y_train, y_test, classes, folder_created)
