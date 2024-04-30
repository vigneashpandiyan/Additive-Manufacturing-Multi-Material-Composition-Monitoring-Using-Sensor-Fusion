# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import

from sklearn.model_selection import train_test_split  # implementing train-test-split
import time
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from Utils import *
from Network import *
from Dataloader import *
from Loss import *
from Generalization import *
from torch.optim.lr_scheduler import StepLR
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
# %%
# Clearing the cache
torch.cuda.empty_cache()
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
# %%
# Hyperparameters for the model training
embedding_dims = 16
batch_size = 256
epochs = 300
windowsize = 5000
Material_1 = "D1"
Material_2 = "D2"

# %%
# GPU Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.get_device_name()

# %%
# Defining the path for the data  ---> Folder path
#http://dx.doi.org/10.5281/zenodo.11094814
total_path = r"C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion\Data"

rawspace_1, classspace = data_pipeline(Material_1, total_path, windowsize)
rawspace_2, classspace = data_pipeline(Material_2, total_path, windowsize)
rawspace = np.stack((rawspace_1, rawspace_2), axis=2)
# %%

X_train, X_test, y_train, y_test = train_test_split(
    rawspace, classspace, test_size=0.30, random_state=123)

index_train = pd.DataFrame(y_train)
index_test = pd.DataFrame(y_test)
# %%

train_ds = Triplet_dataloader(X_train, y_train, index_train,
                              train=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
test_ds = Triplet_dataloader(X_test, y_test, index_test,
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))

#MNIST(test_df, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']


# %%
# Model definition
model = Network(droupout=0.05, emb_dim=embedding_dims)
# model.apply(init_weights)
# model = torch.jit.script(model).to(device)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
criterion = torch.jit.script(TripletLoss())

# %%
# Training the model
model.train()
Loss_value = []
Learning_rate = []

Training_loss_mean = []
Training_loss_std = []

for epoch in tqdm(range(epochs), desc="Epochs"):
    epoch_smoothing = []
    learingrate_value = get_lr(optimizer)
    Learning_rate.append(learingrate_value)
    closs = 0
    scheduler.step()
    for i, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        #torch.Size([100, 1, 5000])
        anchor_img = anchor_img.to(device, dtype=torch.float)
        # anchor_img=anchor_img.unsqueeze(1)
        # print(anchor_img.shape)
        positive_img = positive_img.to(device, dtype=torch.float)
        # positive_img=positive_img.unsqueeze(1)
        # print(positive_img.shape)
        negative_img = negative_img.to(device, dtype=torch.float)
        # negative_img=negative_img.unsqueeze(1)
        # print(negative_img.shape)

        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        loss = criterion(anchor_out, positive_out, negative_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_smoothing.append(loss.item())

        # closs = loss.item()
        closs += loss

        if i % 50 == 0:
            print('[%d  %d] loss: %.4f' % (epoch+1, i+1, loss))

    Training_loss_mean.append(np.mean(epoch_smoothing))
    Training_loss_std.append(np.std(epoch_smoothing))

    loss_train = closs / len(train_loader)
    Loss_value.append(loss_train.cpu().detach().numpy())
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, loss_train))

# %%
# Saving the model
torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict()
            }, "trained_model.pth")


folder_created = os.path.join('Figures/', (str(Material_1)+str(Material_2)))
print(folder_created)
try:
    os.makedirs(folder_created, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")
# %%
# Saving the loss values
Loss_value = np.asarray(Loss_value)
Loss_embeddings = str(Material_1)+str(Material_2)+' Loss_value'+'_Triplet' + '.npy'
np.save(Loss_embeddings, Loss_value, allow_pickle=True)


Training_loss_mean = np.asarray(Training_loss_mean)
Training_loss_mean_file = (str(Material_1)+str(Material_2))+'Training_loss_mean'+'_Triplet' + '.npy'
np.save(Training_loss_mean_file, Training_loss_mean, allow_pickle=True)

Training_loss_std = np.asarray(Training_loss_std)
Training_loss_std_file = (str(Material_1)+str(Material_2))+'Training_loss_std'+'_Triplet' + '.npy'
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
plt.savefig((str(Material_1)+str(Material_2))+'Training loss.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
# Plotting the loss values with mean and standard deviation
plt.rcParams.update(plt.rcParamsDefault)
Loss_value = pd.DataFrame(Loss_value)
Training_loss_mean = pd.DataFrame(Training_loss_mean)
Training_loss_std = pd.DataFrame(Training_loss_std)
under_line = (Training_loss_mean - Training_loss_std)[0]
over_line = (Training_loss_mean + Training_loss_std)[0]
fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(Loss_value, 'purple', linewidth=2.0, label='Triplet loss')
plt.fill_between(
    Training_loss_std.index,
    under_line,
    over_line,
    alpha=.650,
    color='#C79FEF',
    label='Triplet loss Std.'
)
plt.title('Epochs vs Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plot_1 = str(Material_1)+str(Material_2)+' Average_Loss_value' + '.png'
plt.savefig(plot_1, dpi=600, bbox_inches='tight')
plt.show()
plt.clf()

# %%
# Plotting the learning rate
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(6, 3))
plt.plot(Learning_rate, 'g', linewidth=2.0)
plt.title('Epochs vs Learning rate')
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.legend(['Learning rate'])
plt.savefig((str(Material_1)+str(Material_2))+' Learning_rate.png', dpi=600, bbox_inches='tight')
plt.show()
# %%
# Counting the number of parameters
count_parameters(model)
# %%
# Generalization of the model
X_train, X_test, y_train, y_test = generalization(
    Material_1, Material_2, windowsize, total_path, model, device)
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
graph_name_32D = folder_created + 'CNN_Latent_32D' + '_'+'.png'
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
