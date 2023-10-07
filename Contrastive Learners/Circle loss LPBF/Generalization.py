# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:48:52 2023

@author: srpv
"""

from Utils import *
from Network import *
from Dataloader import *
from Loss import *
from Classifier import *
from Visualization import *
from matplotlib import animation


def generalization(Material_1, Material_2, trainset, testset, total_path, model, device):

    folder_created = os.path.join('Figures/', (str(Material_1)+str(Material_2)))
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    train_results, train_labels = compute_embeddings(
        model, trainset, device, (str(Material_1)+str(Material_2))+'_train', folder_created)
    test_results, test_labels = compute_embeddings(
        model, testset, device, (str(Material_1)+str(Material_2))+'_test', folder_created)

    graph_name_2D = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Feature_2D'+'.png'
    plot_embeddings(test_results, test_labels, graph_name_2D)

    graph_name_2D = folder_created+'/' + \
        (str(Material_1)+str(Material_2))+'_Feature_2D_reduced'+'.png'
    plot_embeddings_reduced(test_results, test_labels, graph_name_2D, test_size=0.60)

    graph_name_3D = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Feature_3D' + '.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Tsne_3D'+'.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    graph_name_3D = folder_created+'/' + \
        (str(Material_1)+str(Material_2))+'_Feature_3D_reduced'+'.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Tsne_3D_reduced'+'.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    return train_results, test_results, train_labels, test_labels

    # return bayes_embeddings,bayes_labels,model_classifier
