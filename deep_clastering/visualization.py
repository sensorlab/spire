import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from typing import Tuple

def get_averaged_clusters(
    pseudo_labels:np.array, 
    spectograms:np.ndarray,
    real_labels:np.ndarray|None=None)->Tuple[np.ndarray, list]:
    """
    pseudo_labels: pseudo labels of the samples
    spectograms: raw data of samples, shape = (num_samples, signal_length)
    reduced_features: images features in the reduced space
    real_labels: real labels of the samples
    selected_ind: selected indices of the samples to reduce comput. time
    
    Calculate averaged spectrograms and 
    real classes dist.(if provided) among clusters
    """
    pseudo_classes   = np.unique(pseudo_labels)
    num_classes      = len(pseudo_classes)
    num_real_classes = 0
    spectograms_length = spectograms.shape[1]
    
    # ignore undetermined samples
    if -1 in pseudo_classes:
        num_classes -= 1
    
    if not real_labels is None:
        num_real_classes = len(np.unique(real_labels))
    
    # [k,i] cell stands for number of i'th class representatives in the k'th cluster
    cluster_real_classes = [[0] * num_real_classes for _ in range(num_classes)]

    averaged_spectograms = np.zeros((num_classes, spectograms_length))
    num_samples          = np.zeros(num_classes)
    
    for i in range(len(pseudo_labels)):

        # ignore undetermined samples
        if pseudo_labels[i] == -1:
            continue
        
        curr_pseudo_label = pseudo_labels[i]

        averaged_spectograms[curr_pseudo_label] += spectograms[i]
        num_samples[curr_pseudo_label] += 1
        
        if not real_labels is None:
            cluster_real_classes[curr_pseudo_label][real_labels[i]] += 1
    
    averaged_spectograms /= num_samples[:, np.newaxis]
    
    return averaged_spectograms, cluster_real_classes
    
def plot_averaged_clusters_spectograms(averaged_clusters:np.ndarray):
    """
    averaged_clusters: avaraged spectograms of determined clusters

    Show averaged spectogram for each claster
    """
    fig, ax = plt.subplots()
    
    for i, avg_claster_spectogram in enumerate(averaged_clusters):
        ax.plot(np.squeeze(avg_claster_spectogram), label='Cluster: ' + str(i), alpha=0.5)
            
    ax.set_title('Average per-cluster')
    ax.set_xlabel('FFT bins')
    
    ax.grid()
    ax.legend(bbox_to_anchor=(-0.1, 1.05), ncol=1)
    
    return  fig, ax 


def plot_tsne(
    labels:np.array, 
    reduced_features:np.ndarray, 
    n_components:int=2):
    """
    labels: any kind of data labels
    reduced_features: images features in the reduced space
    
    perform the t-distributed stochastic neighbor embedding
    (https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
    and show it
    """
    tsne_solver = TSNE(n_components=n_components)
    tsne_embedings = tsne_solver.fit_transform(reduced_features)
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        ax.scatter(tsne_embedings[labels == label][:,0], 
                    tsne_embedings[labels == label][:,1], 
                    s=1, 
                    edgecolors=None, 
                    label=str(label)
                   )
    ax.legend()
    return fig, ax

def plot_labels_dist_across_clusters(
    clusters_classes:list,
    labels_dict:dict):
    """
    For each cluster plot real labels dist.
    """
    fig, axs = plt.subplots()
    ax.rc('font', size=12)
    num_clusters = len(clusters_classes)
    for i in range(num_clusters):
        ax = axs[i]
        ax.set_title('Cluster: ' + str(i))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.grid(which='major', axis='y', alpha=0.5)
        
        ax.bar(list(labels_dict.keys()), clusters_classes[i])
        ax.set_xticklabels(list(labels_dict.keys()), rotation=60)
        
    return fig, axs

def plot_var_ratio(
    pca_explained_variance_ratio:np.array, 
    treshhold:float=99/100):
    """
    Plot PCA's components explanined var ratio
    """
    cum_sum_var_ratio = np.cumsum(pca_explained_variance_ratio)
    # Cut the tail
    cum_sum_var_ratio = cum_sum_var_ratio[cum_sum_var_ratio <= treshhold]

    fig, ax = plt.subplots()

    ax.set_title('EVR of the first ' + str(len(cum_sum_var_ratio)) + ' components', size=14)
    ax.set_ylabel('Cumulative sum')
    ax.set_xlabel('Principal components')
    ax.plot(range(len(cum_sum_var_ratio)), cum_sum_var_ratio, color='y')
    
    return fig, ax


    