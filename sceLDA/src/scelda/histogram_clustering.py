import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree
from tqdm import trange

def calc_hists(points, assignments, k = 30, vocab_size = 12):
    '''
    Calculates a histogram of the visual words associated with
    the k nearest points. 

    Parameters
    ----------
        points  :   np.Array
            the coordinates of the datapoints
        assignments :  np.Array
            the visual word assignments
        k   :   int
            the number of nearest neighbors to
            include in histogram
        vocab_size  :   int
            the vocab size used in SLDA (e.g. number of unique
            visual words)
    
    Returns
    -------
        all_hists   :   np.Array
            all of the associated hists for every point
    '''
    tree = KDTree(points)

    all_hists = np.zeros((points.shape[0], vocab_size))
    
    for i in trange(0, points.shape[0]):
        point = points[i, :]
        dist, idx = tree.query(point, k=k)
        indx, count = np.unique(assignments[idx], return_counts=True)

        hist = np.zeros(vocab_size)

        for j in range(0, len(indx)):
            v = indx[j]
            hist[v] = count[j]

        hist = hist/np.sum(count)
        all_hists[i, :] = hist
    
    return all_hists

def cluster_hists(all_hists, n_topics):
    '''
    Clusters all of the histograms to assign a topic
    label

    Parameters
    ----------
        all_hists   :   np.array
            the array of all hists
        n_topics    :   int
            the number of topics to cluster into
    
    Returns
    -------
        topic_labels    :   np.array
            the topic labeling from kmeans
    '''
    kmeans = MiniBatchKMeans(n_clusters=n_topics)
    topic_labels = kmeans.fit_predict(all_hists)
    return topic_labels
    
def plot_topic_labels(topic_labels, points, fig_file):
    '''
    Plots the topic labels from kmeans clustering of hists.

    Parameters
    ----------
        topic_labels    :   np.array
            topic assignments
        points  :   np.array
            array of points
        fig_file    :   str
            path to outfile
    
    Returns
    -------
        None
    '''
    val, count = np.unique(topic_labels, return_counts=True)
    colors = ['#15b01a','#0343df', '#ff81c0', '#f97306', '#653700']
    color_dict = dict(zip(val.tolist(), colors))
    color_assigments = [color_dict[j] for j in topic_labels.tolist()]

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter(x=points[:, 0], y = points[:, 1], c=color_assigments, s = 1)
    plt.savefig(fig_file)