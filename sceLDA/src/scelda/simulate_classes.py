import numpy as np
import pandas as pd

n_genes = 10
n_informative = 2
n_classes = 4
n_cells = 200

def generate_means(n_genes, n_informative, min=50, max=300):
    '''
    Builds an array of means for each class. The first
    n_informative genes have different means between 
    classes. 

    Parameters
    ----------
        n_genes :   int
            the number of total genes in the dataset
        n_informative   :   int
            the number of informative genes (different)
            between classes
        min :   int
            the minimum mean
        max :   int
            the max mean

    Returns
    -------
        mean_arr    :   np.array
            the `n_classes x n_genes` array of means
        variances   :   np.array
            the `n_genes` array of variances 

    '''
    init_means = np.random.randint(min, max, n_genes)
    variances = np.random.randint(1, min/5, n_genes)

    mean_arr = np.zeros((n_classes, n_genes))

    for i in range(0, n_classes):
        mean_arr[i, :] = init_means

    for i in range(1, n_classes):
        for j in range(0, n_informative):
            gene_var = variances[j]

            while True:

                if np.random.rand() > 0.5:
                    new_mean = init_means[j] + np.random.randint(2, 10) * gene_var
                else:
                    new_mean = init_means[j] - np.random.randint(2, 10) * gene_var
                
                if new_mean-5*variances[j] > 0:
                    break

            mean_arr[i, j] = new_mean

    return mean_arr, variances

def generate_data(n_cells, n_genes, n_classes, mean_arr, variances):
    '''
    Generates the data matrix and labeling given means and vars. 

    Parameters
    ----------
        n_cells :   int
            the total number of cells to include in the data
        n_genes :   int
            the number of total genes in the dataset
        n_classes   :   int
            the number of classes/types/clusters in the data
        mean_arr    :   np array
            the n_classes x n_genes array of means
        variances   :   np array
            the variances of each gene
    
    Returns
    -------
        data    :   np array
            the data array cells x genes
        labels  :   np array
            the labeling assignment
        
    '''
    
    last_cell = 0
    cells_per_class = np.floor(n_cells/n_classes)
    data = np.empty((n_cells, n_genes))
    labels = np.empty(n_cells, dtype=int)

    for c in range(0, n_classes):
        
        if c == n_classes - 1:
            cells_in_class = int(n_cells - last_cell) # in case of remainders
        else:
            cells_in_class = int(cells_per_class)
        
        # generate the data for each class
        means = mean_arr[c, :]

        class_data = np.random.normal(means, variances, (cells_in_class, n_genes))
        
        labels[last_cell:last_cell+cells_in_class] = c # save the class assignment
        data[last_cell:last_cell+cells_in_class, :] = class_data #add to the larger dataset
        last_cell += cells_in_class
    
    return data, labels

def visualize_clusters(data, labels, n_classes, filename=None):
    '''
    Generates the data matrix and labeling given means and vars. 

    Parameters
    ----------
        data    :   np array
            the data array cells x genes
        labels  :   np array
            the labeling assignment
        n_classes   :   int
            the number of classes/types/clusters in the data
        filename    :   str
            the path and filename to the figure file
    Returns
    -------
        None
   
    '''
    from sklearn import manifold
    import matplotlib.pyplot as plt

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = Y = tsne.fit_transform(data)

    color_list = ['c', 'm', 'y', 'r', 'g', 'k']
    assert n_classes <= len(color_list)
    label_dict = dict(zip([i for i in range(0, n_classes)], color_list[:n_classes]))
    colors = [label_dict[i] for i in labels]

    plt.clf()
    plt.scatter(Y[:,0], Y[:,1], c = colors)
    
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)

    return

arr, variances = generate_means(n_genes, n_informative)
data, labels = generate_data(n_cells, n_genes, n_classes, arr, variances)



    