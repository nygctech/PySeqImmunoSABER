import numpy as np 
from tqdm import trange
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy import spatial

def get_random_cell(data):
    '''
    Return a random cell data from the data object
    '''

    i = np.random.randint(0, len(data))
    return data.iloc[i, :]

def make_points_arr(data, xcol_name, ycol_name):
    '''
    Creates an array in the same index order as 
    data, with just the x,y coordinates. 

    Parameters
    ----------
        data    :   DataFrame
            data containing cell position 
            and marker information
        xcol_name   : str
            column name of x coordinate
        ycol_name   : str
            column name of y coordinate

    Returns
    -------
        points  :   ndarray
            array of [x,y] coordinates of 
            cells, in same ordering as data
    '''
    points = [(data[xcol_name][i], data[ycol_name][i]) for i in range(0, len(data))]        
    return np.array(points)

def get_k_nearest_points(points, k, start_cell_idx):
    '''
    Returns the index of the k nearest
    cells to the cell at indicated by the start
    index. 

    Parameters
    ---------
        points  : ndarray
            The array of x,y poistions of cells.
        k   :   int
            the number of neighbors to find
        start_cell_idx  : int or array-like
            the index/indices of the cells to find
            the kneighbors of. Can be int or list of 
            int.

    Returns
    -------
        ndarray  : 
            kneighbors of start index. if start_cell_idx
            is list, array is ordered in the same way.
    '''     

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(points)

    if type(start_cell_idx) == int:
        distance, kneighbors = knn.kneighbors(points[start_cell_idx].reshape(1,-1))
        #k = np.random.randint(3,7)
        return kneighbors[0][:k+1]

    else:
        # assume list 
        res = np.empty((len(start_cell_idx), k), dtype=int)
        for i in range(0, len(start_cell_idx)):
            distance, kneighbors = knn.kneighbors(points[i].reshape(1,-1))
            res[i, :] = kneighbors[0][:k+1]
        return res

def make_marker_arr(data, marker_col_start, marker_col_end = None):
    '''
    Extracts just the marker columns from the 
    dataset and returns them as a ndarry.

    Parameters
    ----------
    data    : DataFrame
        the dataset
    marker_col_start    :   int
        index of the first marker column
    marker_col_end  :   int
        index of the last marker column.
        if None, assumes the last column. 

    Returns
    -------
        markers : ndarray
            array of markers measurments
    '''

    if marker_col_end is None:
        markers = data.iloc[:, marker_col_start:]
    else:
        markers = data.iloc[:, marker_col_start:marker_col_end+1]

    return markers.to_numpy(dtype=np.float16)

def calculate_pairwise_distances(points):
    return spatial.distance.cdist(points, points)

def calculate_weighted_markers(distance_matrix, markers, sigma=75, verbose=False):
    featurized_data = np.empty(markers.shape)
    if verbose:
        print('Calculating weights...')
    weights = np.exp(-(np.power(distance_matrix, 2)/sigma**2))
    if verbose:
        print('Featurizing data...')
        for i in trange(0, distance_matrix.shape[0]):
            featurized_data[i, :] = np.mean((weights[i, :] * markers.T).T, axis=0)
    else:
        for i in range(0, distance_matrix.shape[0]):
            featurized_data[i, :] = np.mean((weights[i, :] * markers.T).T, axis=0)
    return featurized_data

def featurize_neighborhood(neighborhood_idx, markers, mode=None):

    neighborhood_markers = markers[neighborhood_idx, :]
    neighborhood_mean = np.mean(neighborhood_markers, axis=0)

    return neighborhood_mean

def cluster_descriptors(all_words, vocab_size):
    '''
    Creates a summarized feature for each visual word
    and clusters it with KMeans. 
    
    Parameters
    ----------
        all_words     :     array
            the visual words to cluster
        vocab_size      :     int
            the number of words in the vocabulary
    
    Returns
    -------
        kmeans     :      sklearn.cluster.KMeans object
            the clustering object
        assignments     :     array
            assignments of each visual word to cluster id
    '''

    bs = vocab_size//4
    kmeans = MiniBatchKMeans(n_clusters = vocab_size, batch_size=bs)
    assignments = kmeans.fit_predict(all_words)

    return kmeans, assignments

def assemble_visual_word_data(points, one_samp=True, sample_ids = None):
    if one_samp:
        ids = np.zeros((points.shape[0],1))
        return np.concatenate([ids, points], axis=1)

    else:
        assert len(sample_ids) == points.shape[0]
        return np.concatenate([ids, points], axis=1)
