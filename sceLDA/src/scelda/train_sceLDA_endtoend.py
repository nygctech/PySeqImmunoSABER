import os, time, tqdm
import numpy as np

import torch
import torch.nn.functional as F

from scipy import spatial
from sklearn.neighbors import NearestNeighbors

import spatialdata_preprocessing
import slda 
from model import VAE, loss
from utils import EarlyStop

# Set up data for PyTorch
test_frac = 0.15
test_size = int(np.round(test_frac * X_downsampled.shape[0]))
data = torch.tensor(X_downsampled.todense(), dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data)
test_data, train_data = torch.utils.data.random_split(dataset, [test_size, data.shape[0]-test_size])

batch_size = 64 # TO DO: make this configurable 
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set up SLDA hyper params
points = spatialdata_preprocessing.make_points_arr(loc_df_downsampled, 'x', 'y')
distance_matrix = spatial.distance.cdist(points, points)

# Constant model params
vocab_size = 100 #best = [75,5,20,50]
n_topics = 5
sigma = 250
n_docs = 200
n_imgs = 1

alpha = torch.ones((n_topics,))/n_topics #uniform priors
beta = torch.ones(vocab_size)/vocab_size

ids = np.zeros((points.shape[0],1))
c = np.concatenate([ids, points], axis=1)

document_locs, docs_in_img = slda.initialize_documents_spatialdata(c, n_docs, n_imgs) #I think this makes sense to fix across training epochs?

#Configuring VAE basics
n_genes = data.shape[1]
net = VAE((1, n_genes), latent_size = 22)
net.to(device)
save_name = "VAE_BatchSize=64_allGenes.pt" #TO DO: make this dynamic
