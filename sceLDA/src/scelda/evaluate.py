import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compare_covar_matrix(X, X_hat):
    norm = torch.linalg.norm(cov(X) - cov(X_hat), ord='fro')
    return norm

def cov(X):
    '''
    Calculate the covariance matrix. 
    '''
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)

def corr(X):
    '''
    Calculate the marker/marker correlation matrix and
    return as torch tensor. 
    '''
    corr = np.corrcoef(X.cpu().detach().numpy(), rowvar=False)
    return torch.tensor(corr).to(device)
