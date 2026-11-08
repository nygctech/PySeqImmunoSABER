import torch
import numpy as np
import torch.nn as nn 
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001)))
                q.append(("ReLu_%d" % i, nn.ReLU(inplace=True)))
                q.append(("Dropout_%d" % i, nn.Dropout(p=0.1)))
        self.mlp = nn.Sequential(OrderedDict(q))
            
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, shape, latent_size = 17, n_cond = 0):
        super(Encoder, self).__init__()
        r, c = shape
        
        self.encode = nn.Sequential(nn.Linear(c, c+10), 
                                    nn.BatchNorm1d(c + 10,  momentum=0.01, eps=0.001),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(c + 10, c),
                                    nn.BatchNorm1d(c),
                                    nn.Sigmoid())
        
        self.calculate_mean = MLP([c + n_cond, 128, 128, 28, latent_size], last_activation=False)
        self.calculate_logvar = MLP([c + n_cond, 128, 128, 28, latent_size], last_activation=False)
        
    def forward(self, x):
        #x = self.encode(x) #is this needed in our case?
        return self.calculate_mean(x), self.calculate_logvar(x)

class Decoder(nn.Module):
    def __init__(self, shape, latent_size = 17, n_cond = 0):
        super(Decoder, self).__init__()
        r, c = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([latent_size, 28,128,128,c], last_activation=False),nn.Sigmoid())
        
    def forward(self, z):
        r, c = self.shape
        return self.decode(z).view(-1, c)

class VAE(nn.Module):
    def __init__(self, shape, latent_size = 17):
        super(VAE, self).__init__()
        self.dim = latent_size
        self.encoder = Encoder(shape, latent_size)
        self.decoder = Decoder(shape, latent_size)
        
    def sampling(self, mean, logvar):
        #reparameterization trick
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean+eps*sigma
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar
    
    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

BCE_loss = nn.BCELoss(reduction="sum")
def loss(X, X_hat, mean, logvar, slda_loss=None):
    reconstruction_loss = BCE_loss(X_hat, X)
    kl_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    if slda_loss == None:
        return reconstruction_loss + kl_divergence
    else:
        return reconstruction_loss + kl_divergence + slda_loss
    