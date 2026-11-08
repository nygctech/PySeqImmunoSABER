
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
from model import loss, VAE
from utils import EarlyStop

############## loading data ###################

data = np.load('/Users/ethan/Dropbox (MIT)/Lab/lda_implementations/vae_slda/core3_markers_dropna.npy')
test_frac = 0.15
test_size = int(np.round(test_frac * data.shape[0]))
data = torch.tensor(data, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data)
test_data, train_data = torch.utils.data.random_split(dataset, [test_size, data.shape[0]-test_size])

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)

############## loading models ###################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = VAE((1, 52), latent_size = 17)
net.to(device)
print(net)
save_name = "VAE.pt"

############### training #########################

lr = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if not (ans == 'y'):
        checkpoint = torch.load(save_name, map_location = device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr
        
max_epochs = 1000
early_stop = EarlyStop(patience = 20, save_name = save_name)
net = net.to(device)

print("training on ", device)
for epoch in range(max_epochs):

    train_loss, n, start = 0.0, 0, time.time()
    for X, in tqdm.tqdm(train_loader, ncols = 50):
        X = X.to(device)
        X_hat, mean, logvar = net(X)

        l = loss(X, X_hat, mean, logvar).to(device)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        n += X.shape[0]

    train_loss /= n
    print('epoch %d, train loss %.4f , time %.1f sec'
          % (epoch, train_loss, time.time() - start))
    
    adjust_lr(optimizer)
    
    if (early_stop(train_loss, net, optimizer)):
        break

checkpoint = torch.load(early_stop.save_name)
net.load_state_dict(checkpoint["net"])