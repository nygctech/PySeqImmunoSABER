import torch
import pandas as pd
import numpy as np
from . import spatialdata_preprocessing

def build_dataset(infile, downsample_size=20000, test_frac=0.15, batch_size=64, x_name='x', y_name='y'):
    df = pd.read_csv(infile)

    if downsample_size < len(df):
        df = df.sample(downsample_size, replace=False)

    loc_df = df.iloc[:, :2] #first two cols should be x,y
    markers_df = df.iloc[:, 2:] #rest should be marker cols.

    test_size = int(np.round(test_frac * markers_df.shape[0]))
    data = torch.tensor(markers_df.todense(), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    test_data, train_data = torch.utils.data.random_split(dataset, [test_size, data.shape[0]-test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)

    points = spatialdata_preprocessing.make_points_arr(loc_df, x_name, y_name)

    return train_loader, test_data, train_data, points

def train_scelda(data, latent_size, lr, epochs, outdir, train_loader, points, vocab_size, n_imgs, n_docs, n_topics, alpha, beta, c, sigma, save_name):
    from model import VAE, loss
    from utils import EarlyStop
    from scipy import spatial
    import torch.nn.functional as F
    import os, time, tqdm
    import slda

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_genes = data.shape[1]
    document_locs, docs_in_img = slda.initialize_documents_spatialdata(c, n_docs, n_imgs)
    distance_matrix = spatial.distance.cdist(points, points)

    #latent_size= 24
    net = VAE((1, n_genes), latent_size = latent_size)
    net.to(device)

    loss_tracker = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

    def adjust_lr(optimizer, decay_rate=0.95):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

    max_epochs = epochs
    slda_loss = 0 # Set the initial SLDA loss.
    early_stop = EarlyStop(patience = 20, save_name = save_name)
    net = net.to(device)

    print("training on ", device)
    for epoch in range(max_epochs):

        train_loss, n, start = 0.0, 0, time.time()
        for i in range(0, 2):
            for X, in tqdm.tqdm(train_loader, ncols = 50):
            #for X, in train_loader:
                X = X.to(device)
                X_hat, mean, logvar = net(X)

                l = loss(X, X_hat, mean, logvar, slda_loss).to(device)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                train_loss += l.cpu().item()
                n += X.shape[0]

            train_loss /= n

        #NOW NEED TO USE THIS PARAMETER SETTING TO DO SLDA...

        #ENCODE DATASET WITH PARTIALLY TRAINED VAE
        decoded_z, mean, logvar = net.forward(data) #embed to latent distrobution
        X_temp = net.sampling(mean, logvar)           #sample within latent distrib

        #USE EMBEDDED DATA TO RUN SLDA

        #Setup here will have to happen every time :/
        featurized_data = spatialdata_preprocessing.calculate_weighted_markers(distance_matrix, X_temp.detach().numpy(), sigma=sigma)
        kmeans, assignments = spatialdata_preprocessing.cluster_descriptors(featurized_data, vocab_size)
        input_arr = slda.build_input_word_array(assignments, c)


        z, docs, n_dk, n_kw, n_d, n_k = slda.initialize_model(input_arr, n_imgs, n_docs, n_topics, vocab_size)


        pi_hat, phi_hat, n_dk, n_kw, n_k, n_d, docs, z, slda_loss = slda.train_model(input_arr, document_locs, docs_in_img, n_imgs, n_docs, n_topics, vocab_size, z, docs, n_dk, n_kw, n_d, n_k, alpha, beta, sigma, n_epochs=10, return_loss=True) #train the SLDA, need to add in a return of the final loss here.


        print('epoch %d, train loss %.4f , time %.1f sec'
            % (epoch, train_loss, time.time() - start))

        loss_tracker.append(train_loss)
        adjust_lr(optimizer)

        if (early_stop(train_loss, net, optimizer)):
            break

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])

    topic_probs, best_topic = slda.get_topic_assignments(phi_hat, pi_hat, document_locs, input_arr, n_docs, c, assignments, sigma, n_topics)
    return loss_tracker, topic_probs, best_topic
