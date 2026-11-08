import image_processing
import slda
import numpy as np
import torch
np.random.seed(512)

all_words = np.load('../../test_data/MSRC_v2/featurized_msrc_shuffled.npy')
all_shapes = np.load('../../test_data/MSRC_v2/all_shapes_msrc_shuffled.npy')
c = np.load('../../test_data/MSRC_v2/c_msrc_shuffled.npy')

vocab_size = 200
n_topics = 15
sigma = 20
n_docs = 40
n_imgs = all_shapes.shape[0]

alpha = torch.ones((n_topics,))/n_topics
beta = torch.ones(vocab_size)/vocab_size

assignments, kmeans = image_processing.cluster_descriptors(all_words, vocab_size)

input_arr = slda.build_input_word_array(assignments, c)
z, docs, n_dk, n_kw, n_d, n_k = slda.initialize_model(input_arr, n_imgs, n_docs, n_topics, vocab_size)

document_locs = slda.initialize_documents(all_shapes, n_docs)

pi_hat, phi_hat, n_dk, n_kw, n_k, n_d, docs, z = slda.train_model(input_arr, document_locs,
                                                     n_imgs, n_docs, n_topics, vocab_size,z, docs, n_dk, n_kw, n_d, n_k, alpha, beta, sigma = sigma, n_epochs = 30)

output = (pi_hat, phi_hat, n_dk, n_kw, n_k, n_d, docs, z )
np.pickle.dump(output, './trained_model_output.pkl')
