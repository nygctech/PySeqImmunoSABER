import image_processing
import slda
import numpy as np
np.random.seed(512)

data_dir = '../../test_data/MSRC_v2/Images/'
image_files = image_processing.get_image_paths(data_dir)
np.random.shuffle(image_files)
c, all_words, all_shapes = image_processing.assemble_visual_word_data(image_files)

np.save('../../test_data/MSRC_v2/featurized_msrc_shuffled.npy', all_words)
np.save('../../test_data/MSRC_v2/all_shapes_msrc_shuffled.npy', all_shapes)
np.save('../../test_data/MSRC_v2/c_msrc_shuffled.npy', c)

