import os
import pickle
import imageio
import numpy as np

def unpickle(file):
    """Load pickled data from a file."""
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data

def load_validation_data(data_folder, mean_image, img_size=32):
    """Load and preprocess validation data."""
    # Construct the full path to the validation data file
    test_file = os.path.join(data_folder, 'val_data')

    # Load data
    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    # Normalize pixel values
    x = x / np.float32(255)

    # Adjust labels to start from 0
    y = np.array([i-1 for i in y])

    # Subtract the mean image
    x -= mean_image

    # Rearrange data into the correct format
    img_size2 = img_size ** 2
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return {'X_test': x, 'Y_test': y.astype('int64')}

def load_data_batch(data_folder, idx, img_size=64):
    """Load and preprocess a batch of training data."""
    # Construct the file name for the batch
    data_file = os.path.join(data_folder, f'train_data_batch_{idx}')

    # Load data
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    # Normalize pixel values and the mean image
    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

    # Adjust labels to start from 0
    y = [i-1 for i in y]

    # Subtract the mean image
    x -= mean_image

    # Rearrange data into the correct format
    img_size2 = img_size ** 2
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # Duplicate data with horizontal flipping for augmentation
    X_train = x[:len(y), :, :, :]
    Y_train = y
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return {'X_train': X_train, 'Y_train': Y_train.astype('int64'), 'mean': mean_image}

def get_images(file, img_size=64):
    """Load images and labels from a file."""
    d = unpickle(file)
    x = d['data']

    # Rearrange data into the correct format
    img_size2 = img_size ** 2
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    return x, d['labels']

# Define paths and mappings
mapping_file = 'labelmapping.txt'
index_to_synset = {}

# Load label mappings
with open(mapping_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        index_to_synset[int(parts[1])] = parts[0]

# Define base paths for data
base_train_path = 'data/train'
base_val_path = 'data/val'

# Create directories for training and validation data
for path in [base_train_path, base_val_path]:
    os.makedirs(path, exist_ok=True)
    for synset_id in index_to_synset.values():
        os.makedirs(os.path.join(path, synset_id), exist_ok=True)

# Process validation data
data_folder = 'downloadeddata/val'
test_file = os.path.join(data_folder, 'val_data')
val_data, labels = get_images(test_file, img_size=64)

# Save validation images
for i, image in enumerate(val_data):
    synset_id = index_to_synset[labels[i]]
    save_path = os.path.join(base_val_path, synset_id, f'{i:08d}.png')
    imageio.imwrite(save_path, image)

# Process and save training data
data_folder = 'downloadeddata/train'
for i in range(1, 11):
    batch_file = os.path.join(data_folder, f'train_data_batch_{i}')
    train_data, labels = get_images(batch_file, img_size=64)
    for j, image in enumerate(train_data):
        synset_id = index_to_synset[labels[j]]
        save_path = os.path.join(base_train_path, synset_id, f'{j:08d}.png')
        imageio.imwrite(save_path, image)
