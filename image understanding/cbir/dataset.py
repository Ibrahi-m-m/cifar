import os
import pickle
import numpy as np

def load_batch(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Batch file not found: {file_path}")
        
    with open(file_path, 'rb') as fo:
        batch_dict = pickle.load(fo, encoding='bytes')
    return batch_dict

def load_dataset(dataset_dir):
    print(f"Loading CIFAR-10 dataset from {dataset_dir}...")
    
    meta_path = os.path.join(dataset_dir, 'batches.meta')
    meta = load_batch(meta_path)
    label_names = [label.decode('utf-8') for label in meta[b'label_names']]
    
    train_images_list = []
    train_labels_list = []
    train_filenames_list = []
    
    for i in range(1, 6):
        batch_path = os.path.join(dataset_dir, f'data_batch_{i}')
        batch = load_batch(batch_path)
        
        raw_images = batch[b'data']
        images = raw_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        
        train_images_list.append(images)
        train_labels_list.append(batch[b'labels'])
        filenames = [f.decode('utf-8') for f in batch[b'filenames']]
        train_filenames_list.extend(filenames)
        
    db_images = np.concatenate(train_images_list, axis=0)
    db_labels = np.concatenate(train_labels_list, axis=0)
    db_filenames = train_filenames_list
    
    test_batch_path = os.path.join(dataset_dir, 'test_batch')
    test_batch = load_batch(test_batch_path)
    
    raw_test_images = test_batch[b'data']
    test_images = raw_test_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch[b'labels'])
    test_filenames = [f.decode('utf-8') for f in test_batch[b'filenames']]
    
    print(f"Dataset loaded. Database size: {len(db_images)} images, Query set size: {len(test_images)} images.")
    return db_images, db_labels, db_filenames, test_images, test_labels, test_filenames, label_names
