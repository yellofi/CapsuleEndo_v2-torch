#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import numpy as np
import pickle
import cv2 

from ce_utils.preprocessing import extract_aug_suffix

"""for training"""
def train_valid_split(data, valid_rate = 0.8):
    n = int(len(data))
    train_idx = np.sort(np.random.choice(n, round(valid_rate*n), replace = False))
    valid_idx = np.sort(np.setdiff1d(np.arange(n), train_idx))

    if type(data) == list:
        data = np.asarray(data)
    return list(data[train_idx]), list(data[valid_idx])

"""input preprocessing and labeling"""
def Image_norm(img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
    img: np.array (h, w, ch) or (n, h, w, ch) ~ [0, 255] integer
    mean and std vaules were used for normalizing input data to ImageNet models in torch 
    """
    return (img/255-mean)/std

def reshape4torch(img, norm = False):
    """
    (sample #, height, width, channel) -> (sample #, channel, height, width)
    """
    if norm == True:
        img = Image_norm(img)
        
    if len(img.shape) == 4:
        img = np.transpose(img, (0, 3, 1, 2))
        return img
    elif len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)
    
def gen_label(data, cls):
#     label = cls*np.ones([data.shape[0]])
    label = cls*np.ones([len(data)])
    return label

"""load image"""
def load_image_from_path(path_list, normalization = False, extract_name = False):
    data = []
    for i in path_list:
        temp = cv2.imread(i)

        data.append(temp)
    if extract_name != False:
        name = []
        for i in path_list:
            name.append(os.path.basename(i))
        return reshape4torch(np.asarray(data), norm = normalization), np.asarray(name)
    else:
        return reshape4torch(np.asarray(data), norm = normalization)

def load_image_by_label(label_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed', 
                        target_sources = ['200121 validation'], cls = 'positive', 
                        frb_switch = [0, 0, 0], sv_switch = False, extract_name = False,
                        data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed/database'):
    
    import pandas as pd
    import itertools
    from ce_utils.preprocessing import extract_aug_suffix
    
    label = pd.read_csv(label_dir + '/label.csv', index_col = 0)
    sources = sorted(list(set(label['source'].tolist())))

    # for source in sources:print(source)

    target_candidates = ['190520 p3_2', '190814 negative', '200121 validation', 'AI re-screening']
    assert len(np.intersect1d(target_sources, target_candidates)) != 0, 'target sources should be in {}'.format(target_candidates)

    actual_target_sources = []

    for target_source in target_sources:
        if target_source == '190520 p3_2':
            actual_target_sources.append([source for source in sources if target_source in source and 'negative' not in source])
        elif target_source == 'AI re-screening':
            actual_target_sources.append([source for source in sources if target_source in source and 'p3_2' not in source])
        elif target_source == '190814 negative':
            actual_target_sources.append([source for source in sources if target_source in source and 'p3_2' not in source])
        else:
            actual_target_sources.append([source for source in sources if target_source in source])

    actual_target_sources = list(itertools.chain(*actual_target_sources))

    # target = pd.DataFrame(columns = label_.columns)

    for i, source in enumerate(actual_target_sources):
        if i == 0:
            target = label[label['source'] == source]
        else:
            target = target.append(label[label['source'] == source])

    target_names = target[target[cls] == 1].index.tolist()

    aug_suffixes =  extract_aug_suffix(frb_switch = frb_switch, sv_switch = sv_switch, mode = 'preprocessing')

    aug_target_paths = []
    for name in target_names:
        for aug_suf in aug_suffixes:
            aug_target_paths.append(data_dir + '/' + name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
        
    if extract_name != False:
        imgs, names = load_image_from_path(aug_target_paths, extract_name = True)
        return imgs, names
    elif extract_name == False:
        imgs = load_image_from_path(aug_target_paths, extract_name = False)
        return imgs
    

def remove_annotation_mark(files):
    for c in range(len(files)):
        for i in range(len(files[c])):
            if 'annotation' in files[c][i]:
                files[c][i] = ''.join(files[c][i].split('_annotation'))
    return files
    
"""training phase load train aug paths and valid images""" 
def train_data_load(data_config, aug_frb = [0, 0, 0], aug_sv = False):
    """
    data_config: ~~~.pkl
    """
    root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'

    data_dir = root + '/database'

    with open(root + '/{}'.format(data_config), "rb") as f:
        data_config = pickle.load(f)

    train_aug_files, valid_files = data_config['train_aug_files'], data_config['valid_files']
    
#     train_aug_files = remove_annotation_mark(train_aug_files)
#     valid_files = remove_annotation_mark(valid_files)
    
    valid_Xs = []
    for i, valid_file in enumerate(valid_files):
        valid_path = [os.path.join(data_dir, f) for f in valid_file]
        valid_Xs.append(load_image_from_path(valid_path))

    target_aug = extract_aug_suffix(aug_frb, aug_sv, mode = 'load')

    train_aug_paths = []
    for train_aug_file in train_aug_files:
        train_aug_paths.append([os.path.join(data_dir, f) for f in train_aug_file 
                                if (f.split('c_')[-1])[:-4] in target_aug])
        
    print('{:<7}| {:<30}| {:<25}| {:<15}'.format('class', 'total augmented training set', 'target training set (x{})'.format(len(target_aug)), 'validation set'))
    for i in range(len(train_aug_files)):
        print('{:<7}| {:<30}| {:<25}| {:<15}'.format(i, len(train_aug_files[i]), len(train_aug_paths[i]), len(valid_Xs[i])))
        
    print()

#     print('total augmented training set:', len(train_aug_files[0]), ',', len(train_aug_files[1]))
#     print('target augmented training set:', len(train_aug_paths[0]), ',', len(train_aug_paths[1]))
#     print('validation set:', len(valid_files[0]), ',', len(valid_files[1]))
    
    return train_aug_paths, valid_Xs
    
def cv_train_data_load(data_config, aug_frb = [0, 0, 0], aug_sv = False, n_fold = 0):
    """
    data_config: ~~~.pkl
    """
    root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'

    data_dir = root + '/database'

    with open(root + '/{}'.format(data_config), "rb") as f:
        data_config = pickle.load(f)

    train_aug_files, valid_files = data_config['{:02d}_train_aug_files'.format(n_fold)], data_config['{:02d}_valid_files'.format(n_fold)]
#     train_aug_files = remove_annotation_mark(train_aug_files)
#     valid_files = remove_annotation_mark(valid_files)
    
    valid_Xs = []
    for i, valid_file in enumerate(valid_files):
        valid_path = [os.path.join(data_dir, f) for f in valid_file]
        valid_Xs.append(load_image_from_path(valid_path))

    target_aug = extract_aug_suffix(aug_frb, aug_sv, mode = 'load')

    train_aug_paths = []
    for train_aug_file in train_aug_files:
        train_aug_paths.append([os.path.join(data_dir, f) for f in train_aug_file 
                                if (f.split('c_')[-1])[:-4] in target_aug])

    print('{:<7}| {:<30}| {:<20}| {:<15}'.format('class', 'total augmented training set', 'target training set', 'validation set'))
    for i in range(len(train_aug_files)):
        print('{:<7}| {:<30}| {:<20}| {:<15}'.format(i, len(train_aug_files[i]), len(train_aug_paths[i]), len(valid_Xs[i])))
    print()
    
    return train_aug_paths, valid_Xs
    
"""plot image"""
def imshow(inp, title=None):
    """Imshow for torch tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
    
#     mean = np.array([0.5, 0.5, 0.5])
#     std = np.array([0.5, 0.5, 0.5])
    
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
    plt.imshow(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
    
"""query for model training and inference"""
def Batch_idxs(data, batch_size = 250):
    """generate the serial batch of data on index-level.
       Usually, the data is too large to be evaluated at once.
    
    Args:
      data: A list or array of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch_idxs: A list, 
    """
    total_size = len(data)
    batch_idxs = []
    start = 0
    while True:
        if total_size >= start + batch_size:
            batch_idxs.append([start + i for i in range(batch_size)])
        elif total_size < start + batch_size:
            batch_idxs.append([start + i for i in range(total_size - start)])
        start += batch_size
        if total_size <= start:
            break
    return batch_idxs

def batch_idxs(dataset, batch_size = 32, shuffle = False):

    idxs = np.arange(len(dataset))
    total_size = len(idxs)
    if shuffle:
        np.random.shuffle(idxs)
    start = 0
    b_idxs = []
    while True:
        if total_size > start + batch_size: 
            b_idxs.append(list(idxs[start:start+batch_size]))  
            start += batch_size
        elif total_size <= start + batch_size: 
            b_idxs.append(list(idxs[start:]))
            break 
    return b_idxs

"""testing phase""" 
"""load test data"""
def test_data_load(filename):
    
    root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'
    data_dir = root + '/database'
    
    with open(root + '/' + filename, "rb") as f:
        data_config = pickle.load(f)

    test_files = data_config['test_files']

    test_names, test_Xs, test_Ys = [], [], []
    for i, test_file in enumerate(test_files):
        test_path = [os.path.join(data_dir, f) for f in test_file]
        print(len(test_path))
        test_Ys.append(gen_label(test_path, int(i)))

        imgs, filenames = load_image_from_path(test_path, extract_name = True)
        test_Xs.append(imgs)
        test_names.append(filenames)
        
    test_name = np.concatenate(test_names)
    test_X = np.concatenate(test_Xs)
    test_Y = np.concatenate(test_Ys)
    
    return test_name, test_X, test_Y

def cv_test_data_load(filename, n_fold = 1):
    
    root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'
    data_dir = root + '/database'
    
    with open(root + '/' + filename, "rb") as f:
        data_config = pickle.load(f)

    test_files = data_config['{:02d}_test_files'.format(n_fold)]

    test_names, test_Xs, test_Ys = [], [], []
    for i, test_file in enumerate(test_files):
        test_path = [os.path.join(data_dir, f) for f in test_file]
        print('class', i, ':', len(test_path))
        test_Ys.append(gen_label(test_path, int(i)))

        imgs, filenames = load_image_from_path(test_path, extract_name = True)
        test_Xs.append(imgs)
        test_names.append(filenames)
        
    test_name = np.concatenate(test_names)
    test_X = np.concatenate(test_Xs)
    test_Y = np.concatenate(test_Ys)
    
    return test_name, test_X, test_Y

def train_image_load(filename, aug_frb = [0, 0, 0], aug_sv = False):
    
    root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'
    data_dir = root + '/database'
    
    with open(root + '/' + filename, "rb") as f:
        data_config = pickle.load(f)

    train_aug_files = data_config['train_aug_files']
    
#     train_aug_files = remove_annotation_mark(train_aug_files)
    
    target_aug = extract_aug_suffix(aug_frb, aug_sv, mode = 'load')
    
    train_aug_paths = []
    for train_aug_file in train_aug_files:
        train_aug_paths.append([os.path.join(data_dir, f) for f in train_aug_file 
                                if (f.split('c_')[-1])[:-4] in target_aug])

    train_names, train_Xs, train_Ys = [], [], []
    
    for i, train_path in enumerate(train_aug_paths):
#     for i, train_file in enumerate(train_files):
#         train_path = [os.path.join(data_dir, f) for f in train_file]
        print(len(train_path))
        train_Ys.append(gen_label(train_path, int(i)))
        imgs, filenames = load_image_from_path(train_path, extract_name = True)
        train_Xs.append(imgs)
        train_names.append(filenames)
        
    train_name = np.concatenate(train_names)
    train_X = np.concatenate(train_Xs)
    train_Y = np.concatenate(train_Ys)
    
    print()
    
    return train_name, train_X, train_Y