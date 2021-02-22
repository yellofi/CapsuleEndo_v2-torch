#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import numpy as np
import pandas as pd
import pickle

from ce_utils.data import train_valid_split
from ce_utils.preprocessing import extract_aug_suffix

root = '/mnt/disk2/data/private_data/SMhospital/capsule'

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='database directory')
    parser.add_argument('--label_dir', action="store", type=str, 
                        default='/1 preprocessed', help='label directory')
    parser.add_argument('--target_sources', action="store", nargs='+', type=str, 
                        default='p3_2', help='target source **default p3_2')
    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default='1 1 1', help='flip, rotate, blurring control switch')
    parser.add_argument('--aug_sv', action="store_true", 
                        default=False, help='saturation and value control switch')
    parser.add_argument('--save_name', action="store", type=str, default='data_config', help='file name')
    
    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args

def __main__(args):
    
    data_dir = root + args.data_dir 
    label_dir = root + args.label_dir 

    img_files = sorted(os.listdir(data_dir))
    label = pd.read_csv(label_dir + '/label.csv', index_col = 0)

    sources = []
    for source in args.target_sources:
        sources.append([source_ for source_ in sorted(set(label.source.values)) if source in source_])

    sources = np.concatenate(sources)

    data_config = {'positive': [], 'negative': []}
    
    for name in label.index.values:  
        if label.loc[name]['source'] in sources:
            if label.loc[name]['positive'] == 1 and label.loc[name]['negative'] == 0:
                data_config['positive'].append(name)
            elif label.loc[name]['positive'] == 0 and label.loc[name]['negative'] == 1:
                data_config['negative'].append(name)

    aug_suffixes = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'preprocessing')
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = True, random_state = 44)

    for i, [(n_train_idx, n_test_idx), (p_train_idx, p_test_idx)] in enumerate(zip(kf.split(data_config['negative']),
                                                                                   kf.split(data_config['positive']))):

        train_neg_id = list(np.asarray(data_config['negative'])[n_train_idx])  
        train_pos_id = list(np.asarray(data_config['positive'])[p_train_idx])

        train_neg_id, valid_neg_id = train_valid_split(train_neg_id)
        train_pos_id, valid_pos_id = train_valid_split(train_pos_id)

        test_neg_id = list(np.asarray(data_config['negative'])[n_test_idx])
        test_pos_id = list(np.asarray(data_config['positive'])[p_test_idx])

        train_pos_files = []
        for name in train_pos_id:
            for aug_suf in aug_suffixes:
                train_pos_files.append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')

        train_neg_files = []
        for name in train_neg_id:
            for aug_suf in aug_suffixes:
                train_neg_files.append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')

        valid_pos_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_pos_id]
        valid_neg_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_neg_id]

        test_pos_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_pos_id]
        test_neg_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_neg_id]

        data_config['{:02d}_train_aug_files'.format(i+1)] = [train_neg_files, train_pos_files]
        data_config['{:02d}_valid_files'.format(i+1)] = [valid_neg_files, valid_pos_files]
        data_config['{:02d}_test_files'.format(i+1)] = [test_neg_files, test_pos_files]
        
        print('{:02d}-fold'.format(i+1))
        print('training set:', len(train_neg_files), len(train_pos_files))
        print('validation set:', len(valid_neg_files), len(valid_pos_files))
        print('testing set:', len(test_neg_files), len(test_pos_files))


    with open(label_dir + '/{}.pkl'.format(args.save_name), "wb") as f:
        pickle.dump(data_config, f)
        
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 data_config_np_cv.py --target_sources p3_2 --aug_frb 0 0 0 --save_name data_config_p3_2_np_---_--_5f_cv
"""

