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
                        default=['p3_2'], help='target source **default p3_2')
#     parser.add_argument('--ab_class', action="store", type=str, 
#                         default='hemorrhagic', help='target abnormal class(ex: hemorrhagic, depressed, ulcer)')
    
    # '+' == 1 or more.
    # '*' == 0 or more.
    # '?' == 0 or 1.
    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default=[0, 0, 0], help='data aug. - flip, rotate, blurring control switch')
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

    sources = np.unique(np.concatenate(sources))
    print(sources)

    data_config = {'negative': [], 'hemorrhagic': [], 'depressed': []}
    
    for name in label.index.values:  
        if label.loc[name]['source'] in sources:
            if label.loc[name]['positive'] == 0 and label.loc[name]['negative'] == 1:
                data_config['negative'].append(name)
            elif (label.loc[name]['hemorrhagic'] == 1 and label.loc[name]['depressed'] == 0 and label.loc[name]['negative'] == 0):
                data_config['hemorrhagic'].append(name)
            elif (label.loc[name]['hemorrhagic'] == 0 and label.loc[name]['depressed'] == 1 and label.loc[name]['negative'] == 0):
                data_config['depressed'].append(name)
            

    aug_suffixes = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'preprocessing')
    
    def mapping_id_to_file(id_list):
        files = []
        for name in id_list:
            for aug_suf in aug_suffixes:
                files.append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
        return files
    
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = True, random_state = 44)

    for i, [(n_train_idx, n_test_idx), (h_train_idx, h_test_idx), (d_train_idx, d_test_idx)] in enumerate(zip(kf.split(data_config['negative']),
                                                                                   kf.split(data_config['hemorrhagic']),
                                                                                   kf.split(data_config['depressed']))):

        train_n_id = list(np.asarray(data_config['negative'])[n_train_idx])  
        train_h_id = list(np.asarray(data_config['hemorrhagic'])[h_train_idx])
        train_d_id = list(np.asarray(data_config['depressed'])[d_train_idx])
                                                                                                         
        train_n_id, valid_n_id = train_valid_split(train_n_id)
        train_h_id, valid_h_id = train_valid_split(train_h_id)
        train_d_id, valid_d_id = train_valid_split(train_d_id)

        test_n_id = list(np.asarray(data_config['negative'])[n_test_idx])  
        test_h_id = list(np.asarray(data_config['hemorrhagic'])[h_test_idx])
        test_d_id = list(np.asarray(data_config['depressed'])[d_test_idx])
 
        train_n_files = mapping_id_to_file(train_n_id)
        train_h_files = mapping_id_to_file(train_h_id)
        train_d_files = mapping_id_to_file(train_d_id)

        valid_n_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_n_id]
        valid_h_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_h_id]
        valid_d_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_d_id]

        test_n_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_n_id]
        test_h_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_h_id]
        test_d_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_d_id]

        data_config['{:02d}_train_aug_files'.format(i+1)] = [train_n_files, train_h_files, train_d_files]
        data_config['{:02d}_valid_files'.format(i+1)] = [valid_n_files, valid_h_files, valid_d_files]
        data_config['{:02d}_test_files'.format(i+1)] = [test_n_files, test_h_files, test_d_files]
        
        print('{:02d}-fold'.format(i+1))
        print('training set:', len(train_n_files), len(train_h_files), len(train_d_files))
        print('validation set:', len(valid_n_files), len(valid_h_files), len(valid_d_files))
        print('testing set:', len(test_n_files), len(test_h_files), len(test_d_files))
        
    with open(label_dir + '/{}.pkl'.format(args.save_name), "wb") as f:
        pickle.dump(data_config, f)
        
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 data_config_nhd_cv.py --target_sources p3_2 190814 negative --aug_frb 0 0 0 --save_name data_config_p3_2_negative_nhd_---_--_5f_cv
"""

