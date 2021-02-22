#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, glob
import numpy as np
import pandas as pd
import pickle
import itertools

from ce_utils.data import train_valid_split
from ce_utils.preprocessing import extract_aug_suffix
from ce_utils.record import progress_bar

root = '/mnt/disk2/data/private_data/SMhospital/capsule'

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='database directory')
    parser.add_argument('--label_dir', action="store", type=str, 
                        default='/1 preprocessed', help='label directory')
    parser.add_argument('--target_sources', action="store", nargs='+', type=str, 
                        default=['200917'], 
                        help='target sources(ex: 200713-1 200713-2 200917 ...)')
    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default=[1, 1, 1], help='flip, rotate, blurring control switch')
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
    sources = sorted(list(set(label['source'].tolist())))
    
    data_config = {'positive': [], 'negative': []}
    data_config['test_pos_id'] = []
    data_config['test_neg_id'] = []
    
    target_sources = ['200121 validation']
    
    actual_target_sources = []

    for target_source in args.target_sources:

        actual_target_sources.append([source for source in sources 
                                      if target_source in source and '190520' not in source and '190814' not in source])

    actual_target_sources = list(itertools.chain(*actual_target_sources))
    
    for i, source in enumerate(actual_target_sources):
        if i == 0:
            target = label[label['source'] == source]
        else:
            target = target.append(label[label['source'] == source])
    
    test = label[label['source'] == '200121 validation']
    
    for name in target.index.values:
        if target.loc[name]['positive'] == 1 and target.loc[name]['negative'] == 0:
            data_config['positive'].append(name)
        elif target.loc[name]['positive'] == 0 and target.loc[name]['negative'] == 1:
            data_config['negative'].append(name)
            
    for name in test.index.values:
        if test.loc[name]['positive'] == 1 and test.loc[name]['negative'] == 0:
            data_config['test_pos_id'].append(name)
        elif test.loc[name]['positive'] == 0 and test.loc[name]['negative'] == 1:
            data_config['test_neg_id'].append(name)

    data_config['train_pos_id'] = None
    data_config['train_neg_id'] = None
    data_config['valid_pos_id'] = None
    data_config['valid_neg_id'] = None

    data_config['train_pos_id'], data_config['valid_pos_id'] = train_valid_split(data_config['positive'])
    data_config['train_neg_id'], data_config['valid_neg_id'] = train_valid_split(data_config['negative'])

    data_config['train_pos_files'] = []
    data_config['train_neg_files'] = []
    
    aug_suffixes = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'preprocessing')
    
    for name in data_config['train_pos_id']:
        for aug_suf in aug_suffixes:
            data_config['train_pos_files'].append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
 
    for name in data_config['train_neg_id']:
        for aug_suf in aug_suffixes:
            data_config['train_neg_files'].append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
    
    data_config['valid_pos_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['valid_pos_id']]
    data_config['valid_neg_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['valid_neg_id']]      
    
    data_config['test_pos_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['test_pos_id']]
    data_config['test_neg_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['test_neg_id']]                                       
    data_config['train_aug_files'] = [data_config['train_neg_files'], data_config['train_pos_files']]
    data_config['valid_files'] = [data_config['valid_neg_files'], data_config['valid_pos_files']]
    data_config['test_files'] = [data_config['test_neg_files'], data_config['test_pos_files']]
    
    print('training set:', len(data_config['train_aug_files'][0]), len(data_config['train_aug_files'][1]))
    print('validation set:',len(data_config['valid_files'][0]), len(data_config['valid_files'][1]))
    print('testing set:',len(data_config['test_files'][0]), len(data_config['test_files'][1]))
    
    with open(label_dir + '/{}.pkl'.format(args.save_name), "wb") as f:
        pickle.dump(data_config, f)
     
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/algorithms

python3 'data configuration(AI feedback)_OnlyUpdate.py' --target 200917 --aug_frb 0 0 0 --save_name data_config_np_---_--_only_200917

python3 'data configuration(AI feedback)_OnlyUpdate.py' --target '200713-1' --aug_frb 0 0 0 --save_name data_config_np_---_--_only_200713_1

python3 'data configuration(AI feedback)_OnlyUpdate.py' --target '200713-2' --aug_frb 0 0 0 --save_name data_config_np_---_--_only_200713_2

"""

