#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default=[1, 1, 1], help='flip, rotate, blurring control switch')
#     parser.add_argument('--aug_frb', action="store", type=list, 
#                         default=[1, 1, 1], help='flip, rotate, blurring control switch')

    parser.add_argument('--aug_sv', action="store", type=bool, 
                        default=True, help='saturation and value control switch')
    parser.add_argument('--save_name', action="store", type=str, default='data_config', help='file name')
    
    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args

def __main__(args):
    
    data_dir = root + args.data_dir 
    label_dir = root + args.label_dir 
    img_files = sorted(os.listdir(data_dir))
    label = pd.read_csv(label_dir + '/label.csv', index_col = 0)
    
    data_config = {'positive': [], 'negative': []}
    data_config['test_pos_id'] = []
    data_config['test_neg_id'] = []

    for name in label.index.values:  
        if label.loc[name]['source'] == '200121 validation':
            if label.loc[name]['positive'] == 1 and label.loc[name]['negative'] == 0:
                data_config['test_pos_id'].append(name)
            elif label.loc[name]['positive'] == 0 and label.loc[name]['negative'] == 1:
                data_config['test_neg_id'].append(name)
        else:
            if label.loc[name]['positive'] == 1 and label.loc[name]['negative'] == 0:
                data_config['positive'].append(name)
            elif label.loc[name]['positive'] == 0 and label.loc[name]['negative'] == 1:
                data_config['negative'].append(name)
    #     elif label.loc[name]['positive'] == 1 and label.loc[name]['negative'] == 1:
    #         print(label.loc[name])

    data_config['train_pos_id'] = None
    data_config['train_neg_id'] = None
    data_config['valid_pos_id'] = None
    data_config['valid_neg_id'] = None

    data_config['train_pos_id'], data_config['valid_pos_id'] = train_valid_split(data_config['positive'])
    data_config['train_neg_id'], data_config['valid_neg_id'] = train_valid_split(data_config['negative'])

    data_config['train_pos_files'] = []
    data_config['train_neg_files'] = []
    data_config['valid_pos_files'] = []
    data_config['valid_neg_files'] = []
    data_config['test_pos_files'] = []
    data_config['test_neg_files'] = []
  
#     for i, file in enumerate(img_files):

#     printProgress(i+1, len(img_files), prefix = 'data configuration', 
#                   suffix = len(img_files), decimals = 1, barLength = 70)

#     original_id = file.split('__c')[0] + '.jpg'

#     # target augmented train files 
#     if (file.split('c_')[-1])[:-4] in extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'load'): 
#         if original_id in data_config['train_pos_id']:
#             data_config['train_pos_files'].append(file)
#         elif original_id in data_config['train_neg_id']:
#             data_config['train_neg_files'].append(file)

#     # valid files
#     if original_id in data_config['valid_pos_id'] and 'c_-_-_-.jpg' in file:
#         data_config['valid_pos_files'].append(file)
#     elif original_id in data_config['valid_neg_id'] and 'c_-_-_-.jpg' in file:
#         data_config['valid_neg_files'].append(file)

#     # test files
#     if original_id in data_config['test_pos_id'] and 'c_-_-_-.jpg' in file:
#         data_config['test_pos_files'].append(file)
#     elif original_id in data_config['test_neg_id'] and 'c_-_-_-.jpg' in file:
#         data_config['test_neg_files'].append(file)
    
    aug_suffixes = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'preprocessing')
    
    for name in data_config['train_pos_id']:
        for aug_suf in aug_suffixes:
            data_config['train_pos_files'].append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
    
    for name in data_config['train_neg_id']:
        for aug_suf in aug_suffixes:
            data_config['train_neg_files'].append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
    
    data_config['valid_pos_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['valid_pos_id']
    data_config['valid_neg_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['valid_neg_id']                                  
    
    data_config['test_pos_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['test_pos_id']
    data_config['test_neg_files'] = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in data_config['test_neg_id']                                   
                                      
    data_config['train_aug_files'] = [data_config['train_neg_files'], data_config['train_pos_files']]
    data_config['valid_files'] = [data_config['valid_neg_files'], data_config['valid_pos_files']]
    data_config['test_files'] = [data_config['test_neg_files'], data_config['test_pos_files']]

    with open(label_dir + '/{}.pkl'.format(args.save_name), "wb") as f:
        pickle.dump(data_config, f)
     
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 data_config_np_v1.py --data_dir '/1 preprocessed/database' --label_dir '/1 preprocessed' --aug_frb 1 1 1 --aug_sv True
"""

