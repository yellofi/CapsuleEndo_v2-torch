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
    parser.add_argument('--ab_classes', action="store", nargs='+', type=str, 
                        default=['red_spot', 'angioectasia', 'active_bleeding', 'erosion', 'ulcer', 'stricture'], 
                        help='target abnormal classes(ex: red_spot angioectasia active_bleeding ...)')
    
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

    data_config = {'negative': []}
    
    for ab_class in args.ab_classes:
        data_config[ab_class] = []
    
    for name in label.index.values:  
        if label.loc[name]['source'] in sources:
            if label.loc[name]['positive'] == 0 and label.loc[name]['negative'] == 1:
                data_config['negative'].append(name)
            elif label.loc[name][args.ab_classes].sum() == 1 and label.loc[name]['negative'] == 0:
                ab_class = args.ab_classes[list(label.loc[name][args.ab_classes] == 1).index(True)]
                data_config[ab_class].append(name)
 
    classes = ['negative'] + args.ab_classes
    
    for cls in classes:
        print(cls, ':', len(data_config[cls]))
    print()
    
    aug_suffixes = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'preprocessing')
    
    def mapping_id_to_file(id_list):
        files = []
        for name in id_list:
            for aug_suf in aug_suffixes:
                files.append(name.split('.jpg')[0] + '_' + aug_suf + '.jpg')
        return files
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = True, random_state = 44)
    
    for n_cls, cls in enumerate(classes):
        print(cls, end = "\n")
#         print(data_config[cls])
        for i, (train_idx, test_idx) in enumerate(kf.split(data_config[cls])):
            train_id = list(np.asarray(data_config[cls])[train_idx])
            train_id, valid_id = train_valid_split(train_id)
            test_id = list(np.asarray(data_config[cls])[test_idx])
            
            train_files = mapping_id_to_file(train_id)
            valid_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in valid_id]
            test_files = [name.split('.jpg')[0] + '__c_-_-_-.jpg' for name in test_id]
            
            if n_cls == 0:
                data_config['{:02d}_train_aug_files'.format(i+1)] = [train_files]
                data_config['{:02d}_valid_files'.format(i+1)] = [valid_files]
                data_config['{:02d}_test_files'.format(i+1)] = [test_files]
            else:
                data_config['{:02d}_train_aug_files'.format(i+1)].append(train_files)
                data_config['{:02d}_valid_files'.format(i+1)].append(valid_files)
                data_config['{:02d}_test_files'.format(i+1)].append(test_files)
                
            print('{:02d}-fold'.format(i+1))
            print('training set:', len(train_files), 'validation set:', len(valid_files), 'testing set:', len(test_files))
        print()
        
    with open(label_dir + '/{}.pkl'.format(args.save_name), "wb") as f:
        pickle.dump(data_config, f)
        
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 data_config_na1a2a3_cv.py --target_sources p3_2 '190814 negative' --ab_classes red_spot angioectasia active_bleeding erosion ulcer stricture --aug_frb 0 0 0 --save_name data_config_p3_2_negative_nh3d3_---_--_5f_cv
"""

