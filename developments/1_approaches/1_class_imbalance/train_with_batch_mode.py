#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, pickle
from itertools import product

import sys
sys.path.append('/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/algorithms')
from ce_utils.data import load_image_from_path, GenerateLabel
from ce_utils.preprocessing import extract_aug_suffix
from ce_model.CE_training import CNN_Training

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_config', action="store", type=str, 
                        default='data_config_p3_2_np-hd_---_--_5f_cv.pkl', help='target data configuration')
    
    parser.add_argument('--model_type', action="store", type=str, 
                        default='NP', help='model type')
    parser.add_argument('--model_dir', action="store", type=str, 
                        default='model', help='model saving directory')

    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default=[1, 1, 1], help='flip, rotate, blurring control switch')
    parser.add_argument('--aug_sv', action="store_true", 
                        default=False, help='saturation and value control switch')
    
    parser.add_argument('--input_shape', action="store", nargs='+', type=int, 
                        default=[3, 512, 512], help='input data shape')
    parser.add_argument('--LRs', action="store", nargs='+', type=float,
                        default='0.0001', help='learning rates for Grid search')  
    parser.add_argument('--BSs', action="store", nargs='+', type=int,
                        default='32', help='batch sizes for Grid search')
    parser.add_argument('--n_epoch', action="store", type=int, 
                        default=500, help='total training epoch')

    parser.add_argument('--network', action="store", type=str, 
                        default='CNN_v1', help='network used for training')
    
    parser.add_argument('--training_verbose', action="store", type=int, 
                        default=3, help='network used for training')
    
    parser.add_argument('--gpu_idx', action="store", type=int, 
                        default=3, help='gpu idx used for training')
    
    args = parser.parse_args()
    print('args={}\n'.format(args))
    
    return args

root = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'

def main(args):

    data_dir = root + '/database'

    with open(root + '/'+ args.data_config, "rb") as f:
        data_config = pickle.load(f)
        
    for b_mode in ['mixed', 'equal']:
#     for b_mode in ['equal']:
        model_spec = 1
        for i in range(5):
            print('{:02d}-fold\n'.format(i+1))

            train_aug_files, valid_files = data_config['{:02d}_train_aug_files'.format(i+1)], data_config['{:02d}_valid_files'.format(i+1)]
    #         target_aug = extract_aug_suffix([0, 0, 0], False, mode = 'load')
            target_aug = extract_aug_suffix(args.aug_frb, args.aug_sv, mode = 'load')

            train_aug_paths = []
            for train_aug_file in train_aug_files:
                train_aug_paths.append([os.path.join(data_dir, f) for f in train_aug_file 
                                        if (f.split('c_')[-1])[:-4] in target_aug])

            valid_Xs = []
            for valid_file in valid_files:
                valid_path = [os.path.join(data_dir, f) for f in valid_file]
                valid_Xs.append(load_image_from_path(valid_path))

            LRs = [0.0001]
            BSs = [32]

            Params = list(product(*[LRs, BSs]))
            
            for lr, bs in Params: 
                
                if b_mode == 'equal':
                    bs = int(bs/2)
                
                print('Learning Rate: {}, Batch Size: {}\n'.format(lr, bs))  

#                 CT = CNN_Training(input_shape = [3, 512, 512], lr = lr, n_batch = bs, n_epoch = 500)
                CT = CNN_Training(input_shape = args.input_shape, lr = lr, n_batch = bs, n_epoch = args.n_epoch,
                                  reproducible = True)

#                 CT.Run(train_aug_paths, valid_Xs, sampling_mode = sp_mode, batch_mode = 'mixed', 
#                        network = 'CNN_v1', model_spec = model_spec, verbose = 3, 
#                        model_dir = 'model', model_name = '{:02d}_B_{}'.format(i+1, sp_mode), GPU_idx = 3)   
                
                CT.Run(train_aug_paths, valid_Xs, sampling_mode = None, batch_mode = b_mode, 
                       network = args.network, model_spec = model_spec, verbose = args.training_verbose, 
                       model_dir = args.model_dir, model_name = '{:02d}_{}_no_sample_{}'.format(i+1, args.model_type, b_mode), 
                       GPU_idx = args.gpu_idx)  

            if i != 0: model_spec = 0

            print('')

if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())

            
"""
python3 train_with_batch_mode.py --data_config data_config_p3_2_np-hd_---_--_5f_cv.pkl --model_type np-hd --aug_frb 0 0 0 --input_shape 3 512 512 --LRs 0.0001 --BSs 32 --n_epoch 500 --network 'CNN_v1' --gpu_idx 3

python3 train_with_batch_mode.py --data_config data_config_p3_2_negative_np-hd_---_--_5f_cv.pkl --model_type np-hd_p32_neg --aug_frb 0 0 0 --input_shape 3 512 512 --LRs 0.0001 --BSs 32 --n_epoch 500 --network 'CNN_v1' --gpu_idx 3

"""

