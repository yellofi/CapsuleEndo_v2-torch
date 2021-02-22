#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, pickle
from itertools import product

import sys
sys.path.append('/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/algorithms')
from ce_utils.data import train_data_load
from ce_model.training import cnn_training

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_config', action="store", type=str, 
                        default='data_config_p3_2_np-hd_---_--_5f_cv.pkl', help='target data configuration')
    
    parser.add_argument('--model_type', action="store", type=str, 
                        default='NP', help='model type')
    parser.add_argument('--save_path', action="store", type=str, 
                        default='', help='model saving directory')

    parser.add_argument('--aug_frb', action="store", nargs='+', type=int,
                        default=[1, 1, 1], help='flip, rotate, blurring control switch')
    parser.add_argument('--aug_sv', action="store_true", 
                        default=False, help='saturation and value control switch')
    
    parser.add_argument('--input_shape', action="store", nargs='+', type=int, 
                        default=[3, 512, 512], help='input data shape')
    parser.add_argument('--LRs', action="store", nargs='+', type=float,
                        default=[0.0001], help='learning rates for Grid search')  
    parser.add_argument('--BSs', action="store", nargs='+', type=int,
                        default=[32], help='batch sizes for Grid search')
    parser.add_argument('--n_epoch', action="store", type=int, 
                        default=500, help='total training epoch')
    
    parser.add_argument('--network', action="store", type=str, 
                        default='CNN_v1', help='network used for training')
    
    parser.add_argument('--model_file', action="store", type=str, 
                        default=None, help='trained model file path ')
    
    parser.add_argument('--gpu_idx', action="store", type=int, 
                        default=3, help='gpu idx used for training')
    
    parser.add_argument('--training_verbose', action="store", type=int, 
                        default=1, help='network used for training')
    
    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))
    
    return args


def main(args):
    
    
#     data_config = 'data_config_np_frb_sv_add_200713_2.pkl'
    train_aug_paths, valid_Xs = train_data_load(args.data_config, args.aug_frb, args.aug_sv)

    Params = list(product(*[args.LRs, args.BSs]))

    model_spec = 1

    for i, (lr, bs) in enumerate(Params):

        print('')
        print('Learning Rate: {}, Batch Size: {}\n'.format(lr, bs))  
        
        ct = cnn_training(input_shape = args.input_shape, lr = lr, n_batch = bs, n_epoch = args.n_epoch,
                              normalization = False, reproducible = False)

        ct.run(train_aug_paths, valid_Xs, sampling_mode = None, batch_mode = 'equal', 
               network = args.network, model_spec = model_spec, verbose = args.training_verbose, 
               save_path = args.save_path, model_name = args.model_type, gpu_idx = args.gpu_idx,
               classifier_file = args.model_file)  

        if i != 0: model_spec = 0

if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())

            
"""
cd /home/project
source pytorch/bin/activate

cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 train.py --data_config data_config_np_frb_sv_add_200713_2.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/3_add_200713_2 --LRs 0.00001 0.0001 0.001 --BSs 32 --gpu_idx 6 --training_verbose 1

python3 train.py --data_config data_config_np_frb_sv_add_200917.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/4_add_200917 --LRs 0.00001 0.0005 0.0001 --BSs 16 --n_epoch 300 --gpu_idx 3 --training_verbose 1

python3 train.py --data_config data_config_np_frb_sv_add_200713_2.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/3_add_200713_2/re-learning_200713_1_model --LRs 0.0001 --BSs 32 --n_epoch 300 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/2_add_200713_1/model/np_binary_0.0001_8_2009101037_199_t_accr_0.9819_t_loss_0.331068_v_accr_0.9400_v_loss_0.373449.pt' --gpu_idx 2

python3 train.py --data_config data_config_np_frb_sv_add_200917.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/4_add_200917 --LRs 0.0001 --BSs 16 --n_epoch 500 --gpu_idx 3 --training_verbose 1 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/3_add_200713_2/re-learning_200713_1_model/model/np_binary_0.0001_32_2010141023_117_t_accr_0.9772_t_loss_0.335948_v_accr_0.9340_v_loss_0.377869.pt'

python3 train.py --data_config data_config_np_frb_sv_add_200713_2.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/3_add_200713_2/re-learning_200713_1_model --LRs 0.0001 --BSs 16 --n_epoch 200 --gpu_idx 3 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/3_add_200713_2/re-learning_200713_1_model/model/np_binary_0.0001_32_2010141023_117_t_accr_0.9772_t_loss_0.335948_v_accr_0.9340_v_loss_0.377869.pt' 

python3 train.py --data_config data_config_np_---_--_only_200713_1.pkl --model_type np_200713_1 --aug_frb 0 0 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/5_only_update --LRs 0.0001 --BSs 16 --n_epoch 500 --gpu_idx 3 --training_verbose 1 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/1_detection_localization/2_data_aug/model/B_-r-_0.0001_8_2007161324_046_t_accr_0.9800_t_loss_0.333166_v_accr_0.9740_v_loss_0.339054.pt'

python3 train.py --data_config data_config_np_---_--_only_200713_2.pkl --model_type np_200713_2 --aug_frb 0 0 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/5_only_update --LRs 0.0001 --BSs 16 --n_epoch 500 --gpu_idx 3 --training_verbose 1 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/5_only_update/model/np_200713_1_0.0001_16_2010211757_431_t_accr_0.9811_t_loss_0.332176_v_accr_0.8909_v_loss_0.423377.pt'

python3 train.py --data_config data_config_np_---_--_only_200917.pkl --model_type np_200917 --aug_frb 0 0 0 --save_path /mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/5_only_update --LRs 0.0001 --BSs 16 --n_epoch 500 --gpu_idx 3 --training_verbose 1 --model_file '/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/model_developlment/2_AI_feedback/5_only_update/model/np_200713_2_0.0001_16_2010221423_203_t_accr_0.9884_t_loss_0.324911_v_accr_0.9405_v_loss_0.371670.pt'

python3 train.py --data_config data_config_np_frb_sv_add_200917.pkl --model_type np_binary --aug_frb 0 1 0 --save_path /mnt/disk1/project/SMhospital/capsule/torch/2_model_development/2_AI_feedback/4_add_200917 --LRs 0.0001 --BSs 16 --n_epoch 300 --gpu_idx 3 --training_verbose 1 --model_file /mnt/disk1/project/SMhospital/capsule/torch/2_model_development/2_AI_feedback/3_add_200713_2/model/np_binary_0.0001_32_2010141023_117_t_accr_0.9772_t_loss_0.335948_v_accr_0.9340_v_loss_0.377869.pt


python3 train.py --data_config data_config_total_np_-r-.pkl --model_type np_total --aug_frb 0 1 0 --save_path /mnt/disk1/project/SMhospital/capsule/torch/2_model_development/2_AI_feedback/5_total --LRs 0.0001 --BSs 16 --n_epoch 500 --gpu_idx 3 --training_verbose 1


"""

