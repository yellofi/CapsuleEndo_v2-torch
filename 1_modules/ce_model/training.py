#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from CE_CNNs import CNN_v1
# from CE_utils import printProgress, Generate_folder
# from CE_utils import Reshape4torch, GenerateLabel, Batch_idxs

import sys
sys.path.append('/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/algorithms')
from ce_utils.record import progress_bar, sec_to_m_s_ms
from ce_utils.data import reshape4torch, gen_label, batch_idxs

import numpy as np
import matplotlib.pyplot as plt
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import time, datetime
from tqdm import tqdm

import torch
from torch import nn
from torchsummary import summary

def split2tv(data, label, rate_t_v = 0.9):
    data_num = len(data)
    train_idx = np.random.choice(data_num, int(rate_t_v*data_num), replace = False)
    valid_idx = np.setdiff1d(np.arange(data_num), train_idx)
    return data[train_idx], label[train_idx], data[valid_idx], label[valid_idx]

def data_sampling(train_paths, mode):
    if mode == 'undersample':
        train_paths_ = train_paths.copy()
        n_min = np.min([len(train_paths_[0]), len(train_paths_[1])])
        target_cls = np.argmax([len(train_paths_[0]), len(train_paths_[1])])
        target_path = np.asarray(train_paths_[target_cls])
        undersampled_paths = list(target_path[sorted(np.random.choice(len(target_path), n_min, replace=False))])
        train_paths_[target_cls] = undersampled_paths

    elif mode == 'oversample':
        train_paths_ = train_paths.copy()
        n_max = np.max([len(train_paths_[0]), len(train_paths_[1])])
        target_cls = np.argmin([len(train_paths_[0]), len(train_paths_[1])])
        target_path = np.asarray(train_paths_[target_cls])
        n_diff = int(n_max-len(target_path))
        if len(target_path) >= n_diff:
            oversampled_paths = list(target_path[sorted(np.random.choice(len(target_path), n_diff, replace=False))])
        elif len(target_path) < n_diff:
            oversampled_paths = list(target_path[sorted(np.random.choice(len(target_path), n_diff, replace=True))])
        train_paths_[target_cls] += oversampled_paths

    return train_paths_

def load_rand_batch(path, label = None, cls = None, batch_size = 50, mode = 'eqaul', norm = False):
    idx = np.random.choice(len(path), batch_size)
    if type(path) == list:
        path = np.asarray(path)
    batch_dir = path[idx]
    batch_x = []
    for i in batch_dir:
        img = cv2.imread(i) # BGR Channel
        batch_x.append(img)
    if mode == 'equal':
        batch_label = gen_label(batch_x, cls)
        return reshape4torch(np.asarray(batch_x), norm = norm), batch_label
    elif mode == 'mixed':
        return reshape4torch(np.asarray(batch_x), norm = norm), label[idx]

def rand_shuffle(x1, x2):
    """
    random shuffle of two paired data -> x, y = shuffle(x, y)
    but, available of one data -> x = shuffle(x, None)
    """
    idx = np.arange(len(x1))
    np.random.shuffle(idx)
    if type(x1) == type(x2):
        return x1[idx], x2[idx] 
    else:
        return x1[idx]

class cnn_training:
    def __init__(self, input_shape, 
                 lr, n_batch, n_epoch, 
                 n_patient = None, 
                 normalization = False,
                 reproducible = False):
        self.input_shape = input_shape
        self.lr, self.n_batch, self.n_epoch, self.n_patient = lr, n_batch, n_epoch, n_patient
        self.norm = normalization
        self.reproducible = reproducible
        
        if self.n_patient == None:
            self.n_patient = self.n_epoch
        
        if self.reproducible == True:
            seed = 3
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def get_device(self, gpu_idx = 3):
#         self.device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() else "cpu")
#         print('')
        if torch.cuda.is_available() and type(gpu_idx) == int:
            self.device = torch.device("cuda:{}".format(gpu_idx))
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device), '\n')
        else:
            self.device = torch.device('cpu')
            print("Device: CPU\n")
            
    def define_model_opt(self, network, summary_show = True):
        
        n_ch, input_h, input_w = self.input_shape
        
        if network == 'CNN_v1':
            from ce_model.cnns import CNN_v1
            network = CNN_v1(n_ch, self.n_cls)
        
        self.model = network
        # model = model.cuda()
        self.model = self.model.to(self.device)
        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = True
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        if summary_show == True:
            summary(self.model, (n_ch, input_h, input_w), device = self.device)
        
    def torch_batch_load(self, train_paths, batch_size = 100, shuffle = False, mode = 'equal'):
        x, y = [], []
        
        if mode == 'equal':
            for i, X_i in enumerate(train_paths):
                x_i, y_i = load_rand_batch(path = X_i, cls = i, 
                                           batch_size = batch_size, mode = 'equal', norm = self.norm)
                x.append(x_i), y.append(y_i)
            x, y = np.concatenate(x), np.concatenate(y)
            
        
        elif mode == 'mixed':
            Y = []
            for i, X_i in enumerate(train_paths):
                Y_i = gen_label(X_i, i)
                Y.append(Y_i)
            X = np.concatenate(train_paths)
            Y = np.concatenate(Y)
            x, y = load_rand_batch(path = X, label = Y, 
                                   batch_size = batch_size, mode = 'mixed', norm = self.norm)

        if shuffle != False:
            x, y = rand_shuffle(x, y)
        x, y = torch.tensor(x, device = self.device).float(), torch.tensor(y, device = self.device).long()
        return x, y

    def validation(self, X, Y, batch_size = 32):
        b_idxs = batch_idxs(X, batch_size)
        output = []
        for b_idx in b_idxs:
            x = torch.tensor(X[b_idx, :, :, :], device = self.device).float() 
#             x = X[batch, :, :, :] 
            o = self.model(x)
            output.append(o)
        output = torch.cat(output)
        loss = self.criterion(output, Y)

        _, pred = torch.max(output, 1)
        return loss, pred         
    
    def training_process(self, sampling_mode = None, batch_mode = 'equal', verbose = 3, model_name = 'NP'):
        
        if sampling_mode == 'oversample':
            if batch_mode == 'equal':
                self.max_iter = np.max(self.class_size) // self.n_batch + 1
            elif batch_mode == 'mixed':
                self.max_iter = np.max(self.class_size) // int(self.n_batch/self.n_cls) + 1
        elif sampling_mode == 'undersample':
            if batch_mode == 'equal':
                self.max_iter = np.min(self.class_size) // self.n_batch + 1
            elif batch_mode == 'mixed':
                self.max_iter = np.min(self.class_size) // int(self.n_batch/self.n_cls) + 1
        elif sampling_mode == None:
            if batch_mode == 'equal':
                self.max_iter = np.max(self.class_size) // self.n_batch + 1
            elif batch_mode == 'mixed':
                self.max_iter = np.sum(self.class_size) // self.n_batch + 1

#         print('{} batches per 1 epoch\n'.format(self.max_iter))
        
        if verbose == 3:
            pbar = tqdm(total=self.n_epoch, unit='epoch', bar_format='{l_bar}{bar:40}{r_bar}')
        
        self.loss_hist, self.accr_hist = [], []
        self.val_loss_hist, self.val_accr_hist = [], []
        
        self.iter_i = 0
        self.epoch_i = 0
        self.patient_i = 0
        
        save_path = self.save_path + '/model'
        os.makedirs(save_path, exist_ok = True)  
        
        print('Iteration {} for 1 epoch\n'.format(self.max_iter))
        
        start_time = time.time()
        
        while True:
            
            if self.iter_i == 0:
                train_loss = 0
                train_correct = 0
                
                if sampling_mode:
                    sampled_train_paths = data_sampling(self.train_paths, mode = sampling_mode)
                    
            if sampling_mode:
                train_x, train_y = self.torch_batch_load(sampled_train_paths, self.n_batch, mode = batch_mode, shuffle = True)
            else:
                train_x, train_y = self.torch_batch_load(self.train_paths, self.n_batch, mode = batch_mode, shuffle = True)
            
            
            
            output = self.model(train_x)
            loss = self.criterion(output, train_y)
            
            _, pred = torch.max(output, 1)
            
            train_loss += loss.item()
            train_correct += torch.mean((pred == train_y.detach()).float()).item()
          
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
            self.iter_i += 1
            
            if verbose == 1:
                progress_bar(self.iter_i, self.max_iter, prefix = 'Epoch {:03d}'.format(self.epoch_i+1), 
                              suffix = '', barLength = 70)

            if self.iter_i % self.max_iter == 0:   
                
                self.epoch_i += 1
                self.patient_i += 1
                
                self.loss_hist.append(train_loss / self.iter_i)
                self.accr_hist.append(train_correct / self.iter_i)
                
                self.model.eval()
                with torch.no_grad():
        
                    valid_loss, valid_pred = self.validation(self.valid_X, self.valid_Y, batch_size = 8)
                    
                    self.val_loss_hist.append(valid_loss.item())
                    self.val_accr_hist.append((torch.mean((valid_pred == self.valid_Y.detach()).float()).item()))
                    
                self.model.train()
                    
                
                if (self.val_accr_hist[-1] == np.max(self.val_accr_hist)): 

                    self.patient_i = 0

                    now = datetime.datetime.now()
                    nowDatetime = now.strftime('%y%m%d%H%M')
                    hyper_params = '{}_{}'.format(self.lr, self.n_batch)
                    tr_spec = 't_accr_{:.4f}_t_loss_{:.6f}'.format(self.accr_hist[-1], self.loss_hist[-1])
                    vl_spec = 'v_accr_{:.4f}_v_loss_{:.6f}'.format(self.val_accr_hist[-1], self.val_loss_hist[-1])
                    model_full_name = '{}_{}_{}_{:03d}_{}_{}.pt'.format(model_name, 
                                                                        hyper_params, nowDatetime, self.epoch_i, tr_spec, vl_spec)
                    torch.save(self.model.state_dict(), save_path + '/' + model_full_name)
                    
                if verbose == 1:
                    train_prt = 'train_loss: {:.5f}, train_accr: {:.3f}'.format(self.loss_hist[-1], self.accr_hist[-1])
                    val_prt = 'val_loss: {:.5f}, val_accr: {:.3f}'.format(self.val_loss_hist[-1], self.val_accr_hist[-1])
                    
                    elapsed_time = time.time() - start_time
                    print("{} | {} | {} elapsed".format(train_prt, val_prt, sec_to_m_s_ms(elapsed_time)))
                    
                if verbose == 2:
                    progress_bar(self.epoch_i, self.n_epoch, 
                                  prefix = 'Training Epoch', suffix = '({}/{})'.format(self.epoch_i, self.n_epoch), 
                                  barLength = 70)
                if verbose == 3:
                    pbar.update(1)
                    
                if self.patient_i == self.n_patient:
                    break
                
                self.plot_history(model_name) 
                self.iter_i = 0
                start_time = time.time()

            if self.epoch_i == self.n_epoch:
                break
        
        if verbose == 3:        
            pbar.close()
                    
    def plot_history(self, model_name, save_dir = 'training_history'):
        
        fig = plt.figure(figsize = (20, 8))
        
        x_axis = np.arange(1, self.epoch_i + 1)
        
#         print(x_axis, self.accr_hist, self.loss_hist, self.val_accr_hist, self.val_loss_hist)
        
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, self.accr_hist, 'b-', label = 'Training Accuracy')
        plt.plot(x_axis, self.val_accr_hist, 'r-', label = 'Validation Accuracy')
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.legend(fontsize = 10)
        plt.grid(True)
#         plt.grid('on')
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, self.loss_hist, 'b-', label = 'Training Loss')
        plt.plot(x_axis, self.val_loss_hist, 'r-', label = 'Validation Loss')
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('Loss', fontsize = 15)
        # plt.yticks(np.arange(0, 0.25, step=0.025))
        plt.legend(fontsize = 12)
#         plt.grid('on')
        plt.grid(True)
#         plt.show()

        save_path = self.save_path + '/training_history'
        os.makedirs(save_path, exist_ok = True)
#         model_name = '_'.join(self.model_full_name.split('_')[0:3])

        hyper_params = '{}_{}'.format(self.lr, self.n_batch)
        model_name = '{}_{}'.format(model_name, hyper_params)
        
#         print(model_name)
        
        fig.savefig(save_path + '/{}_training_plot.png'.format(model_name), bbox_inches='tight')
        plt.close(fig)
        
        np.save(save_path + '/{}_training_log'.format(model_name), [self.loss_hist, self.accr_hist, self.val_loss_hist, self.val_accr_hist])
     
    
    @staticmethod    
    def model_selection(model_path, model_name):
#         model_path = save_path + '/model'
        model_list = np.array([i for i in os.listdir(model_path) if model_name + '_' in i])
        n_model = len(model_list)
        t_loss, v_loss = np.zeros([n_model]), np.zeros([n_model])

        for i, file in zip(range(n_model), model_list):
            t_loss[i] = file.split('t_loss')[-1].split('_')[1]
            v_loss[i] = file.split('v_loss')[-1].split('_')[1][:-3]

        best_idx = np.where((t_loss + v_loss) == np.min(t_loss + v_loss))[0]
        delete_idx = np.setdiff1d(np.arange(n_model), best_idx)
        for i in delete_idx:
            os.remove(model_path + '/' + model_list[i])
        print('Best Model:', model_list[int(best_idx)])
          
    def run(self, train_paths, valid_Xs, sampling_mode = None, batch_mode = 'equal', 
            network = 'CNN_v1', model_spec = 1, verbose = 3, 
            save_path = '/mnt/...', model_name = 'NP', gpu_idx = 3,
            classifier_file = None):
        
        self.save_path = save_path 
        self.train_paths, self.n_cls = train_paths, len(train_paths)
        self.class_size = []
        valid_Ys = []
        
        for i, train_path, valid_x in zip(range(self.n_cls), train_paths, valid_Xs):                   
            valid_y = gen_label(valid_x, i)                                                                
            valid_Ys.append(valid_y)
            self.class_size.append(len(train_path))    
            
        self.valid_X = np.concatenate(valid_Xs)
        self.valid_Y = torch.tensor(np.concatenate(valid_Ys), device = self.device).long()
        
        self.get_device(gpu_idx)
        self.define_model_opt(network, summary_show = model_spec) 
                
        if classifier_file is not None:
        # load the weights into generator
            print("loading classifier_weights from:", classifier_file, '\n')
            self.model.load_state_dict(torch.load(classifier_file, 
                                              map_location=lambda storage, loc: storage.cuda(gpu_idx)))
        
        self.training_process(sampling_mode, batch_mode, verbose, model_name = model_name)
#         self.plot_history(model_name)    
        self.model_selection(self.save_path + '/model', model_name)

