#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
sys.path.append('/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/algorithms')

from ce_utils.data import batch_idxs
from ce_utils.record import progress_bar, sec_to_m_s_ms

import os, glob
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix
from itertools import product

# from tqdm import tqdm
import time
import cv2

def find_models(model_dir, model_name):
    model_list = [i for i in sorted(os.listdir(model_dir)) if model_name in i]
#     model_list = sorted(glob.glob('nh/model/{}*'.format(model_name)))
    return model_list

def compute_accr(y, pred, print_show):
    num_correct = np.sum(y == pred)
    accr = round(100 * num_correct / len(y), 2)
    if print_show == True:
        print("Accuracy: {:.2f} % ({} / {})\n".format(accr, num_correct, len(y)))
    return accr

class cnn_model:
    def __init__(self, network = 'CNN_v1', n_ch = 3, n_cls = 2, 
                 model_dir = 'model/', model_name = 'binary', gpu_idx = 3):
        
        self.network, self.n_ch, self.n_cls, = network, n_ch, n_cls
        self.model_dir, self.model_name = model_dir, model_name
        self.gpu_idx = gpu_idx
        
        self.model_list = find_models(self.model_dir, self.model_name)
        self.get_device(self.gpu_idx)
        
        print('model:')
        for model in self.model_list:
            print(model)
            
        if len(self.model_list) == 1:
            print()
            self.import_model(self.n_ch, self.n_cls, self.model_dir, self.model_list[0])
        else:
            print('The last model is selected\n')
            self.import_model(self.n_ch, self.n_cls, self.model_dir, self.model_list[-1])
        
    def get_device(self, gpu_idx = 3):
#         self.device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and type(gpu_idx) == int:
            self.device = torch.device("cuda:{}".format(gpu_idx))
            current_device = torch.cuda.current_device()
#             print("Device:", torch.cuda.get_device_name(current_device), '\n')
        else:
            self.device = torch.device('cpu')
            print("Device: CPU\n")
            
    def import_model(self, n_ch, n_cls, model_dir, model_file):
        
        if self.network == 'CNN_v1':
            from ce_model.cnns import CNN_v1
            network = CNN_v1(n_ch, n_cls)
        
        self.model = network
        self.model = self.model.to(self.device)
#         self.model.load_state_dict(torch.load(model_dir + model_file))
        self.model.load_state_dict(torch.load(model_dir + model_file, 
                                              map_location=lambda storage, loc: storage.cuda(self.gpu_idx)))
        self.model.eval()

    def prediction(self, x):
#         test_X, test_Y = torch.tensor(test_X, device=self.device).float(), torch.tensor(test_Y, device=self.device).long()
        prob = self.model(torch.tensor(x, device=self.device).float())
        _, pred = torch.max(prob, 1)
        
#         return torch.tensor(output, device = 'cpu').numpy(), pred.tolist()
        return prob.cpu().detach().numpy(), pred.tolist()

    def inference(self, x, batch_size):
        
        start_time = time.time()
        batches = batch_idxs(x, batch_size = batch_size)
        self.score, self.pred = [], []
        for i, batch in enumerate(batches):   
#             score, pred = self.Model_pred(self.model, test_X[batch, :, :, :], test_Y[batch])
            score, pred = self.prediction(x[batch, :, :, :])
            self.score.append(score), self.pred.append(pred)
            progress_bar(i+1, len(batches), prefix = '{}: {}(*{})'.format(len(x), len(batches), batch_size), 
                          suffix = 'prediction', barLength = 50)
        self.score, self.pred = np.concatenate(self.score), np.concatenate(self.pred)
        
        pred_time = time.time() - start_time
        self.pred_time = sec_to_m_s_ms(pred_time)

    def evaluation(self, x, y, batch_size, print_accr = True):
        
        if len(self.model_list) != 1:
            for i, model in enumerate(self.model_list):
                self.import_model(self.n_ch, self.n_cls, self.model_dir, model)
                self.inference(x, batch_size)
                self.accr = compute_accr(y, self.pred, print_show = print_accr)
                
        elif len(self.model_list) == 1:
            self.inference(x, batch_size)
            self.accr = compute_accr(y, self.pred, print_show = print_accr)

    def extract_gradcam(self, img, target_layers = ['conv6_2'], target_class = None):
        
        x = torch.tensor(img, device=self.device).float()
        img_shape = img.shape[2:]
        (dark_xs, dark_ys) = np.where(np.mean(img[0], axis = 0) < 6)
        
#         img = np.transpose(img, (0, 2, 3, 1))
        
        grad_by_layer = []
        def save_gradient(grad):
            grad_by_layer.append(grad)

        feature_by_layer = []

        for name, module in self.model._modules.items():
        #         print(name)
            x = module(x)
            if 'conv' in name:
                x = nn.ReLU()(x)
            if name in target_layers:
                feature_by_layer.append(x) 
                x.register_hook(save_gradient)
            if 'maxp7' == name:
                x = x.view(x.size(0), -1)
        
        logits = x
#         logits = F.softmax(x, -1)
        
        one_hot = np.zeros((img.shape[0], 2), dtype=np.float32)
        one_hot = torch.from_numpy(one_hot).to(self.device).requires_grad_(True)
        
        if target_class == None:
            for i, cls in enumerate(logits.argmax(axis =1).tolist()):
                one_hot[i, cls] = 1
        elif target_class is not None:
            one_hot[:, target_class] = 1
#         one_hot = np.sum(logits * one_hot)

        for target_layer in target_layers:
            self.model._modules[target_layer].zero_grad()
        
        self.model.zero_grad()
        
        logits.backward(gradient=one_hot, retain_graph=True)
        
        cam_by_layer = []
        for features, gradients in zip(feature_by_layer, reversed(grad_by_layer)): 
            cams = []
            for n, feature, gradient in zip(range(len(img)), features, gradients):

                grad_val = gradient.data.cpu().numpy()
                weights = np.mean(grad_val, axis=(1, 2))

                target = feature.cpu().data.numpy()
                cam = np.zeros(target.shape[1:], dtype=np.float32)

                for i, w in enumerate(weights):
                    cam += w * target[i, :, :]
                
#                 cam = np.abs(cam)
                cam = np.maximum(cam, 0)
                cam_re = cv2.resize(cam, img.shape[2:], interpolation=cv2.INTER_CUBIC)
                cam_re = cam_re - np.min(cam_re)
                cam_re = cam_re / np.max(cam_re)
                cam_re[dark_xs, dark_ys] = 0

        #         cams.append(cam_re)

                heatmap = cv2.applyColorMap((cam_re*255).astype('uint8'), cv2.COLORMAP_JET)
                target_pic = np.transpose(img[n], (1, 2, 0))
                cam_on_img = (heatmap*0.3 + target_pic*0.7).astype('uint8')

                cams.append(cam_on_img)  
            cam_by_layer.append(cams)
            
        if len(feature_by_layer) == 1:
            return cams
        elif len(feature_by_layer) > 1:
            return cam_by_layer
    
    def extract_forward_attention(self, img, target_layers = ['conv6_2']):
        
        x = torch.tensor(img, device=self.device).float()
        img_shape = img.shape[2:]
        (dark_xs, dark_ys) = np.where(np.mean(img[0], axis = 0) < 8)

        feature_by_layer = []
        outputs = []
        
        grad_by_layer = []
        def save_gradient(grad):
            grad_by_layer.append(grad)
        
        for name, module in self.model._modules.items():
        #         print(name)
            x = module(x)
            if 'conv' in name:
                x = nn.ReLU()(x)
            if name in target_layers:
                feature_by_layer.append(x) 
                x.register_hook(save_gradient)
                outputs += [x]
            if 'maxp7' == name:
                x = x.view(x.size(0), -1)
        
        logits = x
#         logits = F.softmax(x, -1)
        
        one_hot = np.zeros((img.shape[0], 2), dtype=np.float32)
        one_hot[:, 1] = 1
        one_hot = torch.from_numpy(one_hot).to(self.device).requires_grad_(True)
        one_hot = torch.sum(logits * one_hot)

        for target_layer in target_layers:
            self.model._modules[target_layer].zero_grad()

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
#         logits.backward(gradient=one_hot, retain_graph=True)
        
        cam_by_layer = []
        for features, output in zip(feature_by_layer, outputs): 
            cams = []
            for n, feature, out in zip(range(len(img)), features, output):
                
                alpha = out.cpu().data.numpy()
                weights = np.mean(alpha, axis=(1, 2))
                
                target = feature.cpu().data.numpy()
                cam = np.zeros(target.shape[1:], dtype=np.float32)

                for i, w in enumerate(weights):
                    cam += w * target[i, :, :]
                
#                 cam = np.abs(cam)
                cam = np.maximum(cam, 0)
            #         cam_re = cv2.resize(cam, img.shape[2:])
                cam_re = cv2.resize(cam, img.shape[2:], interpolation=cv2.INTER_CUBIC)
                cam_re = cam_re - np.min(cam_re)
                cam_re = cam_re / np.max(cam_re)
                cam_re[dark_xs, dark_ys] = 0

        #         cams.append(cam_re)

                heatmap = cv2.applyColorMap((cam_re*255).astype('uint8'), cv2.COLORMAP_JET)
                target_pic = np.transpose(img[n], (1, 2, 0))
                cam_on_img = (heatmap*0.3 + target_pic*0.7).astype('uint8')

                cams.append(cam_on_img)
            cam_by_layer.append(cams)
            
        if len(feature_by_layer) == 1:
            return cams
        elif len(feature_by_layer) > 1:
            return cam_by_layer
    
#      def extract_forwardcam(self, img, target_layers = ['conv6_2']):
        
#         x = torch.tensor(img, device=self.device).float()
#         (dark_xs, dark_ys) = np.where(np.mean(img[0], axis = 0) < 8)
        
        
#         gradients = []
#         def save_gradient(grad):
#             gradients.append(grad)

#         features = []
#         outputs = []
        
#         for name, module in self.model._modules.items():
#     #         print(name)
#             x = module(x)
#             if 'conv' in name:
#                 x = nn.ReLU()(x)
#             if name in target_layers:
#                 features.append(x) 
#                 x.register_hook(save_gradient)
#                 outputs += [x]
# #                 gradients += [x]
#             if 'maxp7' == name:
#                 x = x.view(x.size(0), -1)

#         logit = F.softmax(x, -1)
#         one_hot = np.zeros((1, logit.size()[-1]), dtype=np.float32)
#         one_hot[0][logit.argmax().tolist()] = 1
# #         one_hot[0][1] =1
#         one_hot = torch.from_numpy(one_hot).to(self.device).requires_grad_(True)
#         one_hot = torch.sum(one_hot * logit)

#         cams = []
#         for feature, gradient, target_layer in zip(features, outputs, target_layers):
# #         for feature, gradient, target_layer in zip(features, gradients, target_layers):

#             # model.b3_conv1.zero_grad()
#             self.model._modules[target_layer].zero_grad()
#             self.model.zero_grad()
#             one_hot.backward(retain_graph=True)

#             grad_val = gradient.data.cpu().numpy()
#             weights = np.mean(grad_val, axis=(2, 3))[0, :]

#             target = feature.cpu().data.numpy()[0, :]
#             cam = np.zeros(target.shape[1:], dtype=np.float32)

#             for i, w in enumerate(weights):
#                 cam += w * target[i, :, :]

#             cam = np.maximum(cam, 0)
#     #         cam_re = cv2.resize(cam, img.shape[2:])
#             cam_re = cv2.resize(cam, img.shape[2:], interpolation=cv2.INTER_CUBIC)
#             cam_re = cam_re - np.min(cam_re)
#             cam_re = cam_re / np.max(cam_re)
#             cam_re[dark_xs, dark_ys] = 0
#             cams.append(cam_re)
#     #         print(feature.shape, grad.shape, target_layer)

#         return cams
                               
                               
                              
                 
            
            
    

