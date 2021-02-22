#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import numpy as np
import cv2

import time
from ce_utils.preprocessing import cropping
from ce_utils.record import progress_bar, sec_to_m_s_ms
from ce_utils.data import batch_idxs, reshape4torch

# def pre_process(image):
#     image = np.array(image)
#     image_pre = image[32:544, 32:544, :]
#     for i in range(100):
#         for j in range(100):
#             if i + j > 99:
#                 pass
#             else :
#                 image_pre[i, j, :] = 0
#                 image_pre[i, 511 - j, :] = 0
#     return image_pre

def load_frame(frame_path, img_ch = 'bgr'):
    frame = cv2.imread(frame_path)
    if img_ch == 'rgb':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_pre = pre_process(frame)
    frame_pre = cropping(frame)
    return frame_pre

def load_frames(frame_dir):
    #frame 디렉토리에 있는 이미지 파일 이름 가져와서 정렬
    frame_names = np.sort(os.listdir(frame_dir))
    n_frame = frame_names.shape[0]
    frames_pre = []
    
    #이미지 불러오는 시작 시간 
    start_time = time.time()
    for i, frame_name in enumerate(frame_names):
        # 프레임 불러와서 frames_pre 에 저장 
        frame = load_frame(os.path.join(frame_dir, frame_name))
        frames_pre.append(frame)
        progress_bar(i+1, n_frame, prefix = n_frame, suffix = 'loading frames', barLength = 80)
#     frames_pre = np.asarray(frames_pre)
    frames_pre = np.array(frames_pre)
    time_taken = time.time() - start_time
    time_taken = sec_to_m_s_ms(time_taken)
    # 리턴으로 불러온 프레임들과 걸린 시간 
    return frames_pre, time_taken


# def find_clip_idxs(pred_idx, n_frame):
#     if pred_idx.size == 0:
#         pred_idx =  [0, 1, 2, 3, 4]
#     clip_idx = []
#     for idx in pred_idx:
#         clip_idx.append([idx + i - 5 for i in range(11)])
#     clip_idx = np.concatenate(clip_idx)
#     clip_idx = np.unique(clip_idx)
#     idx_start = np.where(clip_idx >= 0)
#     idx_end = np.where(clip_idx < n_frame)
#     return clip_idx[np.intersect1d(idx_start, idx_end)]

"""
model inference
"""

def model_pred(model, frames_pre):
    model.inference(reshape4torch(frames_pre), batch_size = 16)
    probs = model.score
    pred_time = model.pred_time
    return probs, pred_time

def find_clip_idxs(pred_idx, n_frame):
    assert len(pred_idx) > 0, 'model must predict a image as positive at least'
    clip_idx = []
    for p_idx in pred_idx:
        clip_idx += [p_idx + i - 5 for i in range(11) if (p_idx + i-5) < n_frame and (p_idx + i-5) >= 0]
    return np.asarray(sorted(set(clip_idx)))


def get_gradcam(model, frames_pre, clip_idx, batch_size, target_layers = ['conv6_2'], target_class = 0):
    start_time = time.time()
    batches = batch_idxs(clip_idx, batch_size)
    cams = []
    
    for i, batch in enumerate(batches):
        cam_on_img = model.extract_gradcam(reshape4torch(frames_pre[clip_idx[batch]]), target_layers, target_class)
        cams.append(cam_on_img)
        
        progress_bar(i+1, len(batches), barLength = 80,
                      prefix = '{}: {}(*{})'.format(len(clip_idx), len(batches), batch_size), 
                      suffix = 'get gradcam in clip frame')
       
    cams = np.concatenate(cams)

    cam_time = time.time() - start_time
    cam_time = sec_to_m_s_ms(cam_time)
    
    return cams, cam_time

def get_forward_attention(model, frames_pre, clip_idx, batch_size, target_layers = ['conv6_2']):
    start_time = time.time()
    batches = batch_idxs(clip_idx, batch_size)
    cams = []

    for i, batch in enumerate(batches):
        cam_on_img = model.extract_forward_attention(reshape4torch(frames_pre[clip_idx[batch]]), target_layers)
        cams.append(cam_on_img)

        progress_bar(i+1, len(batches), barLength = 80,
                      prefix = '{}: {}(*{})'.format(len(clip_idx), len(batches), batch_size), 
                      suffix = 'get gradcam in clip frame')

    cams = np.concatenate(cams)

    cam_time = time.time() - start_time
    cam_time = sec_to_m_s_ms(cam_time)

    return cams, cam_time

"""
post processing
"""

def merging(image, cam):
    cam_pad = np.zeros([576, 576, 3]) # zero padding
#     cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    cam_pad[32:544,32:544, :] = cam
    return np.concatenate([image, cam_pad], axis = 1)

def captioning(image, softmax, model_type = 'binary'):
    if model_type == 'binary':
        cap = "Predict : {}".format(np.argmax(softmax, axis =0))
        cv2.putText(image, cap, (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cap2 = 'Probability: {0:0.2f}%'.format(100*softmax[1])
        cv2.putText(image, cap2, (760, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif model_type == 'ensemble':
        cap = "Predict : {}".format(np.argmax(softmax[:2], axis =0))
        cv2.putText(image, cap, (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cap2 = 'Hem.: {0:0.2f}%, Ulc.: {1:0.2f}%'.format(100*softmax[3], 100*softmax[5])
        cv2.putText(image, cap2, (700, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
def process_and_save(frame_dir, save_dir, frames_pre, cams, preds, clip_idx, model_type = 'binary'):
    
    frame_names = np.sort(os.listdir(frame_dir))
    
    if not(os.path.isdir(save_dir)):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/clip_frame')
        os.makedirs(save_dir + '/pred_frame')
    
    start_time = time.time()
    for i, idx in enumerate(clip_idx):
        image = cv2.imread(os.path.join(frame_dir, frame_names[idx]))
        
        if preds[idx, 1] >= 0.2:
            merged = merging(image, cams[i])
            captioning(merged, preds[idx], model_type)
        elif preds[idx, 1] < 0.2:
            merged = merging(image, frames_pre[idx])
            captioning(merged, preds[idx], model_type)

        cv2.imwrite(save_dir + '/clip_frame/' + frame_names[idx], merged)
        if model_type == 'binary':
            if np.argmax(preds[idx]) == 1:
                cv2.imwrite(save_dir + '/pred_frame/' + frame_names[idx], merged)
#         elif model_type == 'ensemble':
#             if np.argmax(preds[idx][:2]) == 1:
#                 cv2.imwrite(save_dir + 'pred_frame/' + frame_names[idx], merged)
        progress_bar(i+1, len(clip_idx), prefix = len(clip_idx), suffix = 'process and save --> {}'.format(save_dir), 
                      barLength = 80)
    time_taken = time.time() - start_time
    time_taken = sec_to_m_s_ms(time_taken)
    return time_taken

