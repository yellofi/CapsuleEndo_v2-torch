#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
from itertools import product

def cropping(img):
    img = np.array(img, dtype = 'f4')
    img_pre = img[32:544, 32:544, :]
    for i in range(100):
        for j in range(100):
            if i + j > 99:
                pass
            else :
                img_pre[i, j, :] = 0
                img_pre[i, -j, :] = 0
    return img_pre.astype('uint8')

"""
augmentation
"""

def rotate(img, degree):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D(center = (cols/2, rows/2), angle = degree, scale = 1)
    img_rotated = cv2.warpAffine(img, M, dsize = (rows, cols))
    return img_rotated

def avg_blur(img):
    return cv2.blur(img, (5,5)).astype('uint8')
    
def motion_blur(img):
    kernel_size = 15
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    return cv2.filter2D(img, -1, kernel_motion_blur).astype('uint8')

def edge_enhancement(img):
    kernel_edge = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    return cv2.filter2D(img, -1, kernel_edge).astype('uint8') 

def bgr2_h_s_v(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v

def hsv_control(ch_data, ctr_value, ch_name):
    """
    ch_data: data of channel (h, s, or v) which you want to revise by ctr_value / shape: image.shape[0:2]
    ctr_value: the value that will be added to corresponding channel.
    ch_name: 'h', 's', or 'v'
    """
    ch_data_rev = ch_data.copy()
    if ctr_value > 0:
        ch_data_rev[np.where(ch_data <= 255 - ctr_value)] += ctr_value
    else:
        ch_data_rev[np.where(ch_data + ctr_value >= 0)] -= abs(ctr_value)
    return ch_data_rev

def s_rev(img, s_value):
    h, s, v = bgr2_h_s_v(img)
    s_rev = hsv_control(s, s_value, ch_name = 's')
    return [h, s_rev, v]

def v_rev_after_s_rev(s_rev_outputs, v_value):
    h, s_rev, v = s_rev_outputs
    v_rev = hsv_control(v, v_value, ch_name = 'v')
    v_rev[np.where(v <= 7)] = 0
    img_sv = cv2.merge((h, s_rev, v_rev))
    return cv2.cvtColor(img_sv, cv2.COLOR_HSV2BGR)

def extract_aug_suffix(frb_switch = [1, 1, 1], sv_switch = True, mode = 'load'):
    """
    frb_switch = [1, 1, 1], [0, 0 ,1], [1, 1, 0].... 
    that means [flip, rotate, blur_sharp]
    """
    phase0 = ['_c']
    phase1 = {1: ['-', 'f'], 0: ['-']}
    phase2 = {1: ['-', 'r1', 'r2', 'r3'], 0: ['-']}
    phase3 = {1: ['-', 'ab', 'mb', 'eh'], 0: ['-']}
    phase4 = ['s_-30_v_30', 's_-30_v_-30', 's_30_v_-30', 's_30_v_30']

    if mode == 'load':
        phase_a_items = [phase1[frb_switch[0]], phase2[frb_switch[1]], phase3[frb_switch[2]]]
    elif mode == 'preprocessing':
        phase_a_items = [phase0, phase1[frb_switch[0]], phase2[frb_switch[1]], phase3[frb_switch[2]]]

    phase_a = []
    for i in list(product(*phase_a_items)):
        phase_a.append('_'.join(i))

    if not sv_switch == False:
        phase_b = []
        for i in list(product(*[phase_a, phase4])):
            phase_b.append('_'.join(i))
        return list(np.hstack([phase_a, phase_b]))
    else:
        return phase_a 

def pre_aug(img, aug_suffix = 'c'): 
        
    """
    phase, ex) 'c_f_-_mb_s_-30_v_30' -> 'c_f_-_mb_s-30_v30' -> ['c', 'f', '-', 's-30','v30']
    It allows to preprocess the image in specific phase, but slower it is fit to check preprocessing with small data
    """
    function = {'': (lambda x: x), '-': (lambda x: x),
                'c': (lambda x: cropping(x)),
                'f': (lambda x: np.flipud(x)), 
                'r1': (lambda x: rotate(x, 90)), 
                'r2': (lambda x: rotate(x, 180)), 
                'r3': (lambda x: rotate(x, 270)),
                'ab': (lambda x: avg_blur(x)),
                'mb': (lambda x: motion_blur(x)),
                'eh': (lambda x: edge_enhancement(x)),
                's-30': (lambda x: s_rev(x, -30)),
                's30': (lambda x: s_rev(x, 30)),
                'v-30': (lambda x: v_rev_after_s_rev(x, -30)),
                'v30': (lambda x: v_rev_after_s_rev(x, 30))}
    values = ['-30', '30']
    for i in values:
        if i in aug_suffix:
            aug_suffix = aug_suffix.replace('_{}'.format(i), str(i))
    aug_seg = aug_suffix.split('_')  
    for i, p in zip(range(len(aug_seg)), aug_seg):
        if i == 0:
            p_img = function[p](img)
        else:
            p_img = function[p](p_img)
    return p_img

