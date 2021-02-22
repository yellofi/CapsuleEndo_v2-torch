#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

# import sys
# sys.path.append('/mnt/disk1/project/SMhospital/capsule/torch/1_modules')
from ce_utils.video import load_frames, model_pred, find_clip_idxs, get_gradcam, process_and_save
from ce_utils.record import sec_to_m_s_ms, total_sum_time
from ce_model.cnn_inf import cnn_model

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', action="store", type=str, 
                        default='/mnt/disk1/yunseob/Pytorch/1_CapsuleEndo/2_model_development/', 
                        help='model directory')
    
    parser.add_argument('--model_name', action="store", type=str, 
                        default='np_binary', help='model name')
    
    parser.add_argument('--network', action="store", type=str, 
                        default='CNN_v1', help='network')
    
    parser.add_argument('--gpu_idx', action="store", type=int, 
                        default=3, help='GPU index that you want to use')
    
    parser.add_argument('--frame_root', action="store", type=str, 
                        default='/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200917 videos for AI feedback', 
                        help='frame directory')
    
    parser.add_argument('--save_root', action="store", type=str, 
                        default='./np_binary', help='save directory')
    
    
    args = parser.parse_args()
    print('')
    print('args={}'.format(args))
    
    return args


def main(args):

    model = cnn_model(network = args.network, n_ch = 3, n_cls = 2, 
                      model_dir = args.model_dir, model_name = args.model_name, gpu_idx = args.gpu_idx)


    # readme.txt 읽어옴
    video_log = pd.read_csv(args.frame_root + '/readme.txt', header = 0, delimiter = '|', encoding = "utf-8")


    # 최종 처리 이후 csv 파일에 들어갈 column 이름들 (for statistical analysis)
    record = pd.DataFrame(columns = ['index', 'video', '# of frames', 'video length', 'save frame time', 'load frame time', 
                                     'pred. time', '# of positive frames', '# of clip frames', 'clip length',
                                     'Grad-CAM time','process and save time', 'total time'])

    print('model inference, processing, and saving')

    fps = 5

    for i in range(len(video_log)):
        video_info = video_log.iloc[i]

        idx, video_name, frame_save_time, n_frame = (video_info['index'], 
                                                     video_info['video'], 
                                                     video_info['elapsed time'], 
                                                     video_info['number of frames'])

        print('')
        print(idx, ':', video_name, n_frame, '\n')

        frame_dir = os.path.join(args.frame_root, '{:02d}'.format(idx))

        # 프레임을 numpy 형태로 불러옴 & 걸린 시간도 가져옴
        frames_pre, frame_load_time = load_frames(frame_dir)

        # Binary Model과 이미지 파일을 model_pred_and_time 함수에 넣고 리턴으로 binary model의 예측 결과와 걸린 시간을 가져옴
        # 예측결과는 model의 softmax 결과 값이 담긴 list
        preds, pred_time = model_pred(model, frames_pre)

        # 각각의 예측 모델이 예측한 frame은 softmax의 argmax 결과가 1인 index에 해당하는 frame
        # prediction에 해당하는 index만 가져옴 
        pred_idx = np.where(np.argmax(preds, axis = 1) == 1)[0]
        # prediction index 전후 5프레임씩 더하여, 비디오 클립에 필요한 index를 가져옴
        clip_idx = find_clip_idxs(pred_idx, len(frames_pre))

        # 모델과, 프레임 그리고 grad-cam이 필요한 프레임의 index를 통해 grad cam image를 구함
        cams, cam_time = get_gradcam(model, frames_pre, clip_idx, batch_size = 64, target_layers = ['conv7_2'], target_class = None)


        save_dir = os.path.join(args.save_root, '{:02d}'.format(idx))
        # 원본 frame과 grad-cam frame를 붙여서 아래 위치에 저장함
        processing_time = process_and_save(frame_dir, save_dir, frames_pre, cams, preds, clip_idx)

        total_time = total_sum_time([frame_save_time, frame_load_time, pred_time, cam_time, processing_time])

        # 파일 저장 과정에서 단계별로  소요된 시간 계산
        record.loc[i] = [idx, video_name, n_frame, sec_to_m_s_ms(n_frame/fps), frame_save_time, frame_load_time, 
                         pred_time, len(pred_idx), len(clip_idx), sec_to_m_s_ms(len(clip_idx)/fps),
                         cam_time, processing_time, total_time]


    record.to_csv(args.save_root + '/record.csv',encoding='utf-8-sig', index = False)
    
    return None

if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
    
    
"""
cd /home/yunseob
source pytorch/bin/activate

cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 model_inf_for_SBEC_video.py --frame_root '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200917 videos for AI feedback' --model_dir 'project/SMhospital/capsule/torch/2_model_development/2_AI_feedback/3_add_200713_2/' --model_name np_binary --save_root '/mnt/disk1/project/SMhospital/capsule/torch/3_video_analysis/200917' --gpu_idx 3

python3 model_inf_for_SBEC_video.py --frame_root '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200917 videos for AI feedback' --model_dir '/mnt/disk1/project/SMhospital/capsule/torch/2_model_development/2_AI_feedback/5_total/model/' --model_name np_total --save_root '/mnt/disk1/project/SMhospital/capsule/torch/3_video_analysis/2101 np_total_-r-_-- (200204 videos for clinical test)' --gpu_idx 3
"""

