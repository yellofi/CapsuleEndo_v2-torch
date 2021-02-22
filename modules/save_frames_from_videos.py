#!/usr/bin/env python
# coding: utf-8

import os
import cv2     # for capturing videos
import time

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_dir', action="store", type=str, default='.', help='video directory')
    parser.add_argument('--save_dir', action="store", type=str, default='.', help='frame save directory')
    
    args = parser.parse_args()
    print('')
    print('args={}'.format(args))
    
    return args

def saveframe(video_path, save_dir, sub_folder):
#     video_path = os.path.join(video_dir, video)
#     index = '{:02d}'.format(index)
    save_path = os.path.join(save_dir, sub_folder)
    if not(os.path.isdir(save_path)):
        os.makedirs(save_path)  
    
    video_name = os.path.basename(video_path)
    prefix = video_name.split('.')[0]
    print(video_name, ' | ', save_path, ' | saving .... ')
                        
    cap = cv2.VideoCapture(video_path)   # capturing the video from the given path
    n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # cap.get(7), total # of frame
    frameRate =  cap.get(cv2.CAP_PROP_FPS) # cap.get(5), frame rate
    
    count = 0
    start_time = time.time()
    while(cap.isOpened()):
        frameId =  cap.get(cv2.CAP_PROP_POS_FRAMES)  # cap.get(1), current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if frameId % 5 == 0: #if (frameId % math.floor(frameRate) == 0):
            count += 1
#             filename = save_dir + '/frame_{0:05d}.jpg'.format(count)
            filename = save_path + '/{0}_frame_{1:05d}.jpg'.format(prefix, count)
            cv2.imwrite(filename, frame) # 폴더명 한글 x
    time_taken = time.time() - start_time
    min_sec = time.strftime("%M:%S", time.gmtime(time_taken))
    ms = '{:03d}'.format(int((time_taken - int(time_taken))*1000))
    time_taken = '.'.join([min_sec, ms])
    print('elapsed time: {}, number of frames: {}'.format(time_taken, count))
    return time_taken, count
    
def __main__(args):

    video_list = [i for i in sorted(os.listdir(args.video_dir)) if 'mpg' in i]
    folder_name = range(1, len(video_list)+1)
   
    for index, video in zip(folder_name, video_list):
        print('{:02d} : {}'.format(index, video))
    
    print('number of videos:', int(len(video_list)))
#     start_idx = int(input('프레임 저장을 시작할 비디오 (인덱스):'))
    file = open(args.save_dir + '/' + 'readme.txt', 'w')
    file.write('index|video|elapsed time|number of frames\n')
    for index, video in zip(folder_name, video_list):
        video_path = os.path.join(args.video_dir, video)
        time_taken, count = saveframe(video_path, args.save_dir, sub_folder = '{:02d}'.format(index))
#         file.write('{:02d} | {} | {}\n'.format(index, video, time_taken))
        file.write('{:02d}|{}|{}|{}\n'.format(index, video, time_taken, count))
    file.close()
# print('currnet path:', os.getcwd())
# video_dir = './video/200204 clip for clinical test'
# video_dir = '.' # 현재 폴더에 있는 mpg 동영상 파일을 대상.
# SaveFrame_from_videos(video_dir)
                        
                        
if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 save_frames_from_videos.py --video_dir '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200917 videos for AI feedback' --save_dir '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200917 videos for AI feedback'

python3 save_frames_from_videos.py --video_dir '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200204 clip for clinical test' --save_dir '/mnt/disk2/data/private_data/SMhospital/capsule/0 data/video/200204 clip for clinical test'




"""