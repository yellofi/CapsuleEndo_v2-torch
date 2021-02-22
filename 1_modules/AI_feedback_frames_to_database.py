#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import cv2

from ce_utils.record import progress_bar, sec_to_m_s_ms
from ce_utils.label import duplicate_aggregate
from ce_utils.preprocessing import extract_aug_suffix, pre_aug

root = '/mnt/disk2/data/private_data/SMhospital/capsule'

import argparse
import time
from tqdm import tqdm

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source_dir', action="store", type=str, 
                        default='/0 data/video/200713 clip for AI re-screening', help='original frame directory')
    parser.add_argument('--review_dir', action="store", type=str, 
                        default='/0 data/labeled/200713 clip for AI re-screening', help='reviewd frame directory')
    parser.add_argument('--save_dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='save directory')
    
    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args


def main(args):

    source_dir = root + args.source_dir
    review_dir = root + args.review_dir
    save_dir = root + args.save_dir

    os.makedirs(save_dir, exist_ok = True)

    log = pd.read_csv(source_dir + '/readme.txt', delimiter='|', header=0) 
    df = pd.DataFrame(columns = ['negative', 'positive',
                                 'hemorrhagic', 'red_spot', 'angioectasia', 'active_bleeding', 
                                 'depressed', 'erosion', 'ulcer', 'stricture', 
                                 'protruded', 'ampulla_of_vater', 'lymphoid_follicles', 'small_bowel_tumor',
                                 'etc', 'phlebectasia', 'lymphangiectasia',
                                 'source'])
    print(log)
    
    pbar = tqdm(total=len(log), unit='video', bar_format='{l_bar}{bar:40}{r_bar}')
    
    for i in range(len(log)):

#         prefix = log[log['index'] == i+1]['video'].values[0].split('.mpg')[0] # 나중엔 이름 앞에 prefix를 주고 보낼거라 없애도 됨 
        
        if os.path.exists(review_dir + '/{:02d}'.format(i+1)):
        
            TP_list = os.listdir(review_dir + '/{:02d}'.format(i+1) + '/TP')
            FP_list = glob.glob(review_dir + '/{:02d}/*.jpg'.format(i+1))

            for pos in TP_list:
                df.loc[pos] = [0 for _ in range(len(df.columns))]
                df.loc[pos]['positive'] = 1
                df.loc[pos]['source'] = os.path.basename(review_dir) + '/{:02d}'.format(i+1)

#                 df.loc[prefix + '_' + pos] = [0 for _ in range(len(df.columns))]
#                 df.loc[prefix + '_' + pos]['positive'] = 1
#                 df.loc[prefix + '_' + pos]['source'] = os.path.basename(review_dir) + '/{:02d}'.format(i+1)

            for neg in FP_list:
        
                df.loc[os.path.basename(neg)] = [0 for _ in range(len(df.columns))]
                df.loc[os.path.basename(neg)]['negative'] = 1
                df.loc[os.path.basename(neg)]['source'] = os.path.basename(review_dir) + '/{:02d}'.format(i+1)
                
#                 df.loc[prefix + '_' + os.path.basename(neg)] = [0 for _ in range(len(df.columns))]
#                 df.loc[prefix + '_' + os.path.basename(neg)]['negative'] = 1
#                 df.loc[prefix + '_' + os.path.basename(neg)]['source'] = os.path.basename(review_dir) + '/{:02d}'.format(i+1)

            FN_dir = review_dir + '/{:02d}'.format(i+1) + '/FN'
            if os.path.isdir(FN_dir):
                FN_list = os.listdir(FN_dir)

                for pos in FN_list:
                    df.loc[pos] = [0 for _ in range(len(df.columns))]
                    df.loc[pos]['positive'] = 1
                    df.loc[pos]['source'] = os.path.basename(review_dir) + '/{:02d}/FN'.format(i+1)
        pbar.update(1)
    
    print(df)
    
    existed_files = os.listdir(save_dir)
    existed_files = sorted(list(set([f.split('__c')[0] + '.jpg' for f in existed_files]))) # 나중에 이름 다 '_c'로 바꿀거임

    ex_i = 0
    
    
    start = time.time()
    pbar = tqdm(total=len(df), unit='id', bar_format='{l_bar}{bar:40}{r_bar}')
    for i, name in enumerate(df.index.values):
        
#         progress_bar(i+1, len(df.index.values), prefix = 'augmentation', suffix = '{}'.format(len(df.index.values)), decimals = 1, barLength = 70)
        pbar.update(1)
        
        if 'FN' in df.loc[name]['source']:
            source_path = os.path.join('/'.join(review_dir.split('/')[:-1]), df.loc[name]['source'], name)
        else:
#             source_path = os.path.join('/'.join(source_dir.split('/')[:-1]), df.loc[name]['source'], name) 
            source_path = os.path.join(source_dir, (df.loc[name]['source']).split('/')[-1])
    
            if 'annotation' in name:
                name = ''.join(name.split('_annotation'))
            
            source_path = os.path.join(source_path, name)  
        
        if name in existed_files:
            ex_i += 1
#             print('', name, 'existed')
            continue
        
#         print(source_path)
        img = cv2.imread(source_path)
        aug_suffixes = extract_aug_suffix(frb_switch = [1, 1, 1], sv_switch = True, mode = 'preprocessing')
        for aug_suffix in aug_suffixes:
            aug_img = pre_aug(img, aug_suffix)
            cv2.imwrite(save_dir + '/' + name[:-4] + '_' + aug_suffix + '.jpg', aug_img)
        
#         if i == 10:
#             break
#         break
            
    taken = time.time()-start
    print()
    print(sec_to_m_s_ms(taken), end ="\n")
    print('already augmented files:', ex_i, 'newly augmented files:', len(df.index.values) - ex_i)
    
    df_ = pd.read_csv('/'.join(save_dir.split('/')[:-1]) + '/label.csv', index_col = 0)
    print('existing label:',df_.shape)
    df_ = df_.append(df, sort = False)
    df_ = duplicate_aggregate(df_)
    print('new label:',df_.shape)
    
    df_.to_csv('/'.join(save_dir.split('/')[:-1]) + '/label.csv', encoding='utf-8-sig', index = True)  
    
    
if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
    
    
"""
cd /mnt/disk1/project/SMhospital/capsule/algorithms

python3 'AI feedback frames to database.py' --source_dir '/0 data/video/200713 clip for AI re-screening' --review_dir '/0 data/labeled/200713 clip for AI re-screening' --save_dir '/1 preprocessed/database'

python3 AI_feedback_frames_to_database.py --source_dir '/0 data/video/200713 videos for AI feedback' --review_dir '/0 data/labeled/200713-2' --save_dir '/0 data/labeled/200713-2/preprocessed'

python3 AI_feedback_frames_to_database.py --source_dir '/0 data/video/200713 videos for AI feedback' --review_dir '/0 data/labeled/200713-2' --save_dir '/1 preprocessed/database'

------ 이전까지 prefix가 필요했음

python3 AI_feedback_frames_to_database.py --source_dir '/0 data/video/200917 videos for AI feedback' --review_dir '/0 data/labeled/200917 AI feedback' --save_dir '/0 data/labeled/200917 AI feedback/preprocessed'

python3 AI_feedback_frames_to_database.py --source_dir '/0 data/video/200917 videos for AI feedback' --review_dir '/0 data/labeled/200917 AI feedback' --save_dir '/1 preprocessed/database'

"""