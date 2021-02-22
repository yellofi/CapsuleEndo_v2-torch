#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import numpy as np
import cv2

from ce_utils.record import progress_bar
from ce_utils.preprocessing import extract_aug_suffix, pre_aug

root = '/mnt/disk2/data/private_data/SMhospital/capsule'

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source_dir', action="store", type=str, 
                        default='/0 data/labeled/190520 p3_2/p3_2/protruded', help='original frame directory')
    parser.add_argument('--save_dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='save directory')
    
    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args


def __main__(args):

    source_dir = root + args.source_dir
    save_dir = root + args.save_dir

    os.makedirs(save_dir, exist_ok = True)

    files = os.listdir(source_dir) # negative, hemorrhagic, depressed, ... , etc
    lesions = os.listdir(os.path.join(source_dir)) # negative, red_spot ..., erosion, stricture, ...., ampulla_of_vater...
    
    if os.path.basename(source_dir) == 'etc':
        lesions = ['phlebectasia', 'lymphangiectasia']
    
    paths = []
    for l in lesions:
        for f in glob.glob(os.path.join(source_dir, l) + '/*.jpg'):
            paths.append(f)   
    
    existed_files = os.listdir(save_dir)
    existed_files = sorted(list(set([f.split('__c')[0] + '.jpg' for f in existed_files]))) # 나중에 이름 다 '_c'로 바꿀거임
    
    for i, path in enumerate(paths):
        
        progress_bar(i+1, len(paths), prefix = 'augmentation', suffix = '{}'.format(len(paths)), decimals = 1, barLength = 70)
        file_name = os.path.basename(path)
        
        if file_name in existed_files:
            continue
            
#         if file_name == 'NT___02-33-55___1018475.jpg
        else: 
#         if 'annotation' in file:
#             file = ''.join(file.split('_annotation'))
        
            img = cv2.imread(path)
            aug_suffixes = extract_aug_suffix(frb_switch = [1, 1, 1], sv_switch = True, mode = 'preprocessing')

            for aug_suffix in aug_suffixes:
                aug_img = pre_aug(img, aug_suffix)
                cv2.imwrite(save_dir + '/' + file_name[:-4] + '_' + aug_suffix + '.jpg', aug_img)
            
            
if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())
    

"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 direct_augmentation_and_save.py --source_dir '/0 data/labeled/190520 p3_2/p3_2/protruded' --save_dir '/1 preprocessed/database'
"""

