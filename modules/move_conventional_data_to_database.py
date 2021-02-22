#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, glob
import shutil
import argparse
from ce_utils.record import printProgress


root = '/mnt/disk2/data/private_data/SMhospital/capsule'

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', action="store", type=str, 
                        default='/1 preprocessed/sm_x160_v2', help='original directory')
    parser.add_argument('--move_dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='moving directory')

    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args

['negative', 'positive',
                                 'hemorrhagic', 'red_spot', 'angioectasia', 'active_bleeding', 
                                 'depressed', 'erosion', 'ulcer', 'stricture', 
                                 'protruded', 'ampulla_of_vater', 'lymphoid_follicles', 'small_bowel_tumor',
                                 'etc', 'phlebectasia', 'lymphangiectasia',
                                 'source']
def __main__(args):
    
    data_dir = root + args.data_dir
    move_dir = root + args.move_dir
    
    path = []

    phases = os.listdir(data_dir)
     # train, test
    
    if 'total' in phases:
        phases = ['total']
    for p in phases:
        types = os.listdir(os.path.join(data_dir, p))
        # negative, hemorrhagic, depressed, ... , etc
        for t in types:
            lesions = os.listdir(os.path.join(data_dir, p, t))
            # negative, red_spot ..., erosion, stricture
            if t == 'etc':
                lesions = ['phlebectasia', 'lymphangiectasia']
            for l in lesions:
                for f in glob.glob(os.path.join(data_dir, p, t, l) + '/*.jpg'):
                    path.append(f)      

    for i, f in enumerate(path):
        printProgress(i+1, len(path), prefix = '', suffix = '', decimals = 1, barLength = 70)
        shutil.move(f, os.path.join(move_dir, os.path.basename(f)))

if __name__ == '__main__':
    __main__(parse_arguments())
     
"""
python3 'move conventional data to database.py' --data_dir '/1 preprocessed/sm_x160_v2' --move_dir '/1 preprocessed/database'
"""

