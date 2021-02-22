#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action="store", type=str, 
                        default='/1 preprocessed/database', help='define current working directory')
    parser.add_argument('--original', action="store", type=str, default='___c', help='original char')
    parser.add_argument('--new', action="store", type=str, default='__c', help='original char')
    args = parser.parse_args()
    print('args={}'.format(args))
    
    return args

def __main__(args):
    os.chdir(args.dir)
    for filename in os.listdir(args.dir):
        if args.original in filename:
            new_filename = filename.replace(args.original, args.new)
            os.rename(filename, new_filename)
            
if __name__ == '__main__':
    # invoke the main function of the script
    __main__(parse_arguments())

    
"""
cd /mnt/disk1/project/SMhospital/capsule/torch/1_modules

python3 rename.py --dir /mnt/disk1/project/SMhospital/capsule/torch/1_modules --original configuration --new config 
"""
