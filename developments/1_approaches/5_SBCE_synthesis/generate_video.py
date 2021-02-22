#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import glob
import cv2
import sys
from tqdm import tqdm

def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_dir", action="store", type=str,
                        default="frames/",
                        help="path to the source directory for the frames")
    
    parser.add_argument("--img_type", action="store", type=str,
                        default=".jpg",
                        help="image file type **Default .jpg")
    
    parser.add_argument("--fps", action="store", type=int,
                        default=5,
                        help="frame per sec for the video **Default 5")
    
    parser.add_argument("--out_dir", action="store", type=str,
                        default="",
                        help="path to the output directory for the video")
    
    parser.add_argument("--filename", action="store", type=str,
                        default="video.avi",
                        help="ex) video.avi")

    args = parser.parse_args()

    return args

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 70):
    
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()    
    
def main(args):

    frames = glob.glob(args.img_dir +  '/*{}'.format(args.img_type))

    size = cv2.imread(frames[0]).shape[:-1]
    size = (size[1], size[0])
    out = cv2.VideoWriter(args.out_dir + args.filename, cv2.VideoWriter_fourcc(*'DIVX'), args.fps, size)

#     for i, filename in zip(range(len(frames)), frames):
#         img = cv2.imread(filename) # cv2.imread 파일경로에 한글 x 
#         out.write(img)
#         printProgress(i+1, len(frames), prefix = str(len(frames)))
    
    for filename in tqdm(frames):
        img = cv2.imread(filename) # cv2.imread 파일경로에 한글 x 
        out.write(img)
    
    out.release() 
    
if __name__ == '__main__':
    main(parse_arguments())

"""
python3 generate_video.py --img_dir MSG_GAN_samples_d7/256_x_256 --img_type 388.png 

python3 generate_video.py --img_dir MSG_GAN_samples_d7/256_x_256 --img_type 388.png

python3 generate_video.py --img_dir MSG_GAN_samples_d8/512_x_512 --img_type 776.png --fps 30

python3 generate_video.py --img_dir MSG_GAN_samples_d7/256_x_256 --img_type 388.png --filename MSG_GAN_samples_d7_time_lapse_video.avi --fps 30
"""