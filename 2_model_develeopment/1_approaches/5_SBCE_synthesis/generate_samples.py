#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.backends import cudnn
from torch.nn.functional import interpolate
from scipy.misc import imsave
from tqdm import tqdm

import sys
sys.path.append('/mnt/disk1/yunseob/Pytorch/0_Personal/1_DL/2_GAN/b)_MSG_GAN')
from MSG_GAN.GAN import Generator

# turn on the fast GPU processing mode on
cudnn.benchmark = True


# set the manual seed
# th.manual_seed(3)


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu_idx", action="store", type=int,
                        default=3,
                        help="define the device for the training script")
    
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512,
                        help="latent size for the generator")

    parser.add_argument("--depth", action="store", type=int,
                        default=7,
                        help="depth of the network. **Starts from 1")

    parser.add_argument("--out_depth", action="store", type=int,
                        default=6,
                        help="output depth of images. **Starts from 0")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=300,
                        help="number of synchronized grids to be generated")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="MSG_GAN_interp_animation_frames/",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return th.clamp(data, min=0, max=1)


def progressive_upscaling(images):
    """
    upsamples all images to the highest size ones
    :param images: list of images with progressively growing resolutions
    :return: images => images upscaled to same size
    """
    with th.no_grad():
        for factor in range(1, len(images)):
            images[len(images) - 1 - factor] = interpolate(
                images[len(images) - 1 - factor],
                scale_factor=pow(2, factor)
            )

    return images


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    
    device = th.device("cuda:{}".format(args.gpu_idx) if th.cuda.is_available() else "cpu")
    
    print("Creating generator object ...")
    # create the generator object
#     gen = th.nn.DataParallel(Generator(
#         depth=args.depth,
#         latent_size=args.latent_size
#     ))
    gen = th.nn.DataParallel(Generator(
        depth=args.depth,
        latent_size=args.latent_size
    ), device_ids=[device.index])
    
    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
#     gen.load_state_dict(th.load(args.generator_file))
    gen.load_state_dict(th.load(args.generator_file, map_location=lambda storage, loc: storage.cuda(args.gpu_idx)))

    
    # path for saving the files:
    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)
    
    print("Generating scale synchronized images ...")
    for img_num in tqdm(range(1, args.num_samples + 1)):
        # generate the images:
        with th.no_grad():
            point = th.randn(1, args.latent_size)
            point = (point / point.norm()) * (args.latent_size ** 0.5)
            point = point.to(device)
            ss_images = gen(point)

        # resize the images:
        ss_images = [adjust_dynamic_range(ss_image) for ss_image in ss_images]
        ss_images = progressive_upscaling(ss_images)
        ss_image = ss_images[args.out_depth]

        # save the ss_image in the directory
#         imsave(os.path.join(save_path, str(img_num) + ".png"),
#                ss_image.squeeze(0).permute(1, 2, 0).cpu())
        imsave(os.path.join(save_path, "{:03d}".format(img_num) + ".png"),
               ss_image.squeeze(0).permute(1, 2, 0).cpu())

    print("Generated %d images at %s" % (args.num_samples, save_path))


if __name__ == '__main__':
    main(parse_arguments())

"""
python3 generate_samples.py --generator_file MSG_GAN_models_d7/GAN_GEN_756.pth --gpu_idx 4

python3 generate_samples.py --depth 8 --generator_file MSG_GAN_models_d8/GAN_GEN_481.pth --out_dir MSG_GAN_generated_samples_d8 --gpu_idx 6

python3 generate_samples.py --depth 8 --generator_file MSG_GAN_models_d8/GAN_GEN_1000.pth --out_dir MSG_GAN_1000_generated_samples_d8 --gpu_idx 6

python3 generate_samples.py --depth 8 --generator_file MSG_GAN_models_d8/GAN_GEN_1000.pth --out_dir MSG_GAN_1000_generated_samples_d8_2 --num_samples 1000 --gpu_idx 3

"""