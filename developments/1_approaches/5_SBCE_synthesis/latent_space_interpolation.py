#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" script for generating samples from a trained model """

import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from torch.backends import cudnn
from torchvision.utils import make_grid
from math import sqrt, ceil
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# turn fast mode on
cudnn.benchmark = True

# define the device for the training script
# device = th.device("cuda" if th.cuda.is_available() else "cpu")

def parse_arguments():
    """
    command line arguments parser
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
                        default=9,
                        help="latent size for the generator")

    parser.add_argument("--time", action="store", type=float,
                        default=30,
                        help="Number of seconds for the video to make")

    parser.add_argument("--fps", action="store", type=int,
                        default=30, help="Frames per second in the video")

    parser.add_argument("--smoothing", action="store", type=float,
                        default=2.0, help="Amount of smoothing applied in transitional points")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="interp_animation_frames/",
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
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return th.clamp(data, min=0, max=1)

def progressive_upscaling(images):
    from torch.nn.functional import interpolate
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

def get_image(gen, point):
    images = list(map(lambda x: x.detach(), gen(point)))
    images = [adjust_dynamic_range(image) for image in images]
    images = progressive_upscaling(images)
    # discard 128_x_128 resolution (temporarily)
    images = images[:-2] + images[-1:]
    images = list(map(lambda x: x.squeeze(dim=0), images))
    image = make_grid(
        images,
        nrow=int(ceil(sqrt(len(images)))),
        padding=0
    )
    return image.cpu().numpy().transpose(1, 2, 0)

def main(args):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """
    device = th.device("cuda:{}".format(args.gpu_idx) if th.cuda.is_available() else "cpu")

    import sys
#     sys.path.append('/mnt/disk1/yunseob/Pytorch/0_Personal/3_CelebA/2_GAN/b)_MSG_GAN')
    sys.path.append('/mnt/disk1/yunseob/Pytorch/0_Personal/1_DL/2_GAN/b)_MSG_GAN')
    from MSG_GAN.GAN import Generator

    # create generator object:
    print("Creating a generator object ...")
#     generator = th.nn.DataParallel(
#         Generator(depth=args.depth,
#                   latent_size=args.latent_size).to(device))
    
    generator = th.nn.DataParallel(
        Generator(depth=args.depth,
                  latent_size=args.latent_size), device_ids=[device.index])

    # load the trained generator weights
    print("loading the trained generator weights ...")
#     generator.load_state_dict(th.load(args.generator_file))
    generator.load_state_dict(th.load(args.generator_file, 
                                      map_location=lambda storage, loc: storage.cuda(args.gpu_idx)))
    # total_frames in the video:
    total_frames = int(args.time * args.fps)

    # Let's create the animation video from the latent space interpolation
    # all latent vectors:
    all_latents = th.randn(total_frames, args.latent_size).to(device)
    all_latents = gaussian_filter(all_latents.cpu(), [args.smoothing * args.fps, 0])
    all_latents = th.from_numpy(all_latents)
    all_latents = (all_latents / all_latents.norm(dim=-1, keepdim=True))                   * (sqrt(args.latent_size))

    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    global_frame_counter = 1
    # Run the main loop for the interpolation:
    print("Generating the video frames ...")
    for latent in tqdm(all_latents):
        latent = th.unsqueeze(latent, dim=0)

        # generate the image for this point:
        img = get_image(generator, latent)

        # save the image:
        plt.imsave(os.path.join(args.out_dir, str(global_frame_counter) + ".png"), img)

        # increment the counter:
        global_frame_counter += 1

    # video frames have been generated
    print("Video frames have been generated at:", args.out_dir)


if __name__ == "__main__":
    main(parse_arguments())
    
"""
python3 latent_space_interpolation.py --depth 7 --gpu_idx cuda:2 --generator_file MSG_GAN_models_d7/GAN_GEN_756.pth --out_dir "MSG_GAN_latent_animation_frames_d7/"

python3 latent_space_interpolation.py --depth 8 --gpu_idx 3 --generator_file MSG_GAN_models_d8/GAN_GEN_1000.pth --out_dir "MSG_GAN_latent_animation_frames_d8/"
"""

