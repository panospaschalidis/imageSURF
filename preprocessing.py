"""standard libraries"""
import argparse
import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import imageio.v2
import pdb
"""third party imports"""
from torch.utils.data import Dataset

"""local specific imports"""

def save_to_npz(dir_path, new_dir_path):
    dir_names = glob.glob(os.path.join(dir_path,'*'))
    new_dirnames = glob.glob(os.path.join(new_dir_path,'*'))
    for path, new_path in zip(dir_names[0:3], new_dirnames):
        print(os.path.split(path)[1])
        list_of_paths = glob.glob(os.path.join(path,'*'))
        for val,instance_path in enumerate(list_of_paths):
            print(f"{val}/{len(list_of_paths)}")
            #new_path = os.mkdir(os.path.join(new_path,os.path.split(path)[1]))

            with open(os.path.join(instance_path,'intrinsics.txt')) as f:
                focal = float(f.read().split()[0])
            #np.save(os.path.join(path,focal),focal)
            pose_paths = sorted(glob.glob(os.path.join(instance_path,'pose','*')))
            pose = np.array([
                np.loadtxt(pose_path).reshape(4,4) for pose_path in pose_paths
                ])
            rgb_paths = sorted(glob.glob(os.path.join(instance_path,'rgb','*')))
            rgb = np.array([
                imageio.v2.imread(rgb_path) for rgb_path in rgb_paths
                ])
            np.savez_compressed(
                os.path.join(new_path,os.path.split(instance_path)[1]),
                rgb=rgb,
                pose=pose,
                focal=focal
                )

parser = argparse.ArgumentParser('Turn shapenet data into compressed preprosecced form')
parser.add_argument('--shapenet_path', 
                    type=str, 
                    default=None, 
                    help='path to shapenet category')
parser.add_argument('--data_path', 
                    type=str, 
                    default=None, 
                    help='path to data directory')

args = parser.parse_args()
dir_path = args.shapenet_path
new_dir_path = args.data_path

desired_paths = [
    path 
    for path in glob.glob(os.path.join(dir_path,'*')) 
    if os.path.isdir(path)
]
os.makedirs(new_dir_path, exist_ok=True)
for path in desired_paths:
    os.makedirs(
        os.path.join(
            new_dir_path,
            os.path.basename(path)
        ),
        exist_ok=True
    )

save_to_npz(dir_path, new_dir_path)

