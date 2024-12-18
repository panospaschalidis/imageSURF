import os
import sys
import logging
import time
import shutil
import argparse
import pdb
import wandb
import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl
from model.common import ray_tracing

#logger_py = logging.getLogger(__name__)

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument("--resume", "-r", action="store_true", 
                help="continue training")
parser.add_argument("--project_name", type=str, 
                help="path of model named after project name")
parser.add_argument("--sviews", type=str, 
                help="source views being used for feature extraction")
parser.add_argument("--nviews", type=int, 
                help="number of views")
args = parser.parse_args()


cfg = dl.load_config(
    os.path.join(os.getcwd(),args.config), 
    'configs/default.yaml'
)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
image_resolution = cfg['dataloading']['img_size']
cfg['model']['encoder']['scale_size'] = image_resolution

# params
out_dir = cfg['training']['out_dir']
checkpoint_path = cfg['training']['checkpoints_path']
#backup_every = cfg['training']['backup_every']
epochs = cfg['training']['epochs']
exit_after = args.exit_after
batch_size_train = cfg['dataloading']['batchsize']['train']
batch_size_val = cfg['dataloading']['batchsize']['val']
n_workers = cfg['dataloading']['n_workers']
lr = cfg['training']['learning_rate']
test_loader = dl.get_dataloader(cfg, mode='test')
#iter_test = iter(test_loader)
# init network
model_cfg = cfg['model']
model = mdl.NeuralNetwork(model_cfg)
# init checkpoints and load
checkpoint_io = mdl.CheckpointIO(checkpoint_path, project_name=args.project_name)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
model = load_dict.get('model', model)
rendering_cfg = cfg['rendering']
renderer = mdl.Renderer(model, rendering_cfg, device=device)
sviews = torch.tensor(sorted(list(map(int, args.sviews.split()))), dtype=torch.long)
indices = torch.arange(args.nviews).tolist()
ind = sorted(sviews)
for val, i  in enumerate(ind):
    if  val>0:
        i = i -1
    indices.pop(i)
os.makedirs(os.path.join(os.getcwd(),'exported_images'), exist_ok=True)
z_near, z_far = 0.8, 1.8
for val, data in enumerate(test_loader):
        print(f"{val} instance {data['instance'][val]}")

        batch_size, _, *im_shape = data['images'].shape
        rays_dir, rays_origin = ray_tracing(
            im_shape, 
            data['focal'][-1], 
            data['poses'][:,[2],...]
        )
        rays_dir = rays_dir.to(device)
        rays_origin = rays_origin.unsqueeze(1).repeat_interleave(rays_dir.shape[1],dim=1).to(device)
        pdb.set_trace()
        rays_dir = rays_dir.permute(0,2,1,3).flatten(1,2)
        rays_origin = rays_origin.permute(0,2,1,3).flatten(1,2)
        rays_dir_spl = torch.split(rays_dir,1000, dim =1) 
        rays_ori_spl = torch.split(rays_origin,1000, dim =1) 
        rgb = []
        depth = []  
        encoding_dict = { 
            'images': data['images'],
            'poses': data['poses'],
            'focal':data['focal'],
            'prcpal_p': data['prcpal_p']
        }
        encoding_dict['poses'] = encoding_dict['poses'][torch.arange(batch_size)[:,None],sviews,...][...,:3,:]#b_sxviewsx3x4
        encoding_dict['images'] = encoding_dict['images'][torch.arange(batch_size)[:,None],sviews,...]#b_Sxviewsx3xHXW
        for key, value in encoding_dict.items():
            encoding_dict[key] = value.to(device)
        renderer.model.encode(encoding_dict)
        with torch.no_grad():
            for val in tqdm(range(len(rays_dir_spl))):
              pred = renderer(
                      rays_dir_spl[val],
                      rays_ori_spl[val],
                      'unisurf',
                      val=val,
                      it=it, 
                      mask=None, 
                      eval_=True
                      )   
              rgb.append(pred['rgb'])
              depth.append(pred['depth'])
        pdb.set_trace() 
        rgb_pred = torch.cat(rgb, dim=1).squeeze().reshape(2,*im_shape[1:],im_shape[0])
        depth_pred = torch.cat(depth, dim=1).squeeze().reshape(2,*im_shape[1:])
        rgb_hat = rgb_pred.cpu().numpy()
        depth_hat = depth_pred.cpu().numpy()

        norm_depth = (depth_hat-np.min(depth_hat, axis=(1,2))[:,None,None])/(np.max(depth_hat,axis=(1,2))-np.min(depth_hat,axis=(1,2)))[:,None,None]
        np.savez_compressed(
            os.path.join(
                os.getcwd(),
                'exported_images',
                data['instance'][val]+'.npz'),
           rgb=rgb_hat,
           depth=norm_depth)
       


