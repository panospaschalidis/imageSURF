import os
import torch
from collections import defaultdict
from model.common import (
    ray_tracing, bbox_sampling, get_tensor_values, sample_patch_points, arange_pixels,
    lego_sample_patch_points
)
from tqdm import tqdm
import logging
from model.losses import Loss
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
import pdb
import time

class Trainer(object):
    ''' Trainer object for the UNISURF.

    Args:
        model (nn.Module): model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): config file
        device (device): pytorch device
    '''

    def __init__(self, renderer, optimizer, cfg, device=None, **kwargs):
        self.renderer = renderer
        self.optimizer = optimizer
        self.device = device
        self.n_training_points = cfg['n_training_points']
        self.n_eval_points = cfg['n_training_points']
        self.overwrite_visualization = True

        self.rendering_technique = cfg['type']

        self.loss = Loss(
            cfg['lambda_l1_rgb'], 
            cfg['lambda_normals'],
            cfg['lambda_occ_prob']
        )
        self.encode = cfg['encoder']['use']
        self.nviews = cfg['encoder']['nviews']
        
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        
        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.renderer.train()
        self.optimizer.zero_grad()
        #print('before self.compute_loss-train_step')
        #print(time.time())
        loss_dict = self.compute_loss(data, it=it)
        #print('after self.compute_loss-train_step')
        #print(time.time())
        loss = loss_dict['loss']
        loss.backward()
        #print('after loss backward')
        #print(time.time())
        self.optimizer.step()
        return loss_dict

    #def eval_step(self, data):
    def eval_step(self, data, it=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.renderer.eval()
        eval_dict = {}
        #with torch.no_grad():
        try:
            with torch.no_grad():
                eval_dict = self.compute_loss(
                       data, eval_mode=True, it=it)
        except Exception as e:
            print(e)

        for (k, v) in eval_dict.items():
            eval_dict[k] = v.item()

        return eval_dict
    
    def render_visdata(self, data, resolution, it, out_render_path):
        device = self.device 
        batch_size, nviews, *im_shape = data['images'].shape
        rays_dir, rays_origin = ray_tracing(
            im_shape, 
            data['focal'][-1], 
            data['poses']
        )
        #torch.manual_seed(2)
        index = torch.randint(nviews, (1,))
        #print(index)
        #bsx(H*W)xnviewsx3->bsx(H*W)x1x3->bsx(H*W)x3
        rays_dir = rays_dir[:,:,index,:].squeeze(-2).to(device)
        #rays_origin = rays_origin.unsqueeze(1).repeat_interleave(np.prod([*im_shape[1:]]),dim=1)
        rays_origin = rays_origin.unsqueeze(1).repeat_interleave(np.prod(im_shape[1:]),dim=1)[:,:,index,:].squeeze(-2).to(device)
        rays_dir_spl = torch.split(rays_dir,2000, dim =1)
        rays_ori_spl = torch.split(rays_origin,2000, dim =1)
        rgb = []
        depth = [] 
        if self.encode:
            encoding_dict = {
                'images': data['images'],
                'poses': data['poses'],
                'focal':data['focal'],
                'prcpal_p': data['prcpal_p']
            } 
            views = torch.cat(
                [torch.randperm(encoding_dict['poses'].shape[1])[:self.nviews][None,...]
                    for i in range(batch_size)])
            encoding_dict['poses'] = encoding_dict['poses'][torch.arange(batch_size)[:,None],views,...][...,:3,:]#b_sxviewsx3x4
            encoding_dict['images'] = encoding_dict['images'][torch.arange(batch_size)[:,None],views,...]#b_Sxviewsx3xHXW
            for key, value in encoding_dict.items():
                encoding_dict[key] = value.to(self.device)
            self.renderer.model.encode(encoding_dict)
        with torch.no_grad():
            for val in range(len(rays_dir_spl)):
              #print(f"{val} from {len(rays_dir_spl)}")
              #print(rays_dir_spl[val].shape)
              #if val==9:
               # pdb.set_trace()
              pred = self.renderer(
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
                
        rgb_pred = torch.cat(rgb, dim=1).cpu()
        depth_pred = torch.cat(depth, dim=1).cpu()
        rgb_hat = rgb_pred.detach().cpu().numpy()
        depth_hat = depth_pred.detach().cpu().numpy()
        #rgb_hat = (rgb_hat * 255).astype(np.uint8)
        #img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        # since p_loc1 correspond to coordinates p_loc1 to
        # p_loc1[...,1] to x and p_loc[...,0] to y respectively
        # (inverse effect prompts from not not using indexing ='xy' 
        # in torch.meshgrid)
        # coordinates in a cartesian plane and image coordinates in
        # matrix form are inverted, therefore
        rgb_hat = rgb_hat.squeeze().transpose(1,0).reshape(im_shape).transpose(1,2,0)
        depth_hat = (depth_hat-np.min(depth_hat))/(np.max(depth_hat)-np.min(depth_hat))
        depth_hat =  depth_hat.squeeze().reshape(im_shape[1:])
        target = (0.5*(data['images'].squeeze()[index,...].squeeze().permute(1,2,0))+0.5).numpy()
        return rgb_hat, depth_hat, target

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
       
        img = data.get('img.').to(device)
        img_idx = data.get('img.idx')
        batch_size, _, h, w = img.shape
        mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        # focal is not stored to self.device in order to perform computations inside
        # lego_sample_patch_points
        focal = data.get('img.focal')
        return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, focal)

    def compute_loss(self, data, eval_mode=False, it=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        n_points = self.n_eval_points if eval_mode else self.n_training_points
        #(img, mask_img, world_mat, camera_mat, scale_mat, img_idx, focal) = self.process_data_dict(data)
        # Shortcuts
        device = self.device
        batch_size, _, _, h, w = data['images'].shape
        im_shape = data['images'].shape[-3:]
        rays_dir, rays_origin = ray_tracing(
            im_shape, 
            data['focal'][-1], 
            data['poses']
        )
        if not 'bound_boxes' in data or it>307000:
            data['bound_boxes'] = None
        rays_dir, rays_origin, rgb_gt = bbox_sampling(
            rays_dir, 
            rays_origin, 
            data['bound_boxes'],
            data['images'],
            self.n_training_points
        )
        rays_dir = rays_dir.to(device)
        rays_origin = rays_origin.to(device)
        rgb_gt = rgb_gt.to(device)
        if self.encode:
            encoding_dict = {
                'images': data['images'],
                'poses': data['poses'],
                'focal':data['focal'],
                'prcpal_p': data['prcpal_p']
            }
            views = torch.cat(
                [torch.randperm(encoding_dict['poses'].shape[1])[:self.nviews][None,...]
                    for i in range(batch_size)])
            encoding_dict['poses'] = encoding_dict['poses'][torch.arange(batch_size)[:,None],views,...][...,:3,:]#b_sxviewsx3x4
            encoding_dict['images'] = encoding_dict['images'][torch.arange(batch_size)[:,None],views,...]#b_Sxviewsx3xHXW
            for key, value in encoding_dict.items():
                encoding_dict[key] = value.to(self.device)
            self.renderer.model.encode(encoding_dict)
        
        out_dict = self.renderer(
            rays_dir, rays_origin, 
            self.rendering_technique,
            it=it, mask=None, eval_=eval_mode
        )
        loss_dict = self.loss(out_dict['rgb'], rgb_gt, out_dict['normal'])
        return loss_dict
