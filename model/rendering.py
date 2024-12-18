import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from .common import (
    get_mask, image_points_to_world, origin_to_world, best_views_selector)
import pdb
import time
import torch.nn.functional as functional


epsilon = 1e-6
class Renderer(nn.Module):
    ''' Renderer class containing unisurf
    surf rendering and phong rendering(adapted from IDR)
    
    Args:
        model (nn.Module): model
        cfg (dict): network configs
        model_bg (nn.Module): model background (coming soon)
    '''

    def __init__(self, model, cfg, device=None,
                 model_bg=None, **kwargs):
        super().__init__()
        self._device = device
        self.n_max_network_queries = cfg['n_max_network_queries']
        self.white_background = cfg['white_background']
        self.cfg=cfg
        self.depth_range = self.cfg['depth_range'] #[2, 6]
        self.nviews = self.cfg['encoder']['nviews'] 
        self.encoder = self.cfg['encoder']['use']

        self.model = model.to(device)
        if model_bg is not None:
            self.model_bg = model.to(device)
        else:
            self.model_bg = None


    def forward(self, rays_dir, rays_origin,
                rendering_technique, val=None, encoding_dict=None, 
                add_noise=False, eval_=False, mask=None, it=0):
        self.val = it
        
        if rendering_technique == 'imagesurf':
            out_dict = self.imagesurf(
                rays_dir, rays_origin, encoding_dict, 
                it=it, add_noise=add_noise, eval_=eval_
            )
        else:
            raise NotImplementedError
        return out_dict
        
    def imagesurf(self, rays_dir, rays_origin, encoding_dict,
                add_noise=False, it=100000, eval_=False):
        # Get configs
        batch_size, n_points, _ = rays_dir.shape
        device = self._device
        rad = self.cfg['radius']
        ada_start = self.cfg['interval_start']
        ada_end = self.cfg['interval_end']
        ada_grad = self.cfg['interval_decay']
        steps = self.cfg['num_points_in']
        steps_outside = self.cfg['num_points_out']
        ray_steps = self.cfg['ray_marching_steps']

        depth_range = torch.tensor(self.depth_range)
        n_max_network_queries = self.n_max_network_queries
        
        # Prepare camera projection
        camera_world = rays_origin
        ray_vector = rays_dir
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)
        # Get sphere intersection
        #depth_intersect,_ = get_sphere_intersection(
        #    camera_world[:,0], ray_vector, r=rad
        #)
        depth_intersect = torch.tensor(self.depth_range, device=self._device)[None,None,:].\
            repeat_interleave(ray_vector.shape[1], dim =1).\
            repeat_interleave(ray_vector.shape[0], dim=0).float()
        #print('before ray_marching call')
        #print(time.time())
        # Find surface
        with torch.no_grad():
            d_i = self.ray_marching(
                camera_world, ray_vector, self.model, it,
                n_secant_steps=8, 
                n_steps=[int(ray_steps),int(ray_steps)+1],
                depth_range=depth_range,
                rad=rad
            )
        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0
        d_i = d_i.detach()
        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()
        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device)
            dists[mask_pred] = d_i[mask_pred]
            dists[mask_zero_occupied] = 0
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask
            dists = dists
        # Project depth to 3d poinsts
        camera_world = camera_world.reshape(-1, 3)
        ray_vector = ray_vector.reshape(-1, 3)
        points = camera_world + ray_vector * dists.reshape(-1,1)
        points = points.view(-1,3)
        # Define interval
        #depth_intersect[:,:,0] = torch.Tensor([0.0]).to(device) 

        dists_intersect = depth_intersect.reshape(-1, 2)
        d_inter = dists[network_object_mask]
        d_sphere_surf = dists_intersect[network_object_mask.reshape(-1),:][:,1]
        delta = torch.max(ada_start * torch.exp(-1 * ada_grad * it * torch.ones(1)),\
             ada_end * torch.ones(1)).to(device)
        dnp = d_inter - delta
        dfp = d_inter + delta
        dnp = torch.where(dnp < depth_range[0].float().to(device),\
            depth_range[0].float().to(device), dnp)
        dfp = torch.where(dfp >  d_sphere_surf,  d_sphere_surf, dfp)
        # it > 153500 corresponds to epoch > 250
        if (dnp!=0.0).all() and (d_inter!=np.inf).all() and it > 153500:
            #print('all steps included')
            full_steps = steps+steps_outside
        else:
            full_steps = steps

        d_nointer = dists_intersect[~network_object_mask.reshape(-1),:]
        d2 = torch.linspace(0., 1., steps=full_steps, device=device)
        d2 = d2.view(1, 1, -1).repeat(1, d_nointer.shape[0], 1)
        d2 = depth_range[0] * (1. - d2) + d_nointer[:,1].view(1, -1, 1)* d2
        if add_noise:
            di_mid = .5 * (d2[:, :, 1:] + d2[:, :, :-1])
            di_high = torch.cat([di_mid, d2[:, :, -1:]], dim=-1)
            di_low = torch.cat([d2[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(d2.shape[0], full_steps, device=device)
            d2 = di_low + (di_high - di_low) * noise 
        p_noiter = camera_world[~network_object_mask.reshape(-1),:].unsqueeze(-2) \
            + ray_vector[~network_object_mask.reshape(-1),:].unsqueeze(-2) * d2.squeeze().unsqueeze(-1)
        #p_noiter = p_noiter.reshape(-1, 3)
        # Sampling region with surface intersection        
        d_interval = torch.linspace(0., 1., steps=steps, device=device)
        d_interval = d_interval.view(1, 1, -1).repeat(1, d_inter.shape[0], 1)        
        d_interval = (dnp).view(1, -1, 1) * (1. - d_interval) + (dfp).view(1, -1, 1) * d_interval

        if full_steps != steps:
            d_binterval = torch.linspace(0., 1., steps=steps_outside, device=device)
            d_binterval = d_binterval.view(1, 1, -1).repeat(1, d_inter.shape[0], 1)
            d_binterval =  depth_range[0] * (1. - d_binterval) + (dnp).view(1, -1, 1)* d_binterval
            d1,_ = torch.sort(torch.cat([d_binterval, d_interval],dim=-1), dim=-1)
        else:
            d1 = d_interval

        if add_noise:
            di_mid = .5 * (d1[:, :, 1:] + d1[:, :, :-1])
            di_high = torch.cat([di_mid, d1[:, :, -1:]], dim=-1)
            di_low = torch.cat([d1[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(d1.shape[0], full_steps, device=device)
            d1 = di_low + (di_high - di_low) * noise 
        
        p_iter = camera_world[network_object_mask.reshape(-1),:].unsqueeze(-2)\
             + ray_vector[network_object_mask.reshape(-1),:].unsqueeze(-2) * d1.unsqueeze(-1)
        
        # Merge rendering points
        p_fg = torch.zeros(batch_size* n_points, full_steps, 3, device=device)
        p_fg[~network_object_mask.reshape(-1),:] =  p_noiter.view(-1, full_steps,3)
        p_fg[network_object_mask.reshape(-1),:] =  p_iter.view(-1, full_steps,3)
        p_fg = p_fg.reshape(batch_size, n_points,full_steps,-1)
        #p_fg = p_fg.reshape(-1, 3)
        p_fg = p_fg.flatten(1,2)
        #ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
        #ray_vector_fg = -1*ray_vector_fg.reshape(-1, 3)
        ray_vector_fg = ray_vector.reshape(batch_size, n_points,-1).unsqueeze(-2)
        ray_vector_fg = ray_vector_fg.repeat_interleave(full_steps, dim =-2)
        ray_vector_fg = ray_vector_fg.flatten(1,2)
        #ray_vector_fg = ray_vector_fg.flatten(1,2)
        # Run Network
        noise = not eval_
        rgb_fg, logits_alpha_fg = [], []
        for i in range(0, p_fg.shape[1], n_max_network_queries):
            rgb_i, logits_alpha_i = self.model(
                p_fg[:,i:i+n_max_network_queries], 
                ray_vector_fg[:,i:i+n_max_network_queries],
                return_addocc=True, noise=noise, val =self.val
            )
            rgb_fg.append(rgb_i)
            logits_alpha_fg.append(logits_alpha_i)
        depth = torch.zeros(batch_size*n_points, full_steps, device=device)
        depth[~network_object_mask.reshape(-1),:] = d2.squeeze()
        depth[network_object_mask.reshape(-1),:] = d1.squeeze()
        depth = depth.reshape(batch_size, n_points,-1) # b_sxn_pointsxfull_steps
        #print('after model')
        #print(time.time())
        rgb_fg = torch.cat(rgb_fg, dim=1) # b_sx(n_points*full_steps)x3
        logits_alpha_fg = torch.cat(logits_alpha_fg, dim=1) # b_sx(n_points*full_steps)
        rgb = rgb_fg.reshape(batch_size,n_points, full_steps, 3)
        alpha = logits_alpha_fg.view(batch_size,n_points, full_steps)
        rgb = rgb
        weights = alpha * torch.cumprod(torch.cat([torch.ones((*alpha.shape[:2], 1), device=device), 1.-alpha + epsilon ], -1), -1)[..., :-1]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        depth_final = torch.sum(weights*depth,dim=-1)
        if not eval_:
            surface_mask = network_object_mask.view(-1)
            surface_points = points[surface_mask]
            surface_cam = camera_world[surface_mask]
            N = surface_points.shape[0]
            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01      
            #pp = torch.cat([surface_points, surface_points_neig], dim=0)
            #surface_cam = torch.cat([surface_cam, surface_cam], dim=0)
            #net_mask = torch.cat([network_object_mask, network_object_mask],dim=-1)
            #dirs = (pp-surface_cam)/torch.norm(pp-surface_cam, dim=-1)[...,None]
            if len(surface_points)>0:
                dirs = (surface_points-surface_cam)/torch.norm(surface_points-surface_cam, dim=-1)[...,None]
                if torch.isnan(torch.sum(dirs)):
                    pdb.set_trace()
                surf_features = self.feature_cat(
                    surface_points.unsqueeze(0), 
                    dirs.unsqueeze(0), 
                    network_object_mask
                )
                dirs = (surface_points_neig-surface_cam)/torch.norm(surface_points_neig-surface_cam, dim=-1)[...,None]
                surf_features_neig = self.feature_cat(
                    surface_points_neig.unsqueeze(0), 
                    dirs.unsqueeze(0), 
                    network_object_mask
                )
                pp = torch.cat([surf_features, surf_features_neig], dim=0)
                g = self.model.gradient(pp)
                normals_ = g / (g.norm(2, dim=1).unsqueeze(-1) + 10**(-5))
                diff_norm =  torch.norm(normals_[:N] - normals_[N:], dim=-1)
                if torch.isnan(torch.sum(diff_norm)):
                    pdb.set_trace()
            else:
                diff_norm = None
        else:
            surface_mask = network_object_mask
            diff_norm = None
        if self.white_background:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map.unsqueeze(-1))
        out_dict = {
            'rgb': rgb_values,
            'mask_pred': network_object_mask,
            'normal': diff_norm,
            'depth': depth_final
        }
        return out_dict


    def ray_marching(self, ray0, ray_direction, model, it, c=None,
                             tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                             depth_range=[0.8, 1.8], max_points=3500000, rad=1.0):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        tau = 0.5
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()
        #depth_intersect, _ = get_sphere_intersection(ray0[:,0], ray_direction, r=rad)
        depth_intersect = torch.tensor(self.depth_range, device=device)[None,None,:].\
            repeat_interleave(n_pts, dim =1).repeat_interleave(batch_size, dim=0).float()
        d_intersect = depth_intersect[...,1]            
        
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(batch_size, n_pts, 1,1)* d_proposal
        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal
        #p_proposal = torch.load('/home/panagiotis/workstation/repos/unisurf-main/dict.pt')
        #p_proposal = torch.rand(*p_proposal.shape)
        #print('before model call inside ray_marching')
        #print(time.time())
        # Evaluate all proposal points in parallel\
        ray_dirs = ray_direction.unsqueeze(-2).expand(p_proposal.shape)
        with torch.no_grad():
            val = torch.cat([(
                self.model(p_split, ray_dir, val =self.val, only_occupancy=True) - tau)
                for p_split, ray_dir in zip(
                                        torch.split(
                                            p_proposal.reshape(batch_size, -1, 3),
                                            int(max_points / batch_size), dim=1),
                                        torch.split(
                                            ray_dirs.reshape(batch_size, -1, 3),
                                            int(max_points / batch_size), dim=1))], 
                            dim=1).view(batch_size, -1, n_steps)
        #print('after model call inside ray_marching')
        #print(time.time())
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 
        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        # Apply surface depth refinement step (e.g. Secant method)
        if len(f_low)>0:
            d_pred = self.secant(
                f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
                ray_direction_masked, tau, mask)
        else:
            d_pred=f_low
        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out

    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, mask):
        ''' Runs the secant method for interval [d_low, d_high].


            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        # in case no spaces of interest were pinpointed in ray marching 
        # f_low,...,d_high will be of type tensor([], dtype=torch.int64)
        # hence pertinent function is being conditioned to the two possible 
        # cases. Under first case self.model will be called with features = False
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                if len(p_mid)>0:
                    vectors = self.feature_cat(p_mid.unsqueeze(0),ray_direction_masked.unsqueeze(0), mask)
                    f_mid = torch.sigmoid(self.model.infer_occ( 
                            vectors[:,:3], 
                            vectors[:,3:])[...,0]*-10)-tau
                else: 
                    pdb.set_trace()
                    f_mid = self.model(p_mid,  batchwise=False,
                       only_occupancy=True,)[...,0] - tau
            ind_low = f_mid < 0
            ind_low = ind_low
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def transform_to_homogenous(self, p):
        device = self._device
        batch_size, num_points, _ = p.size()
        r = torch.sqrt(torch.sum(p**2, dim=2, keepdim=True))
        p_homo = torch.cat((p, torch.ones(batch_size, num_points, 1).to(device)), dim=2) / r
        return p_homo


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    
    def feature_cat(self, surf_points, ray_dirs, network_object_mask):
        batch_size, n_points = network_object_mask.shape
        splits = torch.sum(network_object_mask, dim=1).tolist()
        # assuming batch size 3 instances
        # full_splits corresponds to the list
        # surf_points [from 1st - from 2nd - from 3rd - from 1st ...] instance
        # surf_points shape -> 2xsurf_points_for_whole_batchx3   
        inter_splits = torch.split(surf_points, splits, dim=1) 
        feature_volume = self.model.encoder.feature_volume
        ind = [l for l in range(batch_size)]

        # surf_points -> 2 x num_of_surf_points x3
        # inter_splits -> 2 x split x 3
        variance_scale = self.model.encoder.f_vol_window/(self.model.encoder.f_vol_window-1)*2
        grid_scale = (variance_scale/self.model.encoder.im_shape).to(surf_points.device)

        feature_vectors = []
        raster_space = []
        for val in range(len(splits)):
            camera_cords = inter_splits[val] @ self.model.w2cpose[val,...,:3].permute(0,2,1)
            camera_cords = camera_cords + self.model.w2cpose[val,...,3].unsqueeze(-2)
            raster_cords = -camera_cords[...,:2]/camera_cords[...,2:] 
            raster_cords[...,0] = raster_cords[...,0] * self.model.focal[-1]
            raster_cords[...,1] = raster_cords[...,1] *(- self.model.focal[-1])
            raster_cords = raster_cords + self.model.prcpal_p[-1]
            grid = raster_cords*grid_scale - 1
            vectors = functional.grid_sample(
                feature_volume[val,...].unsqueeze(0),
                grid.unsqueeze(0),
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(0).permute(1,2,0)
            feature_vectors.append(vectors)

        feature_vectors_0 = torch.cat(feature_vectors, dim=1)

        feature_vectors = self.model.features_pooler(surf_points, ray_dirs, feature_vectors_0)
        if torch.isnan(torch.sum(feature_vectors)):
            pdb.set_trace()
        output = torch.cat((surf_points, feature_vectors), dim=-1).flatten(0,1)
        return output
