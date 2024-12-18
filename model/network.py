"""standard libraries """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import time

"""local specific imports"""
from .encoder import Encoder
from .create_pixelnerf import Pixelnerf_creator
from .create_features_aggregator import Features_aggregator


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field
    
    Args:
        cfg (dict): network configs
    '''

    def __init__(self, cfg, **kwargs):
        super().__init__()
        out_dim = 4
        dim = 3
        self.num_layers = cfg['num_layers']
        self.hiden_size = cfg['hidden_dim']
        self.octaves_pe = cfg['octaves_pe']
        self.octaves_pe_views = cfg['octaves_pe_views']
        self.skips = cfg['skips']
        self.rescale = cfg['rescale']
        self.feat_size = cfg['feat_size']
        geometric_init = cfg['geometric_init'] 
        self.views_encoder = cfg['encoder']['use']
        bias = 0.6

        # init pe
        #dim_embed = dim*self.octaves_pe*2 + dim
        dim_embed_init = dim*self.octaves_pe*2 + dim
        dim_embed =  cfg['encoder']['channels']
        #dim_embed_view = dim + dim*self.octaves_pe_views*2 + dim + dim + self.feat_size
        if self.views_encoder:
            self.encoder =  Encoder(cfg['encoder'])
            
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)
        ### geo network
        dims_geo = [dim_embed_init+dim_embed]+ [ self.hiden_size if i in self.skips else self.hiden_size for i in range(0, self.num_layers-1)] + [self.feat_size+1]
        self.num_layers = len(dims_geo)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
                #out_dim = dims_geo[0]
                #out_dim = dims_geo[l + 1]
            else:
                out_dim = dims_geo[l + 1]

            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                    #torch.nn.init.constant_(lin.bias, 0.0)
                    #torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.softplus = nn.Softplus(beta=100)
        ## appearance network
        #dims_view = [dim_embed_view]+ [ self.hiden_size for i in range(0, 4)] + [3]
        ##dims_view = [dim_embed]+ [ self.hiden_size for i in range(0, 4)] + [3]
        #self.num_layers_app = len(dims_view)
        #for l in range(0, self.num_layers_app - 1):
        #    out_dim = dims_view[l + 1]
        #    lina = nn.Linear(dims_view[l], out_dim)
        #    lina = nn.utils.weight_norm(lina)
        #    setattr(self, "lina" + str(l), lina)
        self.pixelnerf = Pixelnerf_creator(cfg['pixelnerf']['color'])
        self.features_aggregator = Features_aggregator(cfg['pixelnerf']['features'])
        self.normals = cfg['pixelnerf']['features']['normals'] 
        self.sigmoid = nn.Sigmoid()
    
    
    def encode(self, data):
        self.encoder(data['images'].flatten(0,1))
        w2crotmatrix = data['poses'][...,:3,:3].permute(0,1,3,2)
        w2ctranslation = (-w2crotmatrix @ data['poses'][...,:3,3].unsqueeze(3))
        self.w2cpose = torch.cat((w2crotmatrix,w2ctranslation),dim =-1)        
        self.focal = data['focal']
        self.prcpal_p = data['prcpal_p']


    def infer_occ(self, p, feat_vector):
        p = self.transform_points(p/self.rescale)
        try:
            x = torch.cat((p, feat_vector), dim=-1)
        except RuntimeError:
            pdb.set_trace()
        x_init = x
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
    

    def features_pooler(self, points, view_dirs, feature_vectors):
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        view_dirs = self.transform_points_view(view_dirs)
        points = self.transform_points(points/self.rescale)
        pixel_input = torch.cat((points,view_dirs), dim=-1)
        pixel_input = pixel_input.unsqueeze(1).repeat_interleave(
            int(feature_vectors.shape[0]/pixel_input.shape[0]),
            dim=1
        ).flatten(0,1)
        features = self.features_aggregator(pixel_input, feature_vectors)
        return features


    def infer_app(self, occ_vectors, aggr_feature_vector):
       # occ_vectors = occ_vectors.unsqueeze(1).repeat_interleave(
       #     int(feature_vectors.shape[0]/occ_vectors.shape[0]),
       #     dim=1
       # ).flatten(0,1)
        feature_vectors = torch.cat((occ_vectors, aggr_feature_vector), dim=-1)
        color = self.pixelnerf(feature_vectors)
        return color


    def gradient(self, occ_p):
        with torch.enable_grad():
            p = occ_p[...,:3]
            feat_vector = occ_p[...,3:]
            p.requires_grad_(True)
            y = self.infer_occ(p, feat_vector)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients


    def forward(self, p, ray_d=None, val=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        #if val ==1:
         #   pdb.set_trace()
            #camera_cords = p.unsqueeze(1) @ self.w2cpose[...,:3].permute(0,1,3,2).unsqueeze(2)
            #camera_cords = camera_cords + self.w2cpose[...,3][:,:,None,None]
            #raster_cords = -camera_cords[...,:2]/camera_cords[...,2:] 
            #raster_cords[...,0] = raster_cords[...,0] * self.focal[:, None, None, None]
            #raster_cords[...,1] = raster_cords[...,1] *(- self.focal[:, None, None, None])
            #raster_cords = raster_cords + self.prcpal_p[:, None, None, None, :]
        camera_cords = p.unsqueeze(1) @ self.w2cpose[...,:3].permute(0,1,3,2)
        camera_cords = camera_cords + self.w2cpose[...,3][...,None,:]
        raster_cords = -camera_cords[...,:2]/camera_cords[...,2:] 
        raster_cords[...,0] = raster_cords[...,0] * self.focal[:, None, None]
        raster_cords[...,1] = raster_cords[...,1] *(- self.focal[:, None, None])
        raster_cords = raster_cords + self.prcpal_p[:, None, None, :]
        self.encoder.vector_extractor(raster_cords.flatten(0,1).unsqueeze(-2))
        encoder_vectors = self.encoder.vectors.squeeze(-2)
        feat_vector = self.features_pooler(p, ray_d, encoder_vectors)
        x = self.infer_occ(p, feat_vector)

        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif ray_d is not None:
            rgb = self.infer_app(x[...,1:].detach(), feat_vector)
            if return_addocc:
                if noise:
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
                else: 
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
            else:
                return rgb
        elif return_logits:
            return -1*x[...,:1]


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)
