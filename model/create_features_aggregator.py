"""standard libraries"""
import torch
import numpy as np
import pdb
import time 

"""third party imports"""
from torch import nn

"""local specific imports"""
from .customresnet import LinearResnetBlock

class Features_aggregator(nn.Module):
    
    def __init__(self, conf):
        super().__init__()
        # input consists of points concatenated with encoding 
        # of points plus viewing directions
        self.inpt = \
            conf['inpt']*2*conf['octaves_pe'] + conf['inpt'] + \
            conf['inpt']*2*conf['octaves_pe_views'] + conf['inpt']
            
        if conf['normals']:
            self.inpt += 3
        self.outpt = conf['out']
        self.latent_dim = conf['hidden_dim']
        self.resnet_blocks = conf['res_blocks']
        self.mean_layer = conf['mean_layer']
        self.num_in_views =  conf['nviews']
        # pdb.set_trace()
        self.input_linear = nn.Linear(self.inpt, self.latent_dim)
        nn.init.constant_(self.input_linear.bias, 0)
        nn.init.kaiming_normal_(self.input_linear.weight, a=0, mode='fan_in')
        self.resnet = nn.ModuleList(
            [LinearResnetBlock(
                self.latent_dim,
                self.latent_dim)
                for i in range(self.resnet_blocks)
            ]
        )
        self.residual_linear = nn.ModuleList(
             [nn.Linear(
                 self.latent_dim, 
                 self.latent_dim)
                 for i in range(self.mean_layer)
             ]
        )
        for block in self.residual_linear:
            nn.init.constant_(block.bias, 0)
            nn.init.kaiming_normal_(block.weight, a=0, mode='fan_in')
        #self.out_linear = nn.Linear(self.latent_dim, self.outpt)
        #nn.init.constant_(self.out_linear.bias, 0)
        #nn.init.kaiming_normal_(self.out_linear.weight, a=0, mode='fan_in')
    
    def forward(self, inpt, encoder_vectors):
        out = self.input_linear(inpt)
        for ind in range(len(self.resnet)):
            residual = self.residual_linear[ind](encoder_vectors)
            out = out + residual
        if self.num_in_views>1:
            #time for pooling
            out = torch.mean(
                out.reshape(
                    int(out.shape[0]/self.num_in_views),
                    self.num_in_views,
                    *out.shape[1:]
                ),
                dim=1
            )
        #out = self.out_linear(out)  # from 512 to 256
        return out          

