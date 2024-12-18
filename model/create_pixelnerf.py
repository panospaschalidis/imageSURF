"""standard libraries"""
import torch
import numpy as np
import pdb
import time 

"""third party imports"""
from torch import nn

"""local specific imports"""
from .customresnet import LinearResnetBlock

class Pixelnerf_creator(nn.Module):
    
    def __init__(self, conf):
        super().__init__()
        # input consists of points concatenated with encoding 
        # of points plus viewing directions
        self.inpt = conf['inpt']
        self.outpt = conf['out']
        self.latent_dim = conf['hidden_dim']
        self.resnet_blocks = conf['res_blocks']
        self.resnet = nn.ModuleList(
            [LinearResnetBlock(
                self.latent_dim,
                self.latent_dim)
                for i in range(self.resnet_blocks)
            ]
        )
        self.output_linear = nn.Sequential(
          nn.Linear(self.latent_dim, self.outpt),
          nn.Sigmoid()
        )
        nn.init.constant_(self.output_linear[0].bias, 0)
        nn.init.kaiming_normal_(self.output_linear[0].weight, a=0, mode='fan_in')
        # we further add a sigmoid module in order to limit color values 
        # in range 0, 1
        # self.sigmoid = nn.Sigmoid()
    

    def forward(self, inpt):
        out = inpt
        for ind in range(len(self.resnet)):
            out = self.resnet[ind](out)
        rgb = self.output_linear(out)
        return rgb           

