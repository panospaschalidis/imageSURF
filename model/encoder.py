"""standard libraries"""
import torch
import pdb
import torchvision.models as models
import torch.nn.functional as functional 

"""third party imports"""
from torch import nn

"""local specific imports"""

class Encoder(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.im_shape = torch.tensor(conf['scale_size'])
        self.conf = conf
        self.model = getattr(models, self.conf['type'])(pretrained=True)
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.model_modules = self.model._modules
        # condition to skip_first_pooling layer
        if self.conf['skip_pool']:
            self.model_modules.pop('maxpool')

    def vector_extractor(self, grid):
        # grid argument of torch.nn.functionals.grid_sample
        # function should contain values at most in teh range
        # [-1,1]. Thus, input grid needs normalization
        # Since grid values lie at most in the range defined by
        # self.im_shape, we can normalize grid values according 
        # to feature_volume dimensions and then shift them 
        #in the desired range

        # from slef.im_shape to [0,2]
        # division with self.f_vol_window-1 ensures different 
        # mapping for different window sizes or else it would 
        # be always 1. As window size increases pertinent ratio 
        # decreases, hence variance of input grid will be less 
        # increased since samllere windows need mappings that
        # lead to larger variance in order not to have overlapping
        # values
        # print(f"inside encoder.vector_extractor")
        # print(f"before variance_scale")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        variance_scale = self.f_vol_window/(self.f_vol_window-1)*2
        # print(f"after variance_scale")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        # by multiplying outr values that lie at most in the range
        # defined by self.im_shape with 2 and then dividing with 
        # self.im_shape we shift our values at most in the range
        # [0:2,0:2]
        # print(f"before grid_scale")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        grid_scale = variance_scale.to(grid.device)/self.im_shape.to(grid.device)
        # print(f"after grid_scale")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        # by subtracting 1 from the above we get the desired result
        grid = grid*grid_scale - 1
        # print(f"before functional grid")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        self.vectors = functional.grid_sample(
            self.feature_volume,
            grid,
            mode=self.conf['interpolation_mode'],
            padding_mode=self.conf['padding_mode'],
            align_corners=self.conf['align_corners']

        ).permute(0,2,3,1)

        # print(f"after functional grid")
        # print(f"{torch.cuda.memory_allocated(grid.device)/1.e+9}")
        
    def forward(self, x):
        feature_maps = []
        for module in self.model_modules:
            
            x =  getattr(self.model, module)(x)
            
            if module in self.conf['layers_for_sampling']:
                feature_maps.append(x)
        size = feature_maps[0].shape[-2:]
        for val,_ in enumerate(feature_maps):
            feature_maps[val] = functional.interpolate(
                feature_maps[val],
                size,
                mode=self.conf['upsampling_mode'],
                align_corners=self.conf['align_corners']
            )
        self.feature_volume = torch.cat(feature_maps, dim=1)
        self.f_vol_window = torch.tensor(self.feature_volume.shape[-2:])

