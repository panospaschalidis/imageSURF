"""standard libraries"""
import os
import sys
import glob
import torch
import numpy as np
import pdb
import random

"""third party imports"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from multiprocessing import Manager

"""local specific imports"""
#from data.utils import csvreturn, ray_tracing, strf_sampling, random_indices


def get_dataloader(cfg, mode='train', shuffle=True):
    #manager = Manager()
    #shared_dict = manager.dict()
    path = os.path.join(
        cfg['dataloading']['path']+mode
    )
    batch_size = cfg['dataloading']['batchsize'][mode]
    n_workers = cfg['dataloading']['n_workers'][mode]
    shuffle = cfg['dataloading']['shuffle'][mode]
    #dataset = SRNDataset(path, mode, shared_dict=shared_dict)
    dataset = SRNDataset(path, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        drop_last=True
    )
    return dataloader


class SRNDataset(Dataset):

    def __init__(self, path, dat_type):#, shared_dict):
        #super().__init__()
        self.instance_paths = sorted(glob.glob(os.path.join(path,'*')))
        #random.seed(124)
        #pdb.set_trace()
        #self.instance_paths = random.sample(self.instance_paths, 15)
        #self.instance_paths = [self.instance_paths[123]]         
        # SRN authors create perinent dataset under specific conventions
        # reganding the camera coordinate systems. Specifically they consider
        # the yaxis pointing downwards and the zaxis away from world coordinate
        # system. In order to adjust the extracted poses to the standard camera
        # coordinate systems we define self._coord_trans that is going to 
        # transform the given poses
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        self.bbox = True
        # type of dataset train, val, test
        self.dat_type = dat_type
        # self.cached_dict = shared_dict

    def __len__(self):
        return len(self.instance_paths)

    def __getitem__(self,idx):
        
        #if 'item_'+str(idx) in self.cached_dict:
        #    print('cached')
        #    # return self.cached_dict['item_'+str(idx)]
        #else:
        instance_path = self.instance_paths[idx]
        data = np.load(instance_path)
        item = {}
        images = torch.tensor(data['rgb'][...,:3]).div(255).to(torch.float32)
        images = (images-0.5)/0.5
        nviews, H, W, _ = images.shape
        poses = torch.tensor(data['pose'], dtype=torch.float32) @ \
                self._coord_trans
        focal = torch.tensor(data['focal'], dtype=torch.float32)
        item['instance'] = os.path.basename(os.path.splitext(instance_path)[0])
        item['poses'] = poses
        item['focal'] = focal
        item['prcpal_p'] = torch.tensor([H/2, W/2])
        if self.dat_type == 'train':
            # bounding box estimation #
            # each instance consists of a number of images from 
            # various views. Each image contains an object in a
            # white (255) background, thus subtraction of image
            # with 255 
            im_index, rows, _ , _ = torch.where(images!=1)
            indices = im_index.unique(return_counts=True)[1]
            r_ind_max = indices.cumsum(dim=0)-1
            r_ind_min = torch.cat(
                (torch.tensor(0).unsqueeze(0),
                indices.cumsum(dim=0)[:-1])
            )
            # im_index corresponds to each image index inside pertinent instance
            # rows correpond to rows of elements that satisfy torch.where condition
            # since values are being returned in a row major order, rows vector 
            # across the different splits defined by indices contain sorted values,
            # hence all we to do is to extract first and last value from each split
            # in order to obtain min and max row value for each view bounding box 
            # in order to perform same operation for columns of bounding box we
            # traspose matrix images, in order to exploit the row major order 
            # previously mentioned for the columns
            im_index, cols, _ , _ = torch.where(images.permute(0,2,1,3)!=1)
            indices = im_index.unique(return_counts=True)[1]
            c_ind_max = indices.cumsum(dim=0)-1
            c_ind_min =torch.cat(
                (torch.tensor(0).unsqueeze(0),
                indices.cumsum(dim=0)[:-1])
            )
            item['bound_boxes'] = torch.cat( 
                (rows[r_ind_min][:,None],
                rows[r_ind_max][:,None],
                cols[c_ind_min][:,None],
                cols[c_ind_max][:,None]),
                dim=1
            )
        item['images'] = images.permute(0,3,1,2)
        # self.cached_dict['item_'+str(idx)] = item
        return item



