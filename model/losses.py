import torch
from torch import nn
from torch.nn import functional as F
import pdb

class Loss(nn.Module):
    def __init__(self, full_weight, grad_weight, occ_prob_weight):
        super().__init__()
        self.full_weight = full_weight
        self.grad_weight = grad_weight
        self.occ_prob_weight = occ_prob_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) 
        rgb_loss_mse = self.l2_loss(rgb_values, rgb_gt) 

        return rgb_loss, rgb_loss_mse

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            if  torch.cuda.is_available():
                return torch.tensor(0.0).cuda().float()
            else:
                return torch.tensor(0.0).float()
        else:
            return diff_norm.mean()

    def forward(self, rgb_pred, rgb_gt, diff_norm):
        if torch.cuda.is_available():
            rgb_gt = rgb_gt.cuda()
        if self.full_weight != 0.0:
            rgb_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
            rgb_full_loss = rgb_loss[0]
            rgb_loss_mse = rgb_loss[1]
        else:
            if torch.cuda.is_available():
                rgb_full_loss = torch.tensor(0.0).cuda().float()
            else:
                rgb_full_loss = torch.tensor(0.0).float()
        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            if torch.cuda.is_available():
                grad_loss = torch.tensor(0.0).cuda().float()
            else:
                grad_loss = torch.tensor(0.0).float()

        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss
        if torch.isnan(loss):
            breakpoint()

        return {
            'loss': loss,
            'fullrgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
            'mse_loss': rgb_loss_mse
        }


