"""standard libraries"""
import torch
import numpy as np
import pdb
import time

"""third party imports"""
from torch import nn

"""local specific imports"""


class LinearResnetBlock(nn.Module):

    def __init__(self, inpt, outpt):
        super().__init__()
        self.linear_0 = nn.Linear(inpt, outpt)
        nn.init.constant_(self.linear_0.bias, 0)
        nn.init.kaiming_normal_(self.linear_0.weight, a=0, mode='fan_in')
        self.linear_1 = nn.Linear(inpt, outpt)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.zeros_(self.linear_1.weight)

    def forward(self, batch):
        out = self.linear_0(torch.relu(batch))
        out = self.linear_1(torch.relu(out)) + batch
        return out
