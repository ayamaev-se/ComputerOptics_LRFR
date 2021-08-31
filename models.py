import torch
from torch import nn
from torch.nn import *
from torch.jit import *

import torch.nn as nn
import torch.nn.functional as F
# from torchaudio.functional import lfilter as torch_lfilter

from torch.autograd import Function, gradcheck
import matplotlib.pyplot as plt
from torch_radon import Radon

class UNetBlock(Module):

    def __init__(self, in_ch, ch, out_ch, layers):
        super().__init__()

        self.layers = layers
        self.maxpool = nn.AvgPool2d((2,2))
        self.unpool = nn.Upsample(scale_factor=2)

        self.up = []
        for i in range(0, self.layers):
            self.up.append(Sequential(Conv2d(ch, ch, (3,3), padding=(1,1), bias=False), PReLU(ch), Conv2d(ch, ch, (3,3), padding=(1,1), bias=False), PReLU(ch)))

        self.up = ModuleList(self.up)

        self.down = []
        for i in range(0, self.layers):
            self.down.append(Sequential(Conv2d(ch * 2, ch, (3,3), padding=(1,1), bias=False), PReLU(ch), Conv2d(ch, ch, (3,3), padding=(1,1), bias=False), PReLU(ch)))
        self.down = ModuleList(self.down)

        self.cross = Sequential(Conv2d(ch, ch, (3,3), padding=(1,1), bias=False), PReLU(ch), Conv2d(ch, ch, (3,3), padding=(1,1), bias=False), PReLU(ch))

        self.end_layer = Conv2d(ch, out_ch, (1,1), bias=False)
        self.start_layer = Conv2d(in_ch, ch, (3, 3), padding=(1,1), bias=False)
        
    def forward(self, x):
        
        l = {}

        t = self.start_layer(x)
        for i in range(0, self.layers):
            l[i] = self.up[i](t)
            t = l[i]
            t = self.maxpool(t)

        t = self.cross(t)

        t = self.unpool(t)

        for i in range(self.layers - 1, 0, -1):
            t = self.down[i](torch.cat([l[i], t], dim=1))
            t = self.unpool(l[i])

        t = self.down[0](torch.cat([l[0], t], dim=1))

        return self.end_layer(t)
