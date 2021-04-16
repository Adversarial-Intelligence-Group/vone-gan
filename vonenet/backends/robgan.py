import torch
import torch.nn as nn
import torch.nn.functional as F
from .robgan_layers import UpBlock, DownBlock

class Generator(nn.Module):
    def __init__(self, out_ch = 3, ch=64, dim_z=128, bottom_width=4, activation=F.relu, \
            n_classes=0, distribution="normal"):
        super(Generator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        nn.init.xavier_uniform_(self.l1.weight, 1.0)
        self.block2 = UpBlock(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = UpBlock(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = UpBlock(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = UpBlock(ch * 2, ch * 1, activation=activation, upsample=False, n_classes=n_classes)
        self.b6 = nn.BatchNorm2d(ch)
        nn.init.constant_(self.b6.weight, 1.0) #XXX this is different from default initialization method
        self.l6 = nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l6.weight, 1.0)

    def forward(self, z, y):
        h = z
        h = self.l1(h)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = self.l6(h)
        h = torch.tanh(h)
        return h



class Discriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu, bn=False):
        super(Discriminator, self).__init__()
        self.activation = activation
        # self.block1 = OptimizedBlock(3, ch * 2, bn=bn)
        self.block2 = DownBlock(ch * 2, ch * 2, activation=activation, downsample=True, bn=bn)
        self.block3 = DownBlock(ch * 2, ch * 2, activation=activation, downsample=False, bn=bn)
        self.block4 = DownBlock(ch * 2, ch * 2, activation=activation, downsample=False, bn=bn)
        self.l5 = nn.Linear(ch * 2, 1)
        nn.init.xavier_uniform_(self.l5.weight, gain=1.0)
        if n_classes > 0:
            self.l_y = nn.Linear(ch * 2, n_classes)
            nn.init.xavier_uniform_(self.l_y.weight, gain=1.0)

    def forward(self, x):
        h = x
        # h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # global sum pooling (h, w)
        #TODO try to use global avg pooling instead
        h = h.view(h.size(0), h.size(1), -1)
        h = torch.sum(h, 2)
        output = self.l5(h)
        w_y = self.l_y(h)
        return output.view(-1), w_y

