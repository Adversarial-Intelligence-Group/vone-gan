import numpy
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CondBatchNorm2d(nn.Module):
    def __init__(self, size, decay=0.9, eps=2.0e-5):
        super(CondBatchNorm2d, self).__init__()
        self.size = size
        self.eps = eps
        self.decay = decay
        self.register_buffer('avg_mean', torch.zeros(size))
        self.register_buffer('avg_var', torch.ones(size))
        self.register_buffer('gamma_', torch.ones(size))
        self.register_buffer('beta_', torch.zeros(size))

    def forward(self, x, gamma, beta):
        # Intentionally set self.weight == ones and self.bias == zeros
        # because we only want to normalize the input.
        feature = F.batch_norm(x, self.avg_mean, self.avg_var, Variable(self.gamma_), Variable(self.beta_), self.training, self.decay, self.eps)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        return gamma * feature + beta

class CatCondBatchNorm2d(CondBatchNorm2d):
    def __init__(self, size, n_cat, decay=0.9, eps=2.0e-5, initGamma=1.0, initBeta=0):
        super(CatCondBatchNorm2d, self).__init__(size, decay=decay, eps=eps)
        self.gammas = nn.Embedding(n_cat, size)
        nn.init.constant_(self.gammas.weight, initGamma)
        self.betas = nn.Embedding(n_cat, size)
        nn.init.constant_(self.betas.weight, initBeta)

    def forward(self, x, c):
        gamma_c = self.gammas(c)
        beta_c = self.betas(c)
        return super(CatCondBatchNorm2d, self).forward(x, gamma_c, beta_c)



def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, \
            pad=1, activation=F.relu, upsample=False, n_classes=0):
        super(UpBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, math.sqrt(2.0))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, math.sqrt(2.0))
        if n_classes > 0:
            self.b1 = CatCondBatchNorm2d(in_channels, n_cat=n_classes)
            self.b2 = CatCondBatchNorm2d(hidden_channels, n_cat=n_classes)
        else:
            self.b1 = nn.BatchNorm2d(in_channels)
            nn.init.constant_(self.b1.weight, 1.0)
            self.b2 = nn.BatchNorm2d(hidden_channels)
            nn.init.constant_(self.b2.weight, 1.0)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight, 1.0)

    def residual(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        f1 = self.residual(x, y)
        f2 = self.shortcut(x)
        return f1 + f2



def _downsample(x):
    return F.avg_pool2d(x, 2)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
            ksize=3, pad=1, activation=F.relu, downsample=False, bn=False):
        super(DownBlock, self).__init__()
        self.activation = activation
        self.bn = bn
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2.0))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2.0))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            nn.init.xavier_uniform_(self.c_sc.weight, gain=1.0)
        if self.bn:
            self.b1 = nn.BatchNorm2d(hidden_channels)
            nn.init.constant_(self.b1.weight, 1.0)
            self.b2 = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.b2.weight, 1.0)
            if self.learnable_sc:
                self.b_sc = nn.BatchNorm2d(out_channels)
                nn.init.constant_(self.b_sc.weight, 1.0)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.b1(self.c1(h)) if self.bn else self.c1(h)
        h = self.activation(h)
        h = self.b2(self.c2(h)) if self.bn else self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.b_sc(self.c_sc(x)) if self.bn else self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
