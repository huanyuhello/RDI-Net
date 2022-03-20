# import torch
import torch.nn as nn
from utils.flops import count_conv2d, count_linear, count_bn, count_relu, count_softmax


class Conv2d(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride, padding=0, bias=False):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.ops = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x, flops=None):
        out = self.ops(x)
        if flops is not None:
            flops.append(count_conv2d(x, out, self.kernel_size))
        return out


class BatchNorm2d(nn.Module):
    def __init__(self, plane):
        super(BatchNorm2d, self).__init__()
        self.ops = nn.BatchNorm2d(plane)

    def forward(self, x, flops=None):
        out = self.ops(x)
        if flops is not None:
            flops.append(count_bn(x))
        return out


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.ops = nn.ReLU()

    def forward(self, x, flops=None):
        out = self.ops(x)
        if flops is not None:
            flops.append(count_relu(x))
        return out


class Softmax(nn.Module):
    def __init__(self, dim):
        super(Softmax, self).__init__()
        self.ops = nn.Softmax(dim)

    def forward(self, x, flops=None):
        out = self.ops(x)
        if flops is not None:
            flops.append(count_softmax(x))
        return out


class Linear(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Linear, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.ops = nn.Linear(in_planes, out_planes)

    def forward(self, x, flops=None):
        out = self.ops(x)
        if flops is not None:
            flops.append(count_linear(x, self.in_planes, self.out_planes))
        return out
