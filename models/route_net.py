'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch.nn.functional as F
import torch.nn.init as init
from models.base import *


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


############################################################################################################
# stage 1 train with uniformed sampling
# stage 2 train router and fix network
# stage 3 train network and fix router
############################################################################################################
class RTRouteBlock(nn.Module):

    def __init__(self, block, in_planes, planes, stride=1, downsample=None, option=3, uniform_sample=False,
                 freeze_gate=False, freeze_net=False):
        super(RTRouteBlock, self).__init__()
        self.freeze_gate = freeze_gate
        self.freeze_net = freeze_net
        self.option = option
        self.uniform_sample = uniform_sample
        self.block = block(in_planes, planes, stride, freeze_net=freeze_net, downsample=downsample)
        self.router = rtRouter(in_planes, option, freeze_gate=freeze_gate)
        self._flops = 0

    def reset_flops(self):
        self._flops = 0

    def forward(self, input, sample=None, prev=None, pprev=None):
        if self.uniform_sample:
            assert (not self.freeze_gate and not self.freeze_net)
            assert (sample is not None)
            assert (input.size(0) == sample.size(0))
            onehot = torch.zeros(sample.size(0), self.option, dtype=input.dtype, device=input.device)
            onehot = onehot.scatter_(1, sample, 1)
            onehot = onehot.view(onehot.size(0), onehot.size(1), 1, 1)
            out = self.block(input, onehot)
            ln, fea = None, None
        elif self.freeze_gate:
            ln, fea = self.router(input, prev, pprev)
            assert (not self.uniform_sample and not self.freeze_net)
            choice = torch.argmax(ln.squeeze(), dim=1, keepdim=True)
            onehot = torch.zeros(choice.size(0), self.option, dtype=choice.dtype, device=choice.device)
            onehot = onehot.scatter_(1, choice, 1)
            onehot = onehot.view(onehot.size(0), onehot.size(1), 1, 1)
            out = self.block(input, onehot)
        else:
            ln, fea = self.router(input, prev, pprev)
            assert (not self.uniform_sample and not self.freeze_gate)
            ln = ln.view(ln.size(0), ln.size(1), 1, 1)
            out = self.block(input, ln)

        if not self.training:
            self._flops += self.block._flops
            self._flops += self.router._flops
            self.block.reset_flops()
            self.router.reset_flops()

        return out, ln, fea


############################################################################################################
# Network
############################################################################################################

class RouteNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, freeze_gate=False,
                 uniform_sample=False, freeze_net=False):
        super(RouteNet, self).__init__()
        self.in_planes = 16
        self.num_class = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.uniform_sample = uniform_sample
        self.num_block = num_blocks

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.linear = nn.Linear(64, num_classes)

        self._flops = 0

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, freeze_gate, uniform_sample, freeze_net):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(RTRouteBlock(block, self.in_planes, planes, stride, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net))
            self.in_planes = planes * block.expansion
        return layers

    def reset_flops(self):
        self._flops = 0

    def append_flops(self, block):
        if not self.training:
            self._flops += block._flops
            block.reset_flops()

    def forward(self, x, sample=None):
        out = F.relu(self.bn1(self.conv1(x)))
        if not self.training:
            self._flops += count_conv2d(x, out, 3)
        lns = []
        if self.uniform_sample:
            for i, block in enumerate(self.layer1):
                out, ln, lc = block(out, sample[:, i, :])
            for i, block in enumerate(self.layer2):
                out, ln, lc = block(out, sample[:, i + self.num_block[0], :])
            for i, block in enumerate(self.layer3):
                out, ln, lc = block(out, sample[:, i + self.num_block[0] + self.num_block[1], :])
        else:
            for i, block in enumerate(self.layer1):
                if len(lns) == 0:
                    out, ln, lc = block(out)
                elif len(lns) == 1:
                    out, ln, lc = block(out, lns[-1])
                else:
                    out, ln, lc = block(out, lns[-1], lns[-2])
                lns.append(ln.squeeze())
                self.append_flops(block)
            for i, block in enumerate(self.layer2):
                out, ln, lc = block(out, lns[-1], lns[-2])
                lns.append(ln.squeeze())
                self.append_flops(block)
            for i, block in enumerate(self.layer3):
                out, ln, lc = block(out, lns[-1], lns[-2])
                lns.append(ln.squeeze())
                self.append_flops(block)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if not self.training:
            self._flops += count_linear(out, 64, self.num_class)
            sum_flops = self._flops
            self.reset_flops()
            return out, sum_flops

        elif self.uniform_sample:
            return out, None

        else:
            probs = torch.stack(lns, dim=0).permute(1, 0, 2)
            return out, probs


class RouteNetDeep(nn.Module):

    def __init__(self, block, layers, num_classes=1000, freeze_gate=False, uniform_sample=False, freeze_net=False):
        super(RouteNetDeep, self).__init__()

        self.in_planes = 64
        self.uniform_sample = uniform_sample
        self.layers = layers
        self.num_class = num_classes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._flops = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, freeze_gate=False, uniform_sample=False, freeze_net=False):

        layers = nn.ModuleList()

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample = None

        layers.append(RTRouteBlock(block, self.in_planes, planes, stride,
                                   downsample=downsample, freeze_gate=freeze_gate,
                                   uniform_sample=uniform_sample, freeze_net=freeze_net))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(RTRouteBlock(block, self.in_planes, planes, freeze_gate=freeze_gate,
                                       uniform_sample=uniform_sample, freeze_net=freeze_net))
        return layers

    def reset_flops(self):
        self._flops = 0

    def append_flops(self, block):
        if not self.training:
            self._flops += block._flops
            block.reset_flops()

    def forward(self, x, sample=None):
        # See note [TorchScript super()]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        lns, feas = [], []

        if self.uniform_sample:
            for i, block in enumerate(self.layer1):
                out, ln, fea = block(out, sample[:, i, :])
            for i, block in enumerate(self.layer2):
                out, ln, fea = block(out, sample[:, i + self.layers[0], :])
            for i, block in enumerate(self.layer3):
                out, ln, fea = block(out, sample[:, i + self.layers[0] + self.layers[1], :])
            for i, block in enumerate(self.layer4):
                out, ln, fea = block(out, sample[:, i + self.layers[0] + self.layers[1] + self.layers[2], :])
        else:
            for i, block in enumerate(self.layer1):
                if len(feas) == 0:
                    out, ln, fea = block(out)
                elif len(feas) == 1:
                    out, ln, fea = block(out, feas[-1])
                else:
                    out, ln, fea = block(out, feas[-1], feas[-2])
                lns.append(ln.squeeze())
                feas.append(fea)
                self.append_flops(block)
            for i, block in enumerate(self.layer2):
                out, ln, fea = block(out, feas[-1], feas[-2])
                lns.append(ln.squeeze())
                feas.append(fea)
                self.append_flops(block)
            for i, block in enumerate(self.layer3):
                out, ln, fea = block(out, feas[-1], feas[-2])
                lns.append(ln.squeeze())
                feas.append(fea)
                self.append_flops(block)
            for i, block in enumerate(self.layer4):
                out, ln, fea = block(out, feas[-1], feas[-2])
                lns.append(ln.squeeze())
                feas.append(fea)
                self.append_flops(block)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if not self.training:
            self._flops += count_linear(out, 64, self.num_class)
            sum_flops = self._flops
            self.reset_flops()
            return out, sum_flops
        elif self.uniform_sample:
            return out, None
        else:
            probs = torch.stack(lns, dim=0).permute(1, 0, 2)
            return out, probs


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetDeep(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetDeep, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        layers = nn.ModuleList()

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.in_planes, planes, stride, downsample))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return layers

    def forward(self, x):
        # See note [TorchScript super()]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def ResNet20(num_class):
    return ResNet(BasicBlock, [3, 3, 3], num_class)


def ResNet32(num_class):
    return ResNet(BasicBlock, [5, 5, 5], num_class)


def ResNet38(num_class):
    return ResNet(BasicBlock, [6, 6, 6], num_class)


def ResNet44(num_class):
    return ResNet(BasicBlock, [7, 7, 7], num_class)


def ResNet56(num_class):
    return ResNet(BasicBlock, [9, 9, 9], num_class)


def ResNet74(num_class):
    return ResNet(BasicBlock, [12, 12, 12], num_class)


def ResNet110(num_class):
    return ResNet(BasicBlock, [18, 18, 18], num_class)


def ResNet152(num_class):
    return ResNet(BasicBlock, [25, 25, 25], num_class)


def ResNet1202(num_class):
    return ResNet(BasicBlock, [200, 200, 200], num_class)


def ResNet50(num_class):
    return ResNetDeep(Bottleneck, [3, 4, 6, 3], num_class)


def ResNet101(num_class):
    return ResNetDeep(Bottleneck, [3, 4, 23, 3], num_class)


def test(net):
    # import numpy as np
    # block = RTRouteBlock(32, 64)
    return None
