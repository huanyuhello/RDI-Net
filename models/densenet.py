'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


############################################################################################################
# Gates
############################################################################################################

class GateI(nn.Module):
    def __init__(self, in_planes, options, beta=25):
        super(GateI, self).__init__()
        self.beta = beta
        assert (beta != 0)

        self.control = nn.Sequential(
            nn.Linear(in_planes, in_planes),
            nn.Linear(in_planes, options),
            nn.Softmax(1))

    def forward(self, x):
        out = F.avg_pool2d(x, x.size(2))
        out = out.view(out.size(0), -1)

        out = self.control(out)
        out = F.softmax(self.beta * out, dim=1)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return out


############################################################################################################
# Block with Gate
############################################################################################################

class GateBlockI(nn.Module):
    def __init__(self, Gate, in_planes, growth_rate, beta=25):
        super(GateBlockI, self).__init__()
        self.block = Bottleneck(in_planes, growth_rate)
        self.gate = Gate(in_planes, options=2, beta=beta)
        self.skip = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, growth_rate + in_planes, kernel_size=1, bias=False))

    def forward(self, input):
        gate = self.gate(input)
        out0 = self.block(input)
        out1 = self.skip(input)

        if self.training:
            out = out0 * gate[:, 0, :, :].unsqueeze(1) + out1 * gate[:, 1, :, :].unsqueeze(1)
        else:
            _, mask = gate.max(1)
            mask0 = torch.where(mask == 0, torch.full_like(mask, 1), torch.full_like(mask, 0)).float()
            mask1 = torch.where(mask == 1, torch.full_like(mask, 1), torch.full_like(mask, 0)).float()
            out = out0 * mask0.unsqueeze(1) + out1 * mask1.unsqueeze(1)
        return out

############################################################################################################
# Net work
############################################################################################################


class DenseSkipNet(nn.Module):
    def __init__(self, gate, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, beta=25):
        super(DenseSkipNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(gate, block, num_planes, nblocks[0], beta=beta)
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(gate, block, num_planes, nblocks[1], beta=beta)
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(gate, block, num_planes, nblocks[2], beta=beta)
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(gate, block, num_planes, nblocks[3], beta=beta)
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, gate, block, in_planes, nblock, beta=25):
        layers = []
        for i in range(nblock):
            layers.append(block(gate, in_planes, self.growth_rate, beta=beta))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNetSkip(num_classes=10, beta=25):
    return DenseSkipNet(GateI, GateBlockI, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes, beta=beta)


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


def test():
    net = densenet_cifar()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)

# test()
