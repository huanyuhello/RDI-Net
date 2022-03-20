import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.flops import *
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from models.gcn_layer import GraphConvolution, LayerGraphConvolution, LayerGraph


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


############################################################################################################
# basic block
############################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class rtbBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, freeze_net=False, downsample=None):
        super(rtbBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        if freeze_net:
            for p in self.parameters():
                p.requires_grad = False

        self._flops = 0

    def reset_flops(self):
        self._flops = 0

    def forward(self, x, ln, lc=None):
        r0 = ln[:, 0, :, :].unsqueeze(1)
        r1 = ln[:, 1, :, :].unsqueeze(1)
        r2 = ln[:, 2, :, :].unsqueeze(1)
        conv1 = F.relu(self.bn1(self.conv1(x)))

        # todo lc mask on conv1 !!!

        if lc is not None:
            lc = lc.view(lc.size(0), lc.size(1), 1, 1)
            conv1 = lc * conv1

        conv2 = self.bn2(self.conv2(conv1))

        # [self.shortcut(x), conv1, conv2, (self.shortcut(x) + conv1), (self.shortcut(x) + conv2)]
        out = r0 * F.relu(self.shortcut(x)) + \
              r1 * F.relu(self.shortcut(x) + conv1) + \
              r2 * F.relu(self.shortcut(x) + conv2)

        ### Flops calc
        if not self.training:
            self._flops += sum(r1) / r1.size(0) * count_conv2d(x, conv1, 3)
            self._flops += sum(r2) / r2.size(0) * (count_conv2d(x, conv1, 3) + count_conv2d(conv1, conv2, 3))

        return out


class rtbBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, freeze_net=False, downsample=None):
        super(rtbBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.stride = stride
        self.downsample = downsample

        if stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None

        if freeze_net:
            for p in self.parameters():
                p.requires_grad = False

        self._flops = 0

    def reset_flops(self):
        self._flops = 0

    def forward(self, x, ln):
        identity = x

        r0 = ln[:, 0, :, :].unsqueeze(1)
        r1 = ln[:, 1, :, :].unsqueeze(1)
        r2 = ln[:, 2, :, :].unsqueeze(1)

        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))

        if self.maxpool is not None:
            conv1 = self.maxpool(conv1)

        conv3_1 = self.bn3(self.conv3(conv1))
        conv3_2 = self.bn3(self.conv3(conv2))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(identity + r1 * conv3_1 + r2 * conv3_2)

        ## Flops calc
        if not self.training:
            self._flops += (sum(r1) + sum(r2)) / r1.size(0) * count_conv2d(x, conv1, 1)
            self._flops += (sum(r1) + sum(r2)) / r1.size(0) * count_conv2d(conv2, conv3_2, 1)
            self._flops += sum(r2) / r2.size(0) * count_conv2d(conv1, conv2, 3)

        return out


############################################################################################################
# Gates
############################################################################################################
class rtRouter(nn.Module):
    def __init__(self, in_planes, option, freeze_gate=False, hard_sample=False):
        super(rtRouter, self).__init__()
        assert (option == 3)
        self.hard_sample = hard_sample
        self.in_planes = in_planes
        self.option = option
        self.fea_dim = 256
        self.layer0 = nn.Linear(in_planes, in_planes)
        self.laybn0 = nn.BatchNorm1d(in_planes)
        self.layer1 = nn.Linear(in_planes, option)
        self.gcn_layers = LayerGraph(in_features=option, out_features=self.fea_dim)
        self.layer2 = nn.Linear(self.fea_dim, option)

        if freeze_gate:
            for p in self.parameters():
                p.requires_grad = False
        self._flops = 0

    def reset_flops(self):
        self._flops = 0

    def forward(self, x, prev=None, pprev=None):
        out = F.avg_pool2d(x, x.size(2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.laybn0(self.layer0(out)))
        out = self.layer1(out)

        out = self.gcn_layers(out, prev, pprev)
        ln = self.layer2(out)
        # ln = out
        ln = F.dropout(ln, p=0.4, training=self.training)
        ln = F.gumbel_softmax(ln, hard=self.hard_sample)

        if not self.training:
            self._flops += count_linear(x, self.in_planes, self.in_planes)
            self._flops += count_linear(x, self.in_planes, self.option)

        return ln, ln


class UniformSample():

    def __init__(self, nblock, options):
        self.samplers = []

        for _ in range(nblock):
            uniform = Uniform(torch.tensor([0.0]), torch.tensor([0.0 + options]))
            self.samplers.append(uniform)

    def sample(self, bs):
        routes = []
        for sampler in self.samplers:
            routes.append(sampler.sample(sample_shape=(bs,)))

        routers = torch.stack(routes).long().permute(1, 0, 2)
        return routers


class NormalSample():

    def __init__(self, nblock):
        self.samplers = []

        for _ in range(nblock):
            uniform = Normal(loc=torch.tensor([3.0]), scale=torch.tensor([1.0]))
            self.samplers.append(uniform)

    def sample(self, bs):
        routes = []
        for sampler in self.samplers:
            routes.append(sampler.sample(sample_shape=(bs,)))

        routers = torch.stack(routes).long().permute(1, 0, 2)
        return routers
