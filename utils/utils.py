'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import datetime
import random
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .dist_utils import DistSummaryWriter, is_main_process
import pathspec,os

class LayerLoss(nn.Module):
    def __init__(self):
        super(LayerLoss, self).__init__()

    def forward(self, logits):
        return torch.nn.functional.smooth_l1_loss(logits, torch.zeros_like(logits))


class ChoiceAverageLoss(nn.Module):
    def __init__(self):
        super(ChoiceAverageLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduce=True, size_average=False)

    def forward(self, logits):
        logits = torch.sum(logits, dim=0)
        return self.kl_div(logits, torch.zeros_like(logits)) / logits.size(0)


class HightLightMaxLoss(nn.Module):
    def __init__(self):
        super(HightLightMaxLoss, self).__init__()

    def forward(self, logits):
        return


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_system(args):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_%s_%s' % (args.model, args.dataset)

    save_path = os.path.join(args.log_path, now + hyper_param_str + '_' + args.note)

    logger = DistSummaryWriter(save_path)

    config_txt = os.path.join(save_path, 'args')

    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(args))
        cp_projects(save_path)

    return logger, save_path


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def load_model(path, rnet):
    source_state = torch.load(path)['state_dict']
    target_state = rnet.state_dict()
    for k, v in source_state.items():
        if 'router' not in k:
            target_state[k[7:]] = v
    return target_state


def get_blocks(backbone):

    if backbone == 'resnet20':
        blocks = [3, 3, 3]
    elif backbone == 'resnet32':
        blocks = [5, 5, 5]
    elif backbone == 'resnet38':
        blocks = [6, 6, 6]
    elif backbone == 'resnet44':
        blocks = [7, 7, 7]
    elif backbone == 'resnet56':
        blocks = [9, 9, 9]
    elif backbone == 'resnet74':
        blocks = [12, 12, 12]
    elif backbone == 'resnet110':
        blocks = [18, 18, 18]
    elif backbone == 'resnet50':
        blocks = [3, 4, 6, 3]
    else:
        blocks = None

    return blocks


def cp_projects(to_path):

    with open('./.gitignore','r') as fp:
        ign = fp.read()
    ign += '\n.git'
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
    all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
    matches = spec.match_files(all_files)
    matches = set(matches)
    to_cp_files = all_files - matches

    for f in to_cp_files:
        dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))


def path2num(path):
    longint = 0
    for p in path:
        assert p == 1 or p == 0
        longint += p
        longint = longint << 1
    return longint
def tensor_path2nums(tensor):
    #tensor block * batch * 2 * 1 * 1
    tensor = torch.argmax(tensor,dim=2)
    tensor = tensor.data.cpu().numpy().squeeze().T
    return [path2num(p) for p in tensor]

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
