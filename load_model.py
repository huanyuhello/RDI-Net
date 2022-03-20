import os
import re
import torch
from utils.utils import load_model
from models.route_net import RouteNet, RouteNetDeep, ResNetDeep
from models.base import rtbBasicBlock, rtbBottleNeck, Bottleneck
from utils.dist_utils import dist_print


def load_pretrain_weights(source_model, target_model):
    if hasattr(source_model, 'state_dict'):
        source_model = {'state_dict': source_model.state_dict()}
        source_state = source_model['state_dict']
    else:
        source_state = source_model

    target_state = target_model.state_dict()

    common = set(
        ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'fc.weight', 'fc.bias'])
    for key in source_state.keys():

        if key in common:
            target_state[key] = source_state[key]
            continue

        if 'linear' in key:
            translated = key
        elif 'downsample' in key:
            layer, num, count, item = re.match('layer(\d+)\.(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'layer%s.%s.block.downsample.%s.%s' % (layer, num, count, item)
        else:
            layer, num, item = re.match('layer(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'layer%s.%s.block.%s' % (layer, num, item)

        if translated in target_state.keys():
            target_state[translated] = source_state[key]
        else:
            print(translated, 'block missing')

    target_model.load_state_dict(target_state)
    return target_model

def load_uniform_weights(source_model, target_model):
    if hasattr(source_model, 'state_dict'):
        source_model = {'state_dict': source_model.state_dict()}
        source_state = source_model['state_dict']
    else:
        source_state = source_model

    target_state = target_model.state_dict()

    for key in source_state.keys():
        if 'router' not in key:
            target_state[key[7:]] = source_state[key]
    target_model.load_state_dict(target_state)

    return target_model

def get_model(model='R110_C10', freeze_gate=False, uniform_sample=False, freeze_net=False, resume_path=None):
    if model == 'R20_C10':
        rnet_checkpoint = './pretrain/R20C10-12fca82f.th'
        layer_config = [3, 3, 3]
        rnet = RouteNet(rtbBasicBlock, layer_config, num_classes=10, freeze_gate=freeze_gate,
                        uniform_sample=uniform_sample, freeze_net=freeze_net)

    elif model == 'R110_C10':
        rnet_checkpoint = './pretrain/R110C10-1d1ed7c2.th'
        layer_config = [18, 18, 18]
        rnet = RouteNet(rtbBasicBlock, layer_config, num_classes=10, freeze_gate=freeze_gate,
                        uniform_sample=uniform_sample, freeze_net=freeze_net)

    elif model == 'R20_C100':
        rnet_checkpoint = './pretrain/R32_C100/pk_E_164_A_0.693.t7'
        layer_config = [3, 3, 3]
        rnet = RouteNet(rtbBasicBlock, layer_config, num_classes=100, freeze_gate=freeze_gate,
                        uniform_sample=uniform_sample, freeze_net=freeze_net)
    elif model == 'R110_C100':
        rnet_checkpoint = './pretrain/R110_C100/pk_E_160_A_0.723.t7'
        layer_config = [18, 18, 18]
        rnet = RouteNet(rtbBasicBlock, layer_config, num_classes=100, freeze_gate=freeze_gate,
                        uniform_sample=uniform_sample, freeze_net=freeze_net)
    elif model == 'R50_ImgNet':
        rnet_checkpoint = './pretrain/resnet50-19c8e357.pth'
        layer_config = [3, 4, 6, 3]
        rnet = RouteNetDeep(rtbBottleNeck, layer_config, num_classes=1000, freeze_gate=freeze_gate,
                            uniform_sample=uniform_sample, freeze_net=freeze_net)
    elif model == 'R101_ImgNet':
        rnet_checkpoint = './pretrain/resnet101-5d3b4d8f.pth'
        layer_config = [3, 4, 23, 3]
        rnet = RouteNetDeep(rtbBottleNeck, layer_config, num_classes=1000, freeze_gate=freeze_gate,
                            uniform_sample=uniform_sample, freeze_net=freeze_net)
    else:
        raise ValueError("model error")

    # if uniform_sample:
    #     dist_print('==> Load pretained model...')
    #     rnet_checkpoint = torch.load(rnet_checkpoint)
    #     load_pretrain_weights(rnet_checkpoint, rnet)
    # elif resume_path is not None:
    #     dist_print('==> Load model from...', resume_path)
    #     rnet.load_state_dict(load_model(resume_path, rnet))
    return rnet, sum(layer_config)

if __name__ == "__main__":
    get_model('R20_C10', uniform_sample=True)
