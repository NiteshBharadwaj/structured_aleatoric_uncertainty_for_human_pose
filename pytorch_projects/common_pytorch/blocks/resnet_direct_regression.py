import os
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torchvision.models.resnet import model_zoo, model_urls

from common_pytorch.base_modules.resnet import resnet_spec, ResNetBackbone
from common_pytorch.base_modules.avg_pool_head import AvgPoolHead, AvgPoolHead2


def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 50
    config.fea_map_size = 0
    return config


class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class ResPoseNet_U(nn.Module):
    def __init__(self, backbone, head, head2, head3):
        super(ResPoseNet_U, self).__init__()
        self.backbone = backbone
        self.head = head
        self.head2 = head2
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.head3 = head3

    def forward(self, x):
        enc = self.backbone(x)
        enc_lat = self.avgpool(enc)
        x = self.head(enc)
        y = self.head2(enc)
        z = self.head3(enc)
        return x, y, z

def get_pose_net(cfg, num_joints, is_cov = False):
    block_type, layers, channels, name = resnet_spec[cfg.num_layers]
    backbone_net = ResNetBackbone(block_type, layers)
    head_net = AvgPoolHead(channels[-1], num_joints * 3, cfg.fea_map_size)
    n_outs_cov = 1128 if is_cov else 174
    head_net_2 = AvgPoolHead(
        channels[-1], num_joints * 3, cfg.fea_map_size)
    head_net_3 = AvgPoolHead2(
        channels[-1], n_outs_cov, cfg.fea_map_size)
    pose_net = ResPoseNet_U(backbone_net, head_net, head_net_2, head_net_3)
    return pose_net


def init_pose_net(pose_net, cfg):
    if cfg.from_model_zoo  and (cfg.pretrained is None or cfg.pretrained=='None'):
        print('Initializing network with resnet weights')
        _, _, _, name = resnet_spec[cfg.num_layers]
        org_resnet = model_zoo.load_url(model_urls[name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        pose_net.backbone.load_state_dict(org_resnet)
        # print('Loading pretrained model from {}'.format(os.path.join(cfg.pretrained, model_file)))
    else:
        if os.path.exists(cfg.pretrained):
            model = torch.load(cfg.pretrained)
            transfer_partial_weights(model['network'], pose_net, prefix='module')
            print("Init Network from pretrained", cfg.pretrained)

def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None, add_prefix=''):
    own_state = obj.state_dict()
    copyCount = 0
    skipCount = 0
    paramCount = len(own_state)
    copied_param_names = []
    skipped_param_names = []
    for name_raw, param in state_dict_other.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            # print('.data conversion for ',name)
        if prefix is not None and not name_raw.startswith(prefix):
            print("skipping {} because of prefix {}".format(name_raw, prefix))
            continue

        # remove the path of the submodule from which we load
        name =  ''.join(name_raw.split(prefix+'.')[1:])

        if name in own_state:
            if hasattr(own_state[name], 'copy_'):  # isinstance(own_state[name], torch.Tensor):
                # print('copy_ ',name)
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                    copyCount += 1
                    copied_param_names.append(name)
                else:
                    print('Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(),
                                                                                         param.size(), name))
                    skipCount += 1
                    skipped_param_names.append(name)

            elif hasattr(own_state[name], 'copy'):
                own_state[name] = param.copy()
                copyCount += 1
                copied_param_names.append(name)
            else:
                print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name, name_raw))
                print(type(own_state[name]))
                skipCount += 1
                skipped_param_names.append(name)
                IPython.embed()
        else:
            skipCount += 1
            print('Warning, no match for {}, ignoring'.format(name))
            skipped_param_names.append(name)
            # print(' since own_state.keys() = ',own_state.keys())

    print('Copied {} elements, {} skipped, and {} target params without source'.format(copyCount, skipCount,
                                                                                       paramCount - copyCount))
    return copied_param_names, skipped_param_names
