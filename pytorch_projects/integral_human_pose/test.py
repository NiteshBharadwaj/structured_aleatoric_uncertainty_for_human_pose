import os
import pprint
import copy
import time
import logging

import torch
from torch.utils.data import DataLoader

# define project dependency
import _init_paths

from integral_human_pose.core.loader import hm36_Dataset, mpii_hm36_Dataset, mpii as MPII_Dataset
# project dependence
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, s_args, s_config, \
    s_config_file
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from common_pytorch.net_modules_uncer import validNet, evalNet, evalSigma, evalNet_2D

# import dynamic config
exec('from blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')
exec('from loss.' + s_config.pytorch.loss + \
     ' import get_default_loss_config, get_loss_func, get_label_func, get_result_func, get_merge_func')

def main():

    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)

    # create log and path
    final_log_path = os.path.dirname(s_args.model)
    log_name = os.path.basename(s_args.model)
    logging.basicConfig(filename=os.path.join(final_log_path, '{}_test.log'.format(log_name)),
                        format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # lable, loss, metric, result and flip function
    logger.info("Defining lable, loss, metric, result and flip function")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)
    result_func = get_result_func(config.loss)
    merge_flip_func = get_merge_func(config.loss)
    is_cov = s_args.is_cov == 'True' or s_args.is_cov == True

    # dataset
    logger.info("Creating dataset")
    train_imdbs = []
    valid_imdbs = []
    for n_db in range(0, len(config.dataset.name)):
        valid_imdbs.append(
            eval(config.dataset.name[n_db])(config.dataset.test_image_set[n_db], config.dataset.path[n_db],
                                            config.train.patch_width, config.train.patch_height,
                                            config.train.rect_3d_width, config.train.rect_3d_height))

    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    # basic data_loader unit
    dataset_name = ""
    for n_db in range(0, len(config.dataset.name)):
        dataset_name = dataset_name + config.dataset.name[n_db] + "_"

    dataset_valid_mpii = MPII_Dataset(valid_imdbs[0], False, '', config.train.patch_width, config.train.patch_height,
                                config.train.rect_3d_width,
                                config.train.rect_3d_height, batch_size,
                                config.dataiter.mean, config.dataiter.std,
                                config.aug, label_func, config.loss)

    valid_data_loader_mpii = DataLoader(dataset=dataset_valid_mpii, batch_size=1, shuffle=False,
                                   num_workers=config.dataiter.threads, drop_last=False)

    visualize = False

    focus_joints = [0]
    focus_bones = [0]

    # Pass none to add all uncertainties
    joint_ids_to_focus_on =  None #get_joint_indices_in_corr(focus_joints, focus_bones)
    # prepare network
    assert os.path.exists(s_args.model), 'Cannot find model!'
    logger.info('Load checkpoint from {}'.format(s_args.model))
    joint_num = dataset_valid_mpii.joint_num
    net = get_pose_net(config.network, joint_num, is_cov)
    net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
    ckpt = torch.load(s_args.model)  # or other path/to/model
    net.load_state_dict(ckpt['network'])
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # test
    logger.info("Test DB size: {}.".format(int(len(dataset_valid_mpii))))
    beginT = time.time()
    preds_in_patch_with_score, vloss, sigmas = \
        validNet(valid_data_loader_mpii, net, config.loss, result_func, loss_func, merge_flip_func,
                 config.train.patch_width, config.train.patch_height, devices,
                 valid_imdbs[0].flip_pairs, flip_test=True)
    print('V Loss: ' + str(vloss))
    beginT = time.time()

    evalNet_2D(0, preds_in_patch_with_score, valid_data_loader_mpii, None,
               config.train.patch_width, config.train.patch_height, config.train.rect_3d_width,
               config.train.rect_3d_height, final_log_path)


    endt3 = time.time() - beginT

    print('Testing %.2f seconds.....' % (time.time() - beginT))

import numpy as np
def get_joint_indices_in_corr(joints, bones):
    joint_idxs_map = np.arange(16*3).reshape(16,3).astype('int32')
    corr_map = np.arange(48*48).reshape(48,48).astype('int32')
    idxs = []
    for j in joints:
        joint_idx = joint_idxs_map[j]
        for i1 in range(joint_idxs_map.shape[1]):
            for i2 in range(i1,joint_idxs_map.shape[1]):
                idxs.append(corr_map[joint_idx[i1],joint_idx[i2]])
    for b in bones:
        j1_idx = joint_idxs_map[ref.edges[b][0]]
        j2_idx = joint_idxs_map[ref.edges[b][1]]
        for i1 in range(joint_idxs_map.shape[1]):
            for i2 in range(joint_idxs_map.shape[1]):
                idxs.append(corr_map[j1_idx[i1],j2_idx[i2]])
    return idxs


def user_aug(img_patch,img_patch_cv, joints, joints_vis):
    return img_patch, img_patch_cv, joints

if __name__ == "__main__":
    main()
