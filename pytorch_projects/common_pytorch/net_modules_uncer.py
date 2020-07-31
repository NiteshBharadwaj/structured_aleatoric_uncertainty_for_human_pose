import os
import torch
import logging
import numpy as np
import pickle

from common.speedometer import BatchEndParam
from common.utility.image_processing_cv import trans_coords_from_patch_to_org_3d
from common.utility.image_processing_cv import rescale_pose_from_patch_to_camera
from common_pytorch.common_loss.integral import softmax_integral_tensor

from torch.nn.parallel.scatter_gather import gather

from common_pytorch.common_loss.loss_recorder import LossRecorder
from common.utility.image_processing_cv import flip
from common.utility.image_processing_cv import debug_vis_patch

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv
from common_pytorch.common_loss.weighted_mse import make_upper_triangular

def validNet(valid_data_loader, network, loss_config, result_func, loss_func, merge_flip_func,
             patch_width, patch_height, devices, flip_pair, flip_test=True, flip_fea_merge=False):
    """
    :param nth_epoch:
    :param valid_data_loader:
    :param network:
    :param loss_config:
    :param result_func:
    :param loss_func:
    :param patch_size:
    :param devices:
    :param tensor_board:
    :return:
    """
    print('in valid')
    network.eval()

    loss_recorder = LossRecorder()

    preds_in_patch_with_score = []
    sigmas = np.zeros((int(len(valid_data_loader)), 2))
    with torch.no_grad():
        db = valid_data_loader.dataset.db
        for idx, _data in enumerate(valid_data_loader):
            if idx%500==0:
                print(idx)
            batch_data = _data[0]
            #img = (batch_data[idx].data).cpu().numpy().reshape(3,256,256).transpose(1,2,0)*255
            if batch_data.shape[1] != 3:
                flip_test = False

            batch_label = _data[1]
            batch_label_weight = _data[2]
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

            outp = network(batch_data)

            loss = loss_func(outp, batch_label, batch_label_weight)
            if len(devices) > 1:
                outp = gather(outp, 0)
            preds = outp[0]

            uncer = outp[1]
            uncer2 = outp[2]
            if flip_test:
                batch_data_flip = flip(batch_data, dims=3)
                preds_flip = network(batch_data_flip)[0]
            del  batch_data

            # loss

            del batch_label

            loss_recorder.update(loss.detach(), valid_data_loader.batch_size)
            del loss

            # get joint result in patch image
            if len(devices) > 1:
                preds = gather(preds, 0)
                uncer = gather(uncer, 0).detach().cpu()
                uncer2 = gather(uncer2, 0).detach().cpu()

            if flip_test:
                if len(devices) > 1:
                    preds_flip = gather(preds_flip, 0)
                if flip_fea_merge:
                    preds = merge_flip_func(preds, preds_flip, flip_pair)
                    res = result_func(loss_config, patch_width, patch_height, preds)
                else:
                    pipws = result_func(loss_config, patch_width, patch_height, preds)
                    pipws_flip = result_func(loss_config, patch_width, patch_height, preds_flip)
                    pipws_flip[:, :, 0] = patch_width - pipws_flip[:, :, 0] - 1
                    for pair in flip_pair:
                        tmp = pipws_flip[:, pair[0], :].copy()
                        pipws_flip[:, pair[0], :] = pipws_flip[:, pair[1], :].copy()
                        pipws_flip[:, pair[1], :] = tmp.copy()
                    res = (pipws + pipws_flip) * 0.5

            else:
                res = result_func(loss_config, patch_width, patch_height, preds)

            uncer = uncer.detach().cpu()
            uncer2 = uncer2.detach().cpu()
            #res = result_func(loss_config, patch_width, patch_height, preds)
            uncer_concat = torch.cat((uncer, uncer2), 1)
            # q_list = [make_upper_triangular(m, False)
            #           for m in torch.unbind(uncer_concat, dim=0)]
            # batch_q = torch.stack(q_list, dim=0)
            # sigma_inv = torch.bmm(batch_q, torch.transpose(batch_q, 1, 2))
            # sigma = b_inv(sigma_inv)
            # tr_sigma = torch.trace(sigma.view(48,48))
            log_det_sigma = torch.sum(uncer, 1)
            preds_in_patch_with_score.append(res)
            sigmas[idx][0] = log_det_sigma
            sigmas[idx][1] = 0
            del preds, batch_label_weight

    _p = None
    for p in preds_in_patch_with_score:
        if _p is not  None:
            _p = np.vstack((_p,p))
        else:
            _p=p
    preds_in_patch_with_score = _p[0: valid_data_loader.dataset.num_samples]
    return preds_in_patch_with_score, loss_recorder.get_avg(), sigmas


def evalNet(nth_epoch, preds_in_patch_with_score, valid_data_loader, imdb, patch_width, patch_height
            , rect_3d_width, rect_3d_height, final_output_path):
    """
    :param nth_epoch:
    :param preds_in_patch_with_score:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """

    print("in eval")
    # From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_img_with_score = []
    for i in range(valid_data_loader.dataset.num_samples):
        n_sample = valid_data_loader.dataset.to_train_indices[i]
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[i], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], patch_width, patch_height,
                                              rect_3d_width, rect_3d_height))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    name_value = evaluate(preds_in_img_with_score.copy(), final_output_path, imdb_list, valid_data_loader.dataset.joint_num,valid_data_loader.dataset)
    for name, value in name_value:
        logging.info('Epoch[%d] Validation-%s=%f', nth_epoch, name, value)
    return name_value[0][1]

def evalSigma(joints_to_focus,preds_in_patch_with_score, valid_data_loader, imdb, patch_width, patch_height
            , rect_3d_width, rect_3d_height, final_output_path, sigmas,viz):
    """
    :param nth_epoch:
    :param preds_in_patch_with_score:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """

    print("in eval")
    # From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_img_with_score = []
    for i in range(valid_data_loader.dataset.num_samples):
        n_sample = valid_data_loader.dataset.to_train_indices[i]
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[i], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], patch_width, patch_height,
                                              rect_3d_width, rect_3d_height))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    if viz==True:
        name_value = evaluate_sigma_h36m(preds_in_img_with_score.copy(), final_output_path, imdb_list, valid_data_loader.dataset.joint_num, valid_data_loader.dataset,
                                         sigmas,joints_to_focus,imdb_list,preds_in_patch_with_score,False,False)
    name_value = evaluate_sigma_h36m(preds_in_img_with_score.copy(), final_output_path, imdb_list, valid_data_loader.dataset.joint_num, valid_data_loader.dataset,
                                     sigmas,joints_to_focus,imdb_list,preds_in_patch_with_score,viz)
    for name, value in name_value:
        logging.info('Validation-%s=%f', name, value)

def cal_avg_l2_jnt_dist(pose_1, pose_2):

    nJoints = pose_1.shape[1] // 3
    batch_size = pose_1.shape[0]

    pose_1 = np.copy(pose_1).reshape(batch_size, nJoints, 3)
    pose_2 = np.copy(pose_2).reshape(batch_size, nJoints, 3)

    diff = pose_1-pose_2

    temp = 0
    for b in range(batch_size):
        temp = temp + np.sum(np.linalg.norm(diff[b], axis=1, keepdims=True)) / nJoints

    return temp / batch_size


def evaluate(preds, save_path,gts,joint_num, dataset):
    preds = preds[:, :, 0:3]
    sample_num = preds.shape[0]
    metrics_num = 0
    pred_to_save = []
    samples = 0
    sum_e = 0
    for i in range(0, sample_num):
        n_sample = dataset.to_train_indices[i]
        gt = gts[n_sample]
        p = preds[i].copy()
        gt_tmp = gt['joints_3d'].copy()
        p_tmp = p.copy()

        gt_2d_kpt = gt['joints_3d'].copy()
        # get camera depth from root joint
        pre_2d_kpt = preds[i].copy()
        # back project
        pre_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
        gt_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
        if not gt_2d_kpt[:,2].sum()==0:

            gt_3d_root = np.reshape(gt['pelvis'], (1, 3))
            gt_vis = gt['joints_3d_vis'].copy()

            pre_2d_kpt[:, 2] = pre_2d_kpt[:, 2] + gt_3d_root[0, 2]
            gt_2d_kpt[:, 2] = gt_2d_kpt[:, 2] + gt_3d_root[0, 2]

            fl = gt['fl'][0:2]
            c_p = gt['c_p'][0:2]
            for n_jt in range(0, joint_num):
                pre_3d_kpt[n_jt, 0], pre_3d_kpt[n_jt, 1], pre_3d_kpt[n_jt, 2] = \
                    CamBackProj(pre_2d_kpt[n_jt, 0], pre_2d_kpt[n_jt, 1], pre_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
                gt_3d_kpt[n_jt, 0], gt_3d_kpt[n_jt, 1], gt_3d_kpt[n_jt, 2] = \
                    CamBackProj(gt_2d_kpt[n_jt, 0], gt_2d_kpt[n_jt, 1], gt_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
        else:
            pre_3d_kpt[:,:2] = pre_2d_kpt[:,:2]
            gt_3d_kpt[:,:2] = gt_2d_kpt[:,:2]

        diffs = [] # [metrics_num, joint_num * 3]
        pre_3d_kpt_no_root = pre_3d_kpt.copy()
        # should align root, required by protocol #1
        pre_3d_kpt = pre_3d_kpt - pre_3d_kpt [6]
        gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt [6]

        diffs.append((pre_3d_kpt - gt_3d_kpt))

        pred_to_save.append({'pred': pre_3d_kpt,
                             'gt': gt_3d_kpt})

        e = cal_avg_l2_jnt_dist(pre_3d_kpt,gt_3d_kpt)
        samples+=1
        sum_e+=e


    avg_e = sum_e/samples
    print('Avg. MPJPE: {}'.format(avg_e))
    name_value = [
        ('MPJPE      :', avg_e)
    ]

    return name_value

from common_pytorch.dataset.hm36 import CamBackProj
#from common.utility.visualization import debug_img_3d_pose
import cv2
def evaluate_sigma_h36m(preds, save_path, gts, joint_num, dataset, sigmas,joints_to_focus,imdb,preds_in_patch_with_score,viz=False, to_log=True):
    preds = preds[:, :, 0:3]
    sample_num = preds.shape[0]
    metrics_num = 0
    pred_to_save = []
    samples = 0
    sum_e = 0
    sum_a = 0

    # sigmas_dets = np.zeros(sigmas.shape[0])
    # for i in range(sigmas.shape[0]):
    #     sigmas_dets[i] = np.linalg.det(1000*sigmas[i])
    # sigmas = sigmas.reshape(sigmas.shape[0],-1)
    # if joints_to_focus is not None:
    #     sigmas = sigmas[:,joints_to_focus]
    # sigmas = np.abs(sigmas).sum(axis=1)
    sigmas = sigmas[:,0]
    sort_order = np.argsort(sigmas)
    sort_order = sort_order[::-1]
    acs = []
    es = []
    for index in range(0, sample_num):
        i = int(sort_order[index])
        n_sample = dataset.to_train_indices[i]
        gt = gts[n_sample]
        p = preds[i].copy()
        gt_tmp = gt['joints_3d'].copy()
        p_tmp = p.copy()

        gt_2d_kpt = gt['joints_3d'].copy()
        # get camera depth from root joint
        pre_2d_kpt = preds[i].copy()
        # back project
        pre_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
        gt_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
        if not gt_2d_kpt[:,2].sum()==0:

            gt_3d_root = np.reshape(gt['pelvis'], (1, 3))
            gt_vis = gt['joints_3d_vis'].copy()

            pre_2d_kpt[:, 2] = pre_2d_kpt[:, 2] + gt_3d_root[0, 2]
            gt_2d_kpt[:, 2] = gt_2d_kpt[:, 2] + gt_3d_root[0, 2]

            fl = gt['fl'][0:2]
            c_p = gt['c_p'][0:2]
            for n_jt in range(0, joint_num):
                pre_3d_kpt[n_jt, 0], pre_3d_kpt[n_jt, 1], pre_3d_kpt[n_jt, 2] = \
                    CamBackProj(pre_2d_kpt[n_jt, 0], pre_2d_kpt[n_jt, 1], pre_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
                gt_3d_kpt[n_jt, 0], gt_3d_kpt[n_jt, 1], gt_3d_kpt[n_jt, 2] = \
                    CamBackProj(gt_2d_kpt[n_jt, 0], gt_2d_kpt[n_jt, 1], gt_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])

            img_path = ref.h36mImageDirCacheVal + '/' + str(n_sample) + '.png'


        else:
            pre_3d_kpt[:,:2] = pre_2d_kpt[:,:2]
            gt_3d_kpt[:,:2] = gt_2d_kpt[:,:2]
            old_path = imdb[i]['image'].split('images')
            img_path = ref.mpiiImgDir + '/' + old_path[1]
            gt_2d = imdb[i]['joints_3d'][:, 0:2]
            gt_2d = gt_2d.reshape(1,gt_2d.shape[0],gt_2d.shape[1])
            width = imdb[i]['width']
            height = imdb[i]['height']
            widths = np.array([width]).reshape(1,-1)
            heights = np.array([height]).reshape(1,-1)
            preds_2d = pre_2d_kpt[:, 0:2].copy()
            preds_2d = preds_2d.reshape(1,preds_2d.shape[0],preds_2d.shape[1])
            a = Accuracy_Direct(preds_2d, gt_2d, widths, heights)
            acs.append(a)
            sum_a+=a
        diffs = []  # [metrics_num, joint_num * 3]
        pre_3d_kpt_no_root = pre_3d_kpt.copy()
        # should align root, required by protocol #1
        pre_3d_kpt = pre_3d_kpt - pre_3d_kpt[6]
        gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[6]

        diffs.append((pre_3d_kpt - gt_3d_kpt))

        pred_to_save.append({'pred': pre_3d_kpt,
                             'gt': gt_3d_kpt})

        e = cal_avg_l2_jnt_dist(pre_3d_kpt, gt_3d_kpt)
        es.append(e)

        samples+=1
        sum_e+=e

        sigma = sigmas[i]
        if to_log:
            print('Id: {} Sigma: {} MPJPE {} Acc {}'.format(n_sample,sigma,e,a))
        if viz:
            pose = [ pre_3d_kpt, gt_3d_kpt]
            res = preds_in_patch_with_score[i]
            res = np.round(res).astype('int32')
            debug_img_3d_pose(img_path, pose, res[:, 0:3], gt['joints_3d_vis'], gt['flip_pairs'], )


    avg_e = sum_e/samples
    avg_a = sum_a/samples
    print('Avg. MPJPE: {}'.format(avg_e))
    print('Avg. Acc: {}'.format(avg_a))
    print('Top 100 MPJPE: {}'.format(np.sum(es[:100])))
    print('Top 100 Acc: {}'.format(np.sum(acs[:100])))
    name_value = [
        ('hm36_16j      :', avg_e)
    ]

    return name_value

def evalNet_2D(nth_epoch, preds_in_patch_with_score, valid_data_loader, imdb, patch_width, patch_height
            , rect_3d_width, rect_3d_height, final_output_path):
    """
    :param nth_epoch:
    :param preds_in_patch_with_score:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """

    print("in eval")
    # From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_img_with_score = []
    for i in range(valid_data_loader.dataset.num_samples):
        n_sample = valid_data_loader.dataset.to_train_indices[i]
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[i], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], patch_width, patch_height,
                                              rect_3d_width, rect_3d_height))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)
    acc_all = valid_data_loader.dataset.dbo.evaluate(preds_in_img_with_score, None)
    print(acc_all)
    acc = acc_all[7][1]
    print('2D Accuracy: '+ str(acc))
    return acc


def evalNetChallenge(nth_epoch, preds_in_patch, valid_data_loader, imdb, final_output_path):
    """
    :param nth_epoch:
    :param preds_in_patch_with:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """
    target_bone_length = 4502.881  # train+val
    # target_bone_length = 4522.828     # train
    # target_bone_length = 4465.869     # val
    print("in eval")

    # 4. From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_camera_space = []
    for n_sample in range(valid_data_loader.dataset.num_samples):
        temp = preds_in_patch[n_sample].copy()
        # db = imdb_list[n_sample]
        # w = db['width']
        # temp[:,2] = temp[:,2]*2000/w

        # preds_in_camera_space.append(
        #     rescale_pose_from_patch_to_camera(temp,
        #                                       target_bone_length,
        #                                       imdb_list[n_sample]['parent_ids']))

        preds_in_camera_space.append(temp)
    preds_in_camera_space = np.asarray(preds_in_camera_space)[:, :, 0:3]

    # 5. Convert joint type
    preds_in_camera_space_cvt = preds_in_camera_space.copy()

    # 6. Evaluate
    name_value = evaluate2(preds_in_camera_space_cvt, final_output_path, imdb_list, valid_data_loader.dataset.joint_num)
    for name, value in name_value:
        logging.info('Epoch[%d] Validation-%s=%f', nth_epoch, name, value)

    return preds_in_patch
