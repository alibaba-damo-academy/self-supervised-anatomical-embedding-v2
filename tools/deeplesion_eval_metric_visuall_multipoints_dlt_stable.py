# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# single process tumor tracking evaluation and visualization

import torch
import torchio as tio
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import json
from utils import make_local_grid_patch
from interfaces import get_sim_embed_loc_multi_embedding_space, get_sim_semantic_embed_loc_multi_embedding_space
import statsmodels.api as sm

data_path = '/data/sdd/user/rawdata/Deeplesion/Images_nifti/'
prediction_path = '/data/sdd/user/results/result-dlt/'
save_path = '/data/sdd/user/visual/visual-dlt_jimmy/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
all_dis = []
all_dis_x = []
all_dis_y = []
all_dis_z = []
draw_results = False

anno_path = '/data/sdd/user/rawdata/DLT-main/data/'
gths = json.load(open(os.path.join(anno_path, 'test.json'), 'r'))
all_keys = list(gths.keys())
all_img = []
for i in range(all_keys.__len__()):
    element = gths[all_keys[i]]
    query_img_name = element['source']
    key_img_name = element['target']
    if query_img_name not in all_img:
        all_img.append(query_img_name)
    if key_img_name not in all_img:
        all_img.append(key_img_name)
import csv

with open("/data/sdd/user/processed_data/deeplesion_dlt_jimmy.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name"])
    for nii_name in all_img:
        writer.writerow([nii_name])

normed_spacing = (2., 2., 2.)
gpu_id = 0
cuda_device = torch.device('cuda:' + str(gpu_id))
count = 0
hit = 0
miss = 0
hit_10 = 0
miss_10 = 0
for ii in range(all_keys.__len__()):
    print(ii)
    element = gths[all_keys[ii]]
    count = count + 1
    query_img_name = element['source'].split('.', 1)[0]
    query_bbx = element[
        'source box']  # [x1,y1,z1,x2-x1,y2-y1,z2-z1] z index is the inverse of the sever deeplesion dataset
    query_center = element['source center']
    key_img_name = element['target'].split('.', 1)[0]
    key_bbx = element['target box']
    key_center = element['target center']
    query_img_info = pickle.load(open(prediction_path + query_img_name + '.pkl', 'rb'))
    key_img_info = pickle.load(open(prediction_path + key_img_name + '.pkl', 'rb'))

    query_img = tio.ScalarImage(os.path.join(data_path, query_img_name + '.nii.gz'))
    query_img_origin_spacing = query_img.spacing
    query_img_origin_shape = query_img.shape
    query_spacing_norm_ratio = np.array(query_img_origin_spacing) / np.array(normed_spacing)
    query_bbx_reshaped = np.array(query_bbx).reshape(2, 3)
    query_bbx_reshaped[1, :] = query_bbx_reshaped[1, :] + query_bbx_reshaped[0, :]
    query_bbx_reshaped[:, 2] = query_img.shape[3] - query_bbx_reshaped[:, 2]
    query_bbx_reshaped[:, 2] = query_bbx_reshaped[::-1, 2]
    query_bbx_normed = query_bbx_reshaped * query_spacing_norm_ratio
    query_center[2] = query_img_origin_shape[3] - query_center[2] - 1

    query_point_fine = (query_bbx_reshaped[1, :] + query_bbx_reshaped[0, :]) / 2 * query_spacing_norm_ratio
    query_center_fine = query_center * query_spacing_norm_ratio
    query_point_fine = query_center_fine

    key_img = tio.ScalarImage(os.path.join(data_path, key_img_name + '.nii.gz'))
    key_img_origin_spacing = key_img.spacing
    key_img_origin_shape = key_img.shape
    key_spacing_norm_ratio = np.array(key_img_origin_spacing) / np.array(normed_spacing)
    key_bbx_reshaped = np.array(key_bbx).reshape(2, 3)
    key_bbx_reshaped[1, :] = key_bbx_reshaped[1, :] + key_bbx_reshaped[0, :]

    key_bbx_reshaped[:, 2] = key_img.shape[3] - key_bbx_reshaped[:, 2]
    key_bbx_reshaped[:, 2] = key_bbx_reshaped[::-1, 2]
    key_bbx_normed = key_bbx_reshaped * key_spacing_norm_ratio
    # key_bbx_normed = np.floor(key_bbx_normed).astype(int)
    key_center[2] = key_img_origin_shape[3] - key_center[2] - 1

    key_point_fine_gt = (key_bbx_reshaped[1, :] + key_bbx_reshaped[0, :]) / 2 * key_spacing_norm_ratio
    # key_point_fine_gt = key_point_fine_gt.astype(int)
    key_center_fine = key_center * key_spacing_norm_ratio
    key_point_fine_gt = key_center_fine

    query_point_fine_re = np.floor(query_point_fine / 2.).astype(int)
    query_point_fine_re_matrix = make_local_grid_patch(query_point_fine_re, 1, 1, 1, query_img_info[0].shape[4],
                                                       query_img_info[0].shape[3], query_img_info[0].shape[2])

    iters = 4
    for j in range(iters):
        if j == 0:
            pts_query = query_point_fine_re_matrix
            pts_query_final = pts_query
        if j % 2 == 0:
            pts_key_inter, s_inter, pts_key, s = get_sim_semantic_embed_loc_multi_embedding_space(query_img_info,
                                                                                                  key_img_info,
                                                                                                  pts_query,
                                                                                                  cuda_device,
                                                                                                  return_interploted=True)
            pts_key_final = pts_key
            pts_key_final_inter = pts_key_inter
        else:
            pts_key_inter, s_inter, pts_key, s = get_sim_semantic_embed_loc_multi_embedding_space(key_img_info,
                                                                                                  query_img_info,
                                                                                                  pts_query,
                                                                                                  cuda_device,
                                                                                                  return_interploted=True)
            pts_query_final = pts_key
            pts_query_final_inter = pts_key_inter
        pts_query = pts_key
    dis = np.linalg.norm(pts_query_final_inter - query_point_fine,
                         axis=1)
    final_q = []
    final_k = []
    scores = []
    for i in range(dis.shape[0]):
        if dis[i] < 100 and s[i] > 1.3:
            final_q.append(pts_query_final_inter[i, :])
            final_k.append(pts_key_final_inter[i, :])
            scores.append(s[i])
    final_q = np.array(final_q)
    final_k = np.array(final_k)
    shift = final_q - query_point_fine
    ones = np.ones((final_q.shape[0], 1))

    final_q_ext = np.concatenate((final_q, ones), axis=1)
    m = []
    w = [0, 1, 2]
    for i in w:
        mod = sm.RLM(final_k[:, i], final_q_ext)
        res = mod.fit()
        m.append(res.params)
    m.append(np.array([0, 0, 0, 1]))
    m = np.array(m)

    shift_affine = np.matmul(m[:3, :3], shift.T).transpose(1, 0)
    pt_k_shift = final_k - shift_affine
    # pt_k_shift = pt_k_shift.astype(int)
    pt_k_shift = np.where(pt_k_shift < 0, 0, pt_k_shift)
    max_shape = np.array(key_img_info[3].shape[2:]) - 1
    max_shape = np.array([max_shape[1], max_shape[2], max_shape[0]])
    pt_k_shift = np.where(pt_k_shift > max_shape, max_shape, pt_k_shift)
    pt_k_shift_mean = np.mean(pt_k_shift, axis=0)
    pt_k_shift_std = np.std(pt_k_shift, axis=0)
    use_ind = np.abs((pt_k_shift[:, 2] - pt_k_shift_mean[2]) < 1)
    pt_k_shift_mean_r = np.mean(pt_k_shift[use_ind, :], axis=0)

    pre = pt_k_shift_mean
    pre = pre + 0.5
    scores = np.array(scores)
    score = scores.mean()
    pre_remap = pre / key_spacing_norm_ratio
    final = key_bbx_normed - pre

    dis = np.linalg.norm((key_point_fine_gt - pre)) * 2.
    dis_x = np.linalg.norm((key_point_fine_gt[0] - pre[0])) * 2.
    dis_y = np.linalg.norm((key_point_fine_gt[1] - pre[1])) * 2.
    dis_z = np.linalg.norm((key_point_fine_gt[2] - pre[2])) * 2.
    all_dis.append(dis)
    all_dis_x.append(dis_x)
    all_dis_y.append(dis_y)
    all_dis_z.append(dis_z)
    final = key_bbx_reshaped - pre_remap
    if (final[0, :] <= 0).all() and (final[1, :] > -0).all():
        hit = hit + 1
    else:
        miss = miss + 1

    if dis <= 10:
        hit_10 = hit_10 + 1
    else:
        miss_10 = miss_10 + 1

    if draw_results:  # set True to draw matching results
        query_point_fine = np.floor(query_point_fine).astype(int)
        q_img = query_img_info[3][0, 0, :, :, :]

        q_img = (q_img + 50)
        q_img[q_img < 51] = 51
        q_img[q_img > 76.2] = 76.2
        q_img = (q_img - 51) / (76.2 - 51)

        slice = q_img[query_point_fine[2], :, :]
        # q_img[query_point_fine[1] - 1:query_point_fine[1] + 2,
        # query_point_fine[0] - 1:query_point_fine[0] + 2] = 1
        fig, ax = plt.subplots(3, 3, figsize=(33, 18))
        fig.suptitle('distance:%.2f' % dis + 'score:%.2f' % score)
        # plt.subplot(1,3,1)
        ax[0, 0].set_title('query')
        ax[0, 0].imshow(slice, cmap='gray')
        ax[0, 0].plot((query_point_fine[0]), (query_point_fine[1]), 'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)

        rect = patches.Rectangle((query_bbx_normed[0, 0], query_bbx_normed[0, 1]),
                                 query_bbx_normed[1, 0] - query_bbx_normed[0, 0],
                                 query_bbx_normed[1, 1] - query_bbx_normed[0, 1]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[0, 0].add_patch(rect)

        slice = q_img[:, query_point_fine[1], :]
        slice = slice[::-1, :]
        ax[1, 0].set_title('query')
        ax[1, 0].imshow(slice, cmap='gray')
        ax[1, 0].plot((query_point_fine[0]), (q_img.shape[0] - query_point_fine[2] - 1), 'o',
                      markerfacecolor='none', markeredgecolor="red",
                      markersize=12, markeredgewidth=2)

        rect = patches.Rectangle((query_bbx_normed[0, 0], q_img.shape[0] - query_bbx_normed[1, 2] - 1),
                                 query_bbx_normed[1, 0] - query_bbx_normed[0, 0],
                                 query_bbx_normed[1, 2] - query_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[1, 0].add_patch(rect)

        # plt(x,y...)

        slice = q_img[:, :, query_point_fine[0]]
        slice = slice[::-1, :]
        ax[2, 0].set_title('query')
        ax[2, 0].imshow(slice, cmap='gray')
        ax[2, 0].plot((query_point_fine[1]), (q_img.shape[0] - query_point_fine[2] - 1), 'o',
                      markerfacecolor='none', markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        rect = patches.Rectangle((query_bbx_normed[0, 1], q_img.shape[0] - query_bbx_normed[1, 2] - 1),
                                 query_bbx_normed[1, 1] - query_bbx_normed[0, 1],
                                 query_bbx_normed[1, 2] - query_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[2, 0].add_patch(rect)

        key_point_fine_gt = np.floor(key_point_fine_gt).astype(int)

        k_img_gt = key_img_info[3][0, 0, :, :, :]

        k_img_gt = (k_img_gt + 50)
        k_img_gt[k_img_gt < 51] = 51
        k_img_gt[k_img_gt > 76.2] = 76.2
        k_img_gt = (k_img_gt - 51) / (76.2 - 51)
        # k_img_gt[key_point_fine_gt[1] - 1:key_point_fine_gt[1] + 2,
        # key_point_fine_gt[0] - 1:key_point_fine_gt[0] + 2] = 1
        # plt.subplot(1,3,2)
        # plt.title('key_gt')
        # plt.imshow(k_img_gt, cmap='gray')
        # plt.savefig(save_path + pair_id + '_' + 'key_gth' + '.png')

        slice = k_img_gt[key_point_fine_gt[2], :, :]

        ax[0, 1].set_title('key_gt')
        ax[0, 1].imshow(slice, cmap='gray')
        ax[0, 1].plot((key_point_fine_gt[0]), (key_point_fine_gt[1]), 'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        rect = patches.Rectangle((key_bbx_normed[0, 0], key_bbx_normed[0, 1]),
                                 key_bbx_normed[1, 0] - key_bbx_normed[0, 0],
                                 key_bbx_normed[1, 1] - key_bbx_normed[0, 1]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[0, 1].add_patch(rect)

        slice = k_img_gt[:, key_point_fine_gt[1], :]
        slice = slice[::-1, :]

        ax[1, 1].set_title('key_gt')
        ax[1, 1].imshow(slice, cmap='gray')
        ax[1, 1].plot((key_point_fine_gt[0]), (k_img_gt.shape[0] - key_point_fine_gt[2] - 1), 'o',
                      markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        rect = patches.Rectangle((key_bbx_normed[0, 0], k_img_gt.shape[0] - key_bbx_normed[1, 2] - 1),
                                 key_bbx_normed[1, 0] - key_bbx_normed[0, 0],
                                 key_bbx_normed[1, 2] - key_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[1, 1].add_patch(rect)

        slice = k_img_gt[:, :, key_point_fine_gt[0]]
        slice = slice[::-1, :]

        ax[2, 1].set_title('key_gt')
        ax[2, 1].imshow(slice, cmap='gray')
        ax[2, 1].plot((key_point_fine_gt[1]), (k_img_gt.shape[0] - key_point_fine_gt[2] - 1), 'o',
                      markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        rect = patches.Rectangle((key_bbx_normed[0, 1], k_img_gt.shape[0] - key_bbx_normed[1, 2] - 1),
                                 key_bbx_normed[1, 1] - key_bbx_normed[0, 1],
                                 key_bbx_normed[1, 2] - key_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[2, 1].add_patch(rect)

        key_img_pre = key_img_info[3][0, 0, :, :, :]
        key_img_pre = (key_img_pre + 50)
        key_img_pre[key_img_pre < 51] = 51
        key_img_pre[key_img_pre > 76.2] = 76.2
        key_img_pre = (key_img_pre - 51) / (76.2 - 51)
        # key_img_pre[ind[1][0] - 1:ind[1][0] + 2,
        # ind[2][0] - 1:ind[2][0] + 2] = 1
        # plt.subplot(1,3,3)
        # plt.title('key_pre')
        # plt.imshow(key_img_pre, cmap='gray')

        pre = np.floor(pre).astype('int')
        slice = key_img_pre[pre[2], :, :]
        ax[0, 2].set_title('key_pre')
        ax[0, 2].imshow(slice, cmap='gray')
        ax[0, 2].plot(pre[0], pre[1], 'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)

        # also draw gt bbox
        rect = patches.Rectangle((key_bbx_normed[0, 0], key_bbx_normed[0, 1]),
                                 key_bbx_normed[1, 0] - key_bbx_normed[0, 0],
                                 key_bbx_normed[1, 1] - key_bbx_normed[0, 1]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[0, 2].add_patch(rect)

        slice = key_img_pre[:, pre[1], :]
        slice = slice[::-1, :]
        ax[1, 2].set_title('key_pre')
        ax[1, 2].imshow(slice, cmap='gray')
        ax[1, 2].plot(pre[0], (key_img_pre.shape[0] - pre[2] - 1),
                      'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)

        # also draw gt bbox
        rect = patches.Rectangle((key_bbx_normed[0, 0], k_img_gt.shape[0] - key_bbx_normed[1, 2] - 1),
                                 key_bbx_normed[1, 0] - key_bbx_normed[0, 0],
                                 key_bbx_normed[1, 2] - key_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[1, 2].add_patch(rect)

        slice = key_img_pre[:, :, pre[0]]
        slice = slice[::-1, :]
        ax[2, 2].set_title('key_pre')
        ax[2, 2].imshow(slice, cmap='gray')
        ax[2, 2].plot(pre[1], (key_img_pre.shape[0] - pre[2] - 1),
                      'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)

        # also draw gt bbox
        rect = patches.Rectangle((key_bbx_normed[0, 1], k_img_gt.shape[0] - key_bbx_normed[1, 2] - 1),
                                 key_bbx_normed[1, 1] - key_bbx_normed[0, 1],
                                 key_bbx_normed[1, 2] - key_bbx_normed[0, 2]
                                 , linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax[2, 2].add_patch(rect)

        plt.tight_layout()

        # plt.show()
        # plt.close()
        plt.savefig(save_path + str(all_keys[ii]) + '_' + str('%.2f' % dis) + '_stable' + '.png')
        plt.close()
print(count, hit, miss)
print(hit / count)
print(count, hit_10, miss_10)
print(hit_10 / count)
print(np.array(all_dis).mean(), np.array(all_dis).std())
print(np.array(all_dis_x).mean(), np.array(all_dis_x).std())
print(np.array(all_dis_y).mean(), np.array(all_dis_y).std())
print(np.array(all_dis_z).mean(), np.array(all_dis_z).std())
