# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# compute on the original embeddings not interpolate to image size
import os
import numpy as np
import torch
import time
import SimpleITK as sitk
import statsmodels.api as sm
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from interfaces import init, get_embedding, get_sim_embed_loc_multi_embedding_space
from utils import read_image, visualize, make_local_grid_patch
from interfaces import get_sim_semantic_embed_loc_multi_embedding_space
from demo import get_random_query_point

os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
config_file = 'configs/samv2/samv2_NIHLN.py'
checkpoint_file = 'checkpoints/SAMv2_iter_20000.pth'

# assume all input image are in torchio "LPS+" direction which equal to "RAI" orientation in ITK-Snap.
im1_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz'
im2_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz'


def main():
    time1 = time.time()
    model = init(config_file, checkpoint_file)
    time2 = time.time()
    print('model loading time:', time2 - time1)
    im1, normed_im1, norm_info_1 = read_image(im1_file)
    im2, normed_im2, norm_info_2 = read_image(im2_file)
    time3 = time.time()
    print('image loading time:', time3 - time2)

    pt1 = get_random_query_point(im1)

    emb1 = get_embedding(normed_im1, model)
    emb2 = get_embedding(normed_im2, model)
    time4 = time.time()
    print('embeddings computing time:', time4 - time3)

    # fixed_point_iterations is slower but more accurate than simple nearest neighbor matching
    score, pt2 = fixed_point_iterations(emb1, norm_info_1, emb2, norm_info_2, im2['shape'], pt1)
    print(pt2, score)
    visualize(im1['img'], im2['img'], norm_info_1, norm_info_2, pt1, pt2, score)


def fixed_point_iterations(emb1, norm_info_1, emb2, norm_info_2, im2_shape, pt1, x_margin=2, y_margin=2, z_margin=2,
                           iterations=4):
    print('performing fixed point iteration')
    pt_query = np.round(pt1).astype('int')
    pt_query_center = np.floor(pt_query * norm_info_1 * 0.5).astype(int)
    # crop a cube region centered in the template point
    pt_query_normed = make_local_grid_patch(pt_query_center, x_margin, y_margin, z_margin,
                                            emb1[0].shape[4], emb1[0].shape[3], emb1[0].shape[2])
    cuda_device = torch.device('cuda:' + str(0))

    for i in range(iterations):
        if i % 2 == 0:
            if i > 0:
                pt_query_normed = np.floor(pt_query * norm_info_1 * 0.5).astype(int)
            pt_key_normed, score = get_sim_semantic_embed_loc_multi_embedding_space(emb1, emb2, pt_query_normed,
                                                                                    cuda_device)
            pt_key = np.round((pt_key_normed * 2 + 0.5) / norm_info_2).astype(int)
            pt_k_final = pt_key
        else:
            pt_query_normed = np.floor(pt_query * norm_info_2 * 0.5).astype(int)
            pt_key_normed, score = get_sim_semantic_embed_loc_multi_embedding_space(emb2, emb1, pt_query_normed,
                                                                                    cuda_device)
            pt_key = np.round((pt_key_normed * 2 + 0.5) / norm_info_1).astype(int)
            pt_q_final = pt_key
            pt_final_score = score
        print(f'score:{score.mean(), score.max()}')
        pt_query = pt_key
    dis = np.linalg.norm((pt_q_final - np.array(pt1)) * norm_info_1, axis=1)
    final_q_all = []
    final_k_all = []
    final_score_all = []
    for i in range(dis.shape[0]):
        if dis[i] < 100 and pt_final_score[i] > .8:
            final_q_all.append(pt_q_final[i, :])
            final_k_all.append(pt_k_final[i, :])
            final_score_all.append(pt_final_score[i])

    final_q_all = np.array(final_q_all)
    final_k_all = np.array(final_k_all)
    final_score_all = np.array(final_score_all)
    print(final_q_all.shape[0])

    final_k_list = list([tuple(t) for t in final_k_all])
    final_k_count = dict((k_ele, final_k_list.count(k_ele)) for k_ele in final_k_list)

    final_q = []
    final_k = []

    for keys in final_k_count:
        ind_k = np.where(np.logical_and(np.logical_and(final_k_all[:, 0] == keys[0], final_k_all[:, 1] == keys[1]),
                                        final_k_all[:, 2] == keys[2]))[0]
        s = final_score_all[ind_k]
        q = final_q_all[ind_k, :]
        final_q.append(q[0, :])
        final_k.append(keys)
    final_q = np.array(final_q)
    final_k = np.array(final_k)

    k_xs = np.unique(final_k[:, 0])
    k_ys = np.unique(final_k[:, 1])
    k_zs = np.unique(final_k[:, 2])
    q_zs = np.unique(final_q[:, 2])

    shift = final_q - pt1
    final_q_mm = final_q * norm_info_1
    final_k_mm = final_k * norm_info_2
    shift_mm = shift * norm_info_1
    # pt_q_shift = final_q - shift

    ones = np.ones((final_q.shape[0], 1))
    final_q_ext = np.concatenate((final_q, ones), axis=1)
    m = []

    if k_zs.shape[0] == 1:
        z_ratro = norm_info_1[2] / norm_info_2[2]
        z_bias = k_zs - q_zs * z_ratro
        k = [0, 1]
        for i in k:
            mod = sm.RLM(final_k[:, i],
                         np.concatenate([final_q_ext[:, 0:1], final_q_ext[:, 1:2], final_q_ext[:, 3:]], axis=1))
            res = mod.fit()
            m.append(res.params)
        m.append([0, 0, z_bias[0]])
        m.append(np.array([0, 0, 1]))
        m = np.array(m)
        m = np.concatenate([m[:, :2], np.array([0, 0, z_ratro, 0]).reshape(-1, 1), m[:, 2:]], axis=1)

    else:
        k = [0, 1, 2]
        for i in k:
            mod = sm.RLM(final_k[:, i], final_q_ext[:, ])
            res = mod.fit()
            m.append(res.params)
        m.append(np.array([0, 0, 0, 1]))
        m = np.array(m)

    shift_affine = np.matmul(m[:3, :3], shift.T).transpose(1, 0)
    pt_k_shift = final_k - shift_affine
    pt_k_shift = pt_k_shift.astype(int)
    pt_k_shift = np.where(pt_k_shift < 0, 0, pt_k_shift)
    max_shape = np.array(im2_shape[1:]) - 1
    max_shape = np.array([max_shape[1], max_shape[0], max_shape[2]])
    pt_k_shift = np.where(pt_k_shift > max_shape, max_shape, pt_k_shift)
    pt_k_shift_mean = np.mean(pt_k_shift, axis=0)
    pt_k_shift_mean = pt_k_shift_mean.astype('int')

    return final_score_all.mean(), pt_k_shift_mean


if __name__ == '__main__':
    main()
