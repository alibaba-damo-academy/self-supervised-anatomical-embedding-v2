# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torch
import torchio as tio
import pickle
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import ipdb
from time import time, sleep

from multiprocessing import Pool, current_process, Queue, Event
from multiprocessing import set_start_method, current_process
from interfaces import get_sim_semantic_embed_loc_multi_embedding_space, get_sim_embed_loc_multi_embedding_space
from utils import make_local_grid_patch
import statsmodels.api as sm

gth_path = '/data/sdd/user/processed_data/ind_files/DL_lesion_pairs_1022_correct.pth'
count = 0
hit = 0
miss = 0
NUM_GPUS = 2
PROC_PER_GPU = 4
margin_x, margin_y, margin_z = 1, 1, 1
stable_matching = True
semantic = True


def infer_element(element):
    gpu_id = (int(current_process().name.split('-')[-1]) - 1) % NUM_GPUS
    torch.cuda.set_device(f'cuda:{gpu_id}')
    cuda_device = torch.device('cuda:' + str(gpu_id))
    data_path = '/data/sdd/user/rawdata/Deeplesion/Images_nifti/'
    prediction_path = '/data/sdd/user/results/result-dlt/'
    normed_spacing = (2., 2., 2.)
    # ------------------------------------image information------------------------------------
    query_img_name = element[0]
    query_bbx = element[1]
    key_img_name = element[2]
    key_bbx = element[3]
    pair_id = element[4]
    query_img_info = pickle.load(open(prediction_path + query_img_name + '.pkl', 'rb'))
    key_img_info = pickle.load(open(prediction_path + key_img_name + '.pkl', 'rb'))

    query_img = tio.ScalarImage(os.path.join(data_path, query_img_name + '.nii.gz'))
    query_img_origin_spacing = query_img.spacing
    query_spacing_norm_ratio = np.array(query_img_origin_spacing) / np.array(normed_spacing)
    query_bbx_reshaped = np.array(query_bbx).reshape(2, 3)
    query_bbx_reshaped[:, 2] = query_img.shape[3] - query_bbx_reshaped[:, 2]
    query_bbx_reshaped[:, 2] = query_bbx_reshaped[::-1, 2]
    query_point_fine = (query_bbx_reshaped[1, :] + query_bbx_reshaped[0, :]) / 2 * query_spacing_norm_ratio

    key_img = tio.ScalarImage(os.path.join(data_path, key_img_name + '.nii.gz'))
    key_img_origin_spacing = key_img.spacing
    key_spacing_norm_ratio = np.array(key_img_origin_spacing) / np.array(normed_spacing)
    key_bbx_reshaped = np.array(key_bbx).reshape(2, 3)
    key_bbx_reshaped[:, 2] = key_img.shape[3] - key_bbx_reshaped[:, 2]
    key_bbx_reshaped[:, 2] = key_bbx_reshaped[::-1, 2]
    key_bbx_normed = key_bbx_reshaped * key_spacing_norm_ratio
    key_point_fine_gt = (key_bbx_reshaped[1, :] + key_bbx_reshaped[0, :]) / 2 * key_spacing_norm_ratio

    query_point_fine_re = np.floor(query_point_fine / 2.).astype(int)
    query_point_fine_re_matrix = make_local_grid_patch(query_point_fine_re, margin_x, margin_y, margin_z,
                                                       query_img_info[0].shape[4],
                                                       query_img_info[0].shape[3], query_img_info[0].shape[2])

    if stable_matching == False:
        pts_query = query_point_fine_re_matrix
        if semantic:
            pts_key_final_inter, s_inter, pts_key, s = get_sim_semantic_embed_loc_multi_embedding_space(query_img_info,
                                                                                                        key_img_info,
                                                                                                        pts_query,
                                                                                                        cuda_device,
                                                                                                        return_interploted=True)
        else:
            pts_key_final_inter, s_inter, pts_key, s = get_sim_embed_loc_multi_embedding_space(query_img_info,
                                                                                               key_img_info,
                                                                                               pts_query,
                                                                                               cuda_device,
                                                                                               return_interploted=True)
        all = np.array(pts_key_final_inter)
        pre = np.array([all[:, 0].mean(), all[:, 1].mean(), all[:, 2].mean()])

    if stable_matching:
        iters = 4
        for ii in range(iters):
            if ii == 0:
                pts_query = query_point_fine_re_matrix
            if ii % 2 == 0:
                if semantic:
                    pts_key_inter, s_inter, pts_key, s = get_sim_semantic_embed_loc_multi_embedding_space(
                        query_img_info, key_img_info, pts_query, cuda_device, return_interploted=True)
                else:
                    pts_key_inter, s_inter, pts_key, s = get_sim_embed_loc_multi_embedding_space(
                        query_img_info, key_img_info, pts_query, cuda_device, return_interploted=True)
                pts_key_final_inter = pts_key_inter
                if ii == 0:
                    pts_key_final_inter_direct = pts_key_inter
            else:
                if semantic:
                    pts_key_inter, s_inter, pts_key, s = get_sim_semantic_embed_loc_multi_embedding_space(
                        key_img_info, query_img_info, pts_query, cuda_device, return_interploted=True)
                else:
                    pts_key_inter, s_inter, pts_key, s = get_sim_embed_loc_multi_embedding_space(
                        key_img_info, query_img_info, pts_query, cuda_device, return_interploted=True)
                pts_query_final_inter = pts_key_inter
            pts_query = pts_key

        dis = np.linalg.norm(pts_query_final_inter - query_point_fine,
                             axis=1)
        final_q_all = []
        final_k_all = []
        final_score_all = []

        for i in range(dis.shape[0]):
            if dis[i] < 100 and s_inter[i] > 1.3:
                final_q_all.append(pts_query_final_inter[i, :])
                final_k_all.append(pts_key_final_inter[i, :])
                final_score_all.append(s_inter[i])

        final_q_all = np.array(final_q_all)
        final_k_all = np.array(final_k_all)
        final_score_all = np.array(final_score_all)
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
        final_q = np.array(final_q).reshape(-1,3)
        final_k = np.array(final_k).reshape(-1,3)

        k_xs = np.unique(final_k[:, 0])
        k_ys = np.unique(final_k[:, 1])
        k_zs = np.unique(final_k[:, 2])
        q_zs = np.unique(final_q[:, 2])

        if  k_zs.shape[0] == 1:
            print('z:',pair_id)
        if  k_ys.shape[0] == 1:
            print('y:',pair_id)
        if  k_xs.shape[0] == 1:
            print('x:',pair_id)

        shift = final_q - query_point_fine
        ones = np.ones((final_q.shape[0], 1))
        final_q_ext = np.concatenate((final_q, ones), axis=1)
        m = []
        if k_zs.shape[0] == 1:
            z_ratro = 1.
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
        pt_k_shift = np.where(pt_k_shift < 0, 0, pt_k_shift)
        if semantic:
            max_shape = np.array(key_img_info[3].shape[2:]) - 1
        else:
            max_shape = np.array(key_img_info[2].shape[2:]) - 1
        max_shape = np.array([max_shape[1], max_shape[2], max_shape[0]])
        pt_k_shift = np.where(pt_k_shift > max_shape, max_shape, pt_k_shift)
        pt_k_shift_mean = np.mean(pt_k_shift, axis=0)
        pre_s = pt_k_shift_mean
        all_direct = np.array(pts_key_final_inter_direct)
        pre_direct = np.array([all_direct[:, 0].mean(), all_direct[:, 1].mean(), all_direct[:, 2].mean()])
        dis_s_d = np.linalg.norm(pre_s - pre_direct)
        # if dis_s_d > 10:
        #     pre = pre_direct
        # else:
        #     pre = pre_s
        pre = pre_s
    pre = pre + 0.5
    pre_remap = pre / key_spacing_norm_ratio
    dis = np.linalg.norm(key_point_fine_gt - pre)
    dis_x = np.linalg.norm(key_point_fine_gt[0] - pre[0])
    dis_y = np.linalg.norm(key_point_fine_gt[1] - pre[1])
    dis_z = np.linalg.norm(key_point_fine_gt[2] - pre[2])
    flag = 0
    final = key_bbx_reshaped - pre_remap
    if (final[0, :] <= 0).all() and (final[1, :] > -0).all():
        flag = 1
    else:
        flag = 0

    final = [flag, dis, dis_x, dis_y, dis_z,pair_id]
    return final


def infer_element_init(q):
    infer_element.q = q


def run_pool():
    set_start_method('spawn')
    queue = Queue()
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)
    gths = torch.load(gth_path)
    all_keys = list(gths.keys())
    all_elements = []
    for i in range(all_keys.__len__()):
        for element in gths[all_keys[i]]:
            all_elements.append(element)
    with Pool(processes=PROC_PER_GPU * NUM_GPUS) as pool:
        allresults = pool.map(infer_element, all_elements)
    return allresults


if __name__ == '__main__':
    start = time()
    results = run_pool()
    hits = 0
    miss = 0
    dis = []
    dis_x = []
    dis_y = []
    dis_z = []
    all_pair = []
    for result in results:
        if result[0] == 1:
            hits += 1
        else:
            miss += 1
        dis.append(result[1])
        dis_x.append(result[2])
        dis_y.append(result[3])
        dis_z.append(result[4])
        all_pair.append(result[5])
    dis = np.array(dis) * 2.
    dis_x = np.array(dis_x) * 2.
    dis_y = np.array(dis_y) * 2.
    dis_z = np.array(dis_z) * 2.
    all_pair = np.array(all_pair)
    print(hits / (hits + miss))
    print(dis.mean(), dis.std(), dis.max())
    print((dis <= 10).sum() / (hits + miss))
    print(dis_x.mean(), dis_x.std(), dis_x.max())
    print(dis_y.mean(), dis_y.std(), dis_y.max())
    print(dis_z.mean(), dis_z.std(), dis_z.max())
    end = time()
    print(end - start)
    print(all_pair[dis>60])
    print(dis[dis > 60])
