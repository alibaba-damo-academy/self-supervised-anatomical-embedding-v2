# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.models import build_detector
from sam import *
import torch.nn.functional as F


def init(config, checkpoint):
    cfg = Config.fromfile(config)
    cfg.gpu_ids = range(1)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    return model


def get_embedding(batched_data, model):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **batched_data)
    return result


def normalize(pt, imshape):
    pt = np.array(pt)
    imshape = np.array(imshape)
    unit_space = 2 / np.array(imshape)
    origin = unit_space / 2 - 1
    pt_normed = origin + pt * unit_space
    return pt_normed


# infer matching point of the given (one) query_point (x,y,z). the query_point is on the norm image
# the final output matching point's location is depend on the imshape parameter.
# i.e. you can set the imshape equals to the original pre-normed key image size, then you can get the
# mapping point on the original key image
def get_sim_embed_loc(query_img_info, key_img_info, query_point, imshape, norm_info, write_sim=False, return_sim=False,
                      use_sim_coarse=True):
    query_point = np.array(query_point)
    cuda_device = torch.device('cuda:' + str(0))
    query_point_fine_re = np.floor(query_point / 2.).astype(int)
    fine_query = query_img_info[0]

    coarse_query = query_img_info[1]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_point_fine_re[2], query_point_fine_re[1], query_point_fine_re[0]].view(-1,
                                                                                                               128)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_point_fine_re[2], query_point_fine_re[1], query_point_fine_re[0]].view(
        -1, 128)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)

    sim_fine = sim_fine.reshape(fine_key.shape[2:])
    sim_coarse = sim_coarse.reshape(coarse_key.shape[2:])
    sim_fine = sim_fine.view(1, 1, sim_fine.shape[0], sim_fine.shape[1], sim_fine.shape[2])
    sim_coarse = sim_coarse.view(1, 1, sim_coarse.shape[0], sim_coarse.shape[1], sim_coarse.shape[2])
    if use_sim_coarse:
        sim = (sim_fine + sim_coarse) / 2
    else:
        sim = sim_fine
        # sim = sim_coarse
    # s im = F.interpolate(sim, tuple(key_img_info[2].shape[2:]), mode='trilinear', align_corners=False)
    sim = F.interpolate(sim, imshape, mode='trilinear', align_corners=False)
    if return_sim:
        return sim
    sim = sim[0, 0, :, :, :]
    ind = torch.where(sim == sim.max())
    z = int(ind[0][0].data.cpu().numpy())
    y = int(ind[1][0].data.cpu().numpy())
    x = int(ind[2][0].data.cpu().numpy())
    if write_sim:
        img = torch.tensor(key_img_info[2]).to(cuda_device)
        img = F.interpolate(img, imshape, mode='trilinear', align_corners=False)
        img_array = img[0, 0, :, :, :].data.cpu().numpy().astype('float32')
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing(norm_info)
        sitk.WriteImage(img_sitk, 'img.nii.gz')

        sim_map_array = sim.data.cpu().numpy().astype('float32')
        sim_map_array = sim_map_array
        sim_map = sitk.GetImageFromArray(sim_map_array)
        sim_map.SetSpacing(norm_info)
        sitk.WriteImage(sim_map, 'sim_map.nii.gz')

    # plt.imshow(torch.flip(sim[:,:,x].data.cpu(),(0,)).numpy())
    # plt.savefig('sim_2.png')
    return [x, y, z], sim.max().data.cpu().numpy()


def get_sim_embed_loc_multi_embedding_space(query_img_info, key_img_info, query_points):
    query_point = np.array(query_points)
    cuda_device = torch.device('cuda:' + str(0))
    fine_query = query_img_info[0]

    coarse_query = query_img_info[1]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_points[:, 2], query_points[:, 1],
                 query_points[:, 0]].transpose(1, 0)

    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_points[:, 2], query_points[:, 1],
                   query_points[:, 0]].transpose(1, 0)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)

    sim_fine = sim_fine.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                             fine_key.shape[4])
    sim_coarse = sim_coarse.view(query_points.shape[0], 1, coarse_key.shape[2], coarse_key.shape[3],
                                 coarse_key.shape[4])
    sim_all = (sim_fine + sim_coarse) / 2
    locs = []
    scores = []

    for i in range(sim_all.shape[0]):
        sim = sim_all[i, 0, :, :, :]
        ind = torch.where(sim == sim.max())
        z = ind[0][0].data.cpu().numpy()
        y = ind[1][0].data.cpu().numpy()
        x = ind[2][0].data.cpu().numpy()
        scores.append(sim.max().data.cpu().numpy())
        locs.append([x, y, z])
    return np.array(locs).astype(int), np.array(scores)


def get_sim_embed_semantic_loc(query_img_info, key_img_info, query_point, imshape, norm_info, write_sim=False,
                               return_sim=False, use_sim_coarse=True):
    query_point = np.array(query_point)
    cuda_device = torch.device('cuda:' + str(0))
    query_point_fine_re = np.floor(query_point / 2.).astype(int)
    fine_query = query_img_info[0]
    seman_query = query_img_info[2]

    coarse_query = query_img_info[1]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    seman_key = key_img_info[2]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_point_fine_re[2], query_point_fine_re[1], query_point_fine_re[0]].view(-1,
                                                                                                               128)
    query_sem = seman_query[0, :, query_point_fine_re[2], query_point_fine_re[1], query_point_fine_re[0]].view(-1,
                                                                                                               128)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)
    key_sem = seman_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_point_fine_re[2], query_point_fine_re[1], query_point_fine_re[0]].view(
        -1, 128)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_sem = torch.einsum("nc,ck->nk", query_sem, key_sem)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)

    sim_fine = sim_fine.reshape(fine_key.shape[2:])
    sim_sem = sim_sem.reshape(fine_key.shape[2:])
    sim_coarse = sim_coarse.reshape(coarse_key.shape[2:])
    sim_fine = sim_fine.view(1, 1, sim_fine.shape[0], sim_fine.shape[1], sim_fine.shape[2])
    sim_sem = sim_sem.view(1, 1, sim_sem.shape[0], sim_sem.shape[1], sim_sem.shape[2])
    sim_coarse = sim_coarse.view(1, 1, sim_coarse.shape[0], sim_coarse.shape[1], sim_coarse.shape[2])
    if use_sim_coarse:
        sim = (sim_fine + sim_coarse + sim_sem) / 3
        # sim = sim_sem
    else:
        sim = (sim_sem + sim_fine) / 2
        # sim = sim_coarse
    # sim = F.interpolate(sim, tuple(key_img_info[2].shape[2:]), mode='trilinear', align_corners=False)
    sim = F.interpolate(sim, imshape, mode='trilinear', align_corners=False)
    if return_sim:
        return sim
    sim = sim[0, 0, :, :, :]
    ind = torch.where(sim == sim.max())
    z = int(ind[0][0].data.cpu().numpy())
    y = int(ind[1][0].data.cpu().numpy())
    x = int(ind[2][0].data.cpu().numpy())
    if write_sim:
        img = torch.tensor(key_img_info[3]).to(cuda_device)
        img = F.interpolate(img, imshape, mode='trilinear', align_corners=False)
        img_array = img[0, 0, :, :, :].data.cpu().numpy().astype('float32')
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing(norm_info)
        sitk.WriteImage(img_sitk, 'img.nii.gz')

        sim_map_array = sim.data.cpu().numpy().astype('float32')
        sim_map_array = sim_map_array
        sim_map = sitk.GetImageFromArray(sim_map_array)
        sim_map.SetSpacing(norm_info)
        sitk.WriteImage(sim_map, 'sim_map.nii.gz')

    # plt.imshow(torch.flip(sim[:,:,x].data.cpu(),(0,)).numpy())
    # plt.savefig('sim_2.png')
    return [x, y, z], sim.max().data.cpu().numpy()


def get_sim_semantic_embed_loc_multi_embedding_space(query_img_info, key_img_info, query_points, cuda_device,
                                                     return_interploted=False):
    # cuda_device = torch.device('cuda:' + str(0))
    fine_query = query_img_info[0]

    coarse_query = query_img_info[1]
    sem_query = query_img_info[2]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    sem_key = key_img_info[2]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_points[:, 2], query_points[:, 1],
                 query_points[:, 0]].transpose(1, 0)

    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_points[:, 2], query_points[:, 1],
                   query_points[:, 0]].transpose(1, 0)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    query_sem = sem_query[0, :, query_points[:, 2], query_points[:, 1],
                query_points[:, 0]].transpose(1, 0)
    key_sem = sem_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)
    sim_sem = torch.einsum("nc,ck->nk", query_sem, key_sem)

    sim_fine = sim_fine.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                             fine_key.shape[4])
    sim_coarse = sim_coarse.view(query_points.shape[0], 1, coarse_key.shape[2], coarse_key.shape[3],
                                 coarse_key.shape[4])
    sim_sem = sim_sem.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                           fine_key.shape[4])
    sim_all = (sim_fine + sim_coarse + sim_sem) / 3
    # sim = sim_coarse
    if return_interploted:
        sim_inter = F.interpolate(sim_all, tuple(key_img_info[3].shape[2:]), mode='trilinear', align_corners=False)
        # sim_inter = sim_inter[:, 0, :, :, :]
        locs_inter = []
        scores_inter = []
        for i in range(sim_inter.shape[0]):
            sim_here = sim_inter[i, 0, :, :, :]
            ind = torch.where(sim_here == sim_here.max())
            z = ind[0][0].data.cpu().numpy()
            y = ind[1][0].data.cpu().numpy()
            x = ind[2][0].data.cpu().numpy()
            scores_inter.append(sim_here.max().data.cpu().numpy())
            locs_inter.append([x, y, z])
    locs = []
    scores = []

    # ind = torch.argmax(sim, dim=1).cpu().numpy()
    # zyx = np.unravel_index(ind, fine_key_vol.shape[2:])
    # xyz = np.vstack(zyx)[::-1] * cfg.local_emb_stride + .5  # add .5 to closer to stride center
    # xyz = xyz.T / key_data['im_norm_ratio']
    # xyz = np.minimum(np.round(xyz.astype(int)), np.array(key_data['im_shape'])[::-1] - 1)
    for i in range(sim_all.shape[0]):
        sim = sim_all[i, 0, :, :, :]
        ind = torch.where(sim == sim.max())
        z = ind[0][0].data.cpu().numpy()
        y = ind[1][0].data.cpu().numpy()
        x = ind[2][0].data.cpu().numpy()
        scores.append(sim.max().data.cpu().numpy())
        locs.append([x, y, z])
    if return_interploted:
        return np.array(locs_inter).astype(int), np.array(scores_inter), np.array(locs).astype(int), np.array(scores)
    else:
        return np.array(locs).astype(int), np.array(scores)


def get_sim_embed_loc_multi_embedding_space(query_img_info, key_img_info, query_points, cuda_device,
                                            return_interploted=False):
    fine_query = query_img_info[0]
    coarse_query = query_img_info[1]

    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_points[:, 2], query_points[:, 1],
                 query_points[:, 0]].transpose(1, 0)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_points[:, 2], query_points[:, 1],
                   query_points[:, 0]].transpose(1, 0)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)
    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)

    sim_fine = sim_fine.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                             fine_key.shape[4])
    sim_coarse = sim_coarse.view(query_points.shape[0], 1, coarse_key.shape[2], coarse_key.shape[3],
                                 coarse_key.shape[4])
    sim_all = (sim_fine + sim_coarse) / 2
    # sim = sim_coarse
    if return_interploted:
        sim_inter = F.interpolate(sim_all, tuple(key_img_info[2].shape[2:]), mode='trilinear', align_corners=False)
        # sim_inter = sim_inter[:, 0, :, :, :]
        locs_inter = []
        scores_inter = []
        for i in range(sim_inter.shape[0]):
            sim_here = sim_inter[i, 0, :, :, :]
            ind = torch.where(sim_here == sim_here.max())
            z = ind[0][0].data.cpu().numpy()
            y = ind[1][0].data.cpu().numpy()
            x = ind[2][0].data.cpu().numpy()
            scores_inter.append(sim_here.max().data.cpu().numpy())
            locs_inter.append([x, y, z])
    locs = []
    scores = []

    # ind = torch.argmax(sim, dim=1).cpu().numpy()
    # zyx = np.unravel_index(ind, fine_key_vol.shape[2:])
    # xyz = np.vstack(zyx)[::-1] * cfg.local_emb_stride + .5  # add .5 to closer to stride center
    # xyz = xyz.T / key_data['im_norm_ratio']
    # xyz = np.minimum(np.round(xyz.astype(int)), np.array(key_data['im_shape'])[::-1] - 1)
    for i in range(sim_all.shape[0]):
        sim = sim_all[i, 0, :, :, :]
        ind = torch.where(sim == sim.max())
        z = ind[0][0].data.cpu().numpy()
        y = ind[1][0].data.cpu().numpy()
        x = ind[2][0].data.cpu().numpy()
        scores.append(sim.max().data.cpu().numpy())
        locs.append([x, y, z])
    if return_interploted:
        return np.array(locs_inter).astype(int), np.array(scores_inter), np.array(locs).astype(int), np.array(scores)
    else:
        return np.array(locs).astype(int), np.array(scores)
