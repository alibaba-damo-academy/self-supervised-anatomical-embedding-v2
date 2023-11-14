# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Written by Xiaoyu Bai
# rearrange the ignore computation into useindex, save computation cost
import os.path
import warnings

import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
import torch.nn.functional as F
from torch import linalg as LA
import time
import pickle
import numpy as np

import math
import matplotlib.pyplot as plt
from ..losses import SupConLoss
import random


@DETECTORS.register_module()
class Sam_uniform_cross_volume_dual_head_mri_ct(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 read_out_head=None,
                 sem_neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Sam_uniform_cross_volume_dual_head_mri_ct, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if neck is not None:
            self.neck = build_neck(neck)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.supcriterion = SupConLoss()

        self.read_out_head = build_neck(read_out_head)
        if sem_neck is not None:
            self.semantic_head = build_neck(sem_neck)
        else:
            self.semantic_head = build_neck(neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img, normalize=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        out1 = self.neck(x[:self.neck.end_level])[0]
        out2 = self.read_out_head([x[-1]])[0]
        out3 = self.semantic_head(x[:self.semantic_head.end_level])[0]
        if normalize:
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)
            out3 = F.normalize(out3, dim=1)
        out1 = out1.type(torch.half)
        out2 = out2.type(torch.half)
        out3 = out3.type(torch.half)
        return [out1, out2, out3]

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        outs = outs + x

        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):

        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'img_name',etc.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        all_intra_info = []
        all_intra_img = []
        all_inter_info = []
        all_inter_img = []

        p = np.random.rand()

        for i in range(img_metas.__len__()):
            im_info = img_metas[i][0]
            if im_info['style'] == 'intra':
                all_intra_info.append(im_info)
                all_intra_img.append(img[i])
            if im_info['style'] == 'inter':
                all_inter_info.append(im_info)
                all_inter_img.append(img[i])

        intra_imgs = torch.stack(all_intra_img)
        mesh_grids = []
        valids = []
        for i in range(all_intra_info.__len__()):
            mesh = all_intra_info[i]['meshgrid']
            mesh_grids.append(mesh)
            valid = all_intra_info[i]['valid']
            valids.append(valid)
            all_intra_info[i].pop('meshgrid')
            all_intra_info[i].pop('valid')
        mesh_grids = torch.stack(mesh_grids).to(intra_imgs.device)
        valids = torch.stack(valids).to(intra_imgs.device)

        losses = dict()
        if p < self.train_cfg.meta_cfg.prob:
            inter_dict = {}
            for image, info in zip(all_inter_img, all_inter_info):
                pair_name = info['filename'].split('_', 1)[0]
                if pair_name not in inter_dict.keys():
                    inter_dict[pair_name] = {}
                if info['tag'] == 'CT':
                    inter_dict[pair_name]['CT'] = {}
                    inter_dict[pair_name]['CT']['filename'] = info['filename']
                    inter_dict[pair_name]['CT']['mask'] = info['mask']
                    inter_dict[pair_name]['CT']['meshgrid'] = info['meshgrid']
                    inter_dict[pair_name]['CT']['image'] = image
                    inter_dict[pair_name]['CT']['spacing'] =  info['resample_spacing']

                if info['tag'] == 'MRI':
                    inter_dict[pair_name]['MRI'] = {}
                    inter_dict[pair_name]['MRI']['filename'] = info['filename']
                    inter_dict[pair_name]['MRI']['mask'] = info['mask']
                    inter_dict[pair_name]['MRI']['meshgrid'] = info['meshgrid']
                    inter_dict[pair_name]['MRI']['image'] = image
                    inter_dict[pair_name]['MRI']['spacing'] = info['resample_spacing']

            for pair in inter_dict:
                mri_mask = inter_dict[pair]['MRI']['mask']
                mri_mesh = inter_dict[pair]['MRI']['meshgrid']
                ct_mask = inter_dict[pair]['CT']['mask']
                ct_mesh = inter_dict[pair]['CT']['meshgrid']
                mri_inds = torch.where(mri_mask == 1)
                mri_min = torch.tensor([ele.min() for ele in mri_inds])
                mri_max = torch.tensor([ele.max() for ele in mri_inds])
                mri_loc_z_y_x = torch.stack([mri_min, mri_max])[:, 1:].type(torch.float32).permute(1, 0)
                ct_inds = torch.where(ct_mask == 1)
                ct_min = torch.tensor([ele.min() for ele in ct_inds])
                ct_max = torch.tensor([ele.max() for ele in ct_inds])
                ct_loc_z_y_x = torch.stack([ct_min, ct_max])[:, 1:].type(torch.float32).permute(1, 0)

                # if mri_loc_z_y_x[0,1] - mri_loc_z_y_x[0,0] > 90:
                #     half =(mri_loc_z_y_x[0,1] - mri_loc_z_y_x[0,0])/2
                #     p_c = np.random.rand()
                #     if p_c < 0.5:
                #         mri_loc_z_y_x[0, 1] = mri_loc_z_y_x[0,0] + half
                #     else:
                #         mri_loc_z_y_x[0, 0] += half
                # spacing is y x z
                ct_spacing = inter_dict[pair]['CT']['spacing']
                mri_spacing =inter_dict[pair]['MRI']['spacing']
                # print(ct_spacing,mri_spacing)
                ratio = mri_spacing / ct_spacing
                ratio = torch.tensor([ratio[2], ratio[0], ratio[1]],dtype=mri_loc_z_y_x.dtype)
                trans_M = torch.zeros((3, 3), dtype=ratio.dtype).scatter_(0, torch.arange(0, 3).view(-1, 3),
                                                                          ratio.view(-1, 3))
                # print(trans_M)
                mri_shape = torch.tensor(mri_mask.shape[1:], dtype=torch.float32)
                ct_shape = torch.tensor(ct_mask.shape[1:], dtype=torch.float32)
                mri_ct_map_loc = torch.matmul(trans_M, mri_loc_z_y_x)
                mri_ct_map_loc = torch.where(mri_ct_map_loc < 0, torch.tensor(0, dtype=mri_ct_map_loc.dtype),
                                             mri_ct_map_loc)
                mri_ct_map_loc[:, 1] = torch.where(mri_ct_map_loc[:, 1] >= ct_shape - 1, ct_shape - 1,
                                                   mri_ct_map_loc[:, 1])
                mri_loc_z_y_x_remap = torch.matmul(torch.linalg.inv(trans_M), mri_ct_map_loc)
                margin = torch.tensor([104, 128, 128])
                possible_range = mri_loc_z_y_x_remap[:, 1] - mri_loc_z_y_x_remap[:, 0] - margin

                if possible_range[0] >= 1:
                    z_s = np.random.randint(0, int(possible_range[0]))
                    z_end = z_s + margin[0]
                else:
                    z_s = 0
                    z_end = int(mri_loc_z_y_x_remap[0, 1] - mri_loc_z_y_x_remap[0, 0])

                if possible_range[1] >= 1:
                    y_s = np.random.randint(0, int(possible_range[1]))
                    y_end = y_s + margin[1]
                else:
                    y_s = 0
                    y_end = int(mri_loc_z_y_x_remap[1, 1] - mri_loc_z_y_x_remap[1, 0])
                if possible_range[2] >= 1:
                    x_s = np.random.randint(0, int(possible_range[2]))
                    x_end = x_s + margin[2]
                else:
                    x_s = 0
                    x_end = int(mri_loc_z_y_x_remap[2, 1] - mri_loc_z_y_x_remap[2, 0])

                mri_crop_region = torch.tensor([z_s,z_end,y_s,y_end,x_s,x_end]).reshape(3,2)
                mri_crop_region = mri_crop_region +  mri_loc_z_y_x_remap[:,0:1]

                ct_crop_region = torch.matmul(trans_M, mri_crop_region)
                jitter_shift = 10
                jitter_ct_loc = torch.randint(-jitter_shift, jitter_shift, (3, 2))
                # print('before', mri_loc_z_y_x, mri_ct_map_loc, inter_dict[pair]['CT']['filename'])
                mri_ct_map_loc = ct_crop_region + jitter_ct_loc
                jitter_mri_loc = torch.randint(-jitter_shift, jitter_shift, (3, 2))
                mri_loc_z_y_x = mri_crop_region + jitter_mri_loc
                mri_ct_map_loc = torch.where(mri_ct_map_loc<0,torch.tensor(0,dtype=mri_ct_map_loc.dtype),mri_ct_map_loc)
                mri_loc_z_y_x = torch.where(mri_loc_z_y_x<0,torch.tensor(0,dtype=mri_loc_z_y_x.dtype),mri_loc_z_y_x)
                mri_loc_z_y_x[:,1] = torch.where(mri_loc_z_y_x[:, 1] >= mri_shape - 1, mri_shape - 1, mri_loc_z_y_x[:, 1])
                mri_ct_map_loc[:,1] = torch.where(mri_ct_map_loc[:, 1] >= ct_shape - 1, ct_shape - 1, mri_ct_map_loc[:, 1])
                mri_ct_map_loc = mri_ct_map_loc.type(torch.int32)
                mri_loc_z_y_x = mri_loc_z_y_x.type(torch.int32)
                inter_dict[pair]['MRI']['crop_info'] = mri_loc_z_y_x
                inter_dict[pair]['CT']['crop_info'] = mri_ct_map_loc
            all_inter_feats = []
            all_inter_meshgrid = []
            all_inter_mask = []
            for pair in inter_dict:
                CT_image = inter_dict[pair]['CT']['image']
                CT_crop_loc = inter_dict[pair]['CT']['crop_info']
                CT_mask = inter_dict[pair]['CT']['mask']
                CT_mesh = inter_dict[pair]['CT']['meshgrid']
                MRI_image = inter_dict[pair]['MRI']['image']
                MRI_crop_loc = inter_dict[pair]['MRI']['crop_info']
                MRI_mask = inter_dict[pair]['MRI']['mask']
                MRI_mesh = inter_dict[pair]['MRI']['meshgrid']
                feat = self.extract_feat(
                    CT_image[:, CT_crop_loc[0, 0]:CT_crop_loc[0, 1], CT_crop_loc[1, 0]:CT_crop_loc[1, 1],
                    CT_crop_loc[2, 0]:CT_crop_loc[2, 1]].unsqueeze(0))
                all_inter_feats.append(feat)
                all_inter_meshgrid.append(
                    CT_mesh[:, CT_crop_loc[0, 0]:CT_crop_loc[0, 1], CT_crop_loc[1, 0]:CT_crop_loc[1, 1],
                    CT_crop_loc[2, 0]:CT_crop_loc[2, 1]])
                all_inter_mask.append(
                    CT_mask[:, CT_crop_loc[0, 0]:CT_crop_loc[0, 1], CT_crop_loc[1, 0]:CT_crop_loc[1, 1],
                    CT_crop_loc[2, 0]:CT_crop_loc[2, 1]])
                feat = self.extract_feat(
                    MRI_image[:, MRI_crop_loc[0, 0]:MRI_crop_loc[0, 1], MRI_crop_loc[1, 0]:MRI_crop_loc[1, 1],
                    MRI_crop_loc[2, 0]:MRI_crop_loc[2, 1]].unsqueeze(0))
                all_inter_feats.append(feat)
                all_inter_meshgrid.append(
                    MRI_mesh[:, MRI_crop_loc[0, 0]:MRI_crop_loc[0, 1], MRI_crop_loc[1, 0]:MRI_crop_loc[1, 1],
                    MRI_crop_loc[2, 0]:MRI_crop_loc[2, 1]])
                all_inter_mask.append(
                    MRI_mask[:, MRI_crop_loc[0, 0]:MRI_crop_loc[0, 1], MRI_crop_loc[1, 0]:MRI_crop_loc[1, 1],
                    MRI_crop_loc[2, 0]:MRI_crop_loc[2, 1]])

            loss = self.loss_inter(all_inter_feats, all_inter_meshgrid, all_inter_mask)
        else:
            feats = self.extract_feat(intra_imgs)
            loss = self.loss_intra(feats, mesh_grids, all_intra_info, valids)
        losses['loss'] = loss
        return losses

    def loss_intra(self, feats, meshgrid, img_metas, valid, *args):

        N, C, D, H, W = meshgrid.shape
        output_half = F.interpolate(meshgrid, size=(int(D / 2), int(H / 2), int(W / 2)), mode='trilinear')
        output_416 = F.interpolate(meshgrid, size=(int(D / 4), int(H / 16), int(W / 16)), mode='trilinear')
        valid_half = F.interpolate(valid, size=(int(D / 2), int(H / 2), int(W / 2)))
        valid_416 = F.interpolate(valid, size=(int(D / 4), int(H / 16), int(W / 16)))

        output_half = output_half.type(torch.half)
        output_416 = output_416.type(torch.half)
        valid_half = valid_half.type(torch.half)
        valid_416 = valid_416.type(torch.half)

        # fine_time_start = time.time()

        for i in range(int(N / 2)):
            result = self.single_intra_fine_loss(feats[0][2 * i:2 * (i + 1)], feats[1][2 * i:2 * (i + 1)],
                                                 output_half[2 * i:2 * (i + 1)], output_416[2 * i:2 * (i + 1)],
                                                 valid_half[2 * i:2 * (i + 1)], valid_416[2 * i:2 * (i + 1)])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 2)
        result = self.single_intra_coarse_loss(feats[1], output_416, valid_416)
        coarse_loss = result['loss']

        loss = fine_loss + coarse_loss
        return loss

    def loss_inter(self, feats, meshs, masks):
        N = feats.__len__()

        for i in range(int(N / 2)):
            result_fine = self.single_inter_fine_loss([feats[2 * i], feats[2 * i + 1]],
                                                 [meshs[2 * i], meshs[2 * i + 1]], [masks[2 * i], masks[2 * i + 1]])
            if i == 0:
                fine_loss = result_fine['loss']
            else:
                fine_loss += result_fine['loss']
        fine_loss = fine_loss / int(N / 2)
        for i in range(int(N / 2)):
            result_coarse = self.single_inter_coarse_loss([feats[2 * i], feats[2 * i + 1]],
                                                   [meshs[2 * i], meshs[2 * i + 1]], [masks[2 * i], masks[2 * i + 1]])
            if i == 0:
                coarse_loss = result_coarse['loss']
            else:
                coarse_loss += result_coarse['loss']
        coarse_loss = coarse_loss / int(N / 2)
        loss = fine_loss + coarse_loss
        return loss

    def single_inter_fine_loss(self, feat, mesh, mask):
        out = dict()
        ct_fine_feat = feat[0][0]
        ct_coarse_feat = feat[0][1]
        ct_mesh = mesh[0].to(ct_fine_feat.device)
        ct_mask = mask[0].to(ct_fine_feat.device)
        ct_local_grid = self.meshgrid3d(ct_fine_feat.shape[2:]).to(ct_fine_feat.device)
        mri_fine_feat = feat[1][0]
        mri_coarse_feat = feat[1][1]
        mri_mesh = mesh[1].to(mri_fine_feat.device)
        mri_mask = mask[1].to(mri_fine_feat.device)
        mri_local_grid = self.meshgrid3d(mri_fine_feat.shape[2:])

        ct_fine_mesh = F.interpolate(ct_mesh.unsqueeze(0), size=ct_fine_feat.shape[2:], mode='trilinear').type(
            torch.half)
        ct_coarse_mesh = F.interpolate(ct_mesh.unsqueeze(0), size=ct_coarse_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_fine_mesh = F.interpolate(mri_mesh.unsqueeze(0), size=mri_fine_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_coarse_mesh = F.interpolate(mri_mesh.unsqueeze(0), size=mri_coarse_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_fine_mask = F.interpolate(mri_mask.type(mri_mesh.dtype).unsqueeze(0), size=mri_fine_feat.shape[2:]).type(
            torch.half)

        mri_fine_foregrond_points = mri_fine_mesh[:, :, mri_fine_mask[0, 0, :, :, :] == 1].squeeze(0)
        mri_fine_foregrond_points_loc = mri_local_grid[mri_fine_mask[0, 0, :, :, :] == 1, :]
        ct_fine_points_loc = ct_local_grid.view(-1, 3)
        mri_fine_points_loc = mri_local_grid.view(-1,3)

        points_select = torch.randperm(mri_fine_foregrond_points.shape[1])[
                        :self.train_cfg.inter_cfg.pre_select_pos_number]

        mri_fine_select_points = mri_fine_foregrond_points[:, points_select]
        mri_fine_all_points = mri_fine_mesh.view(3, -1)
        ct_fine_all_points = ct_fine_mesh.view(3, -1)

        with torch.no_grad():
            mri_ct_dist = LA.norm((mri_fine_select_points.view(-1, mri_fine_select_points.shape[1], 1) - \
                                   ct_fine_all_points.view(-1, 1, ct_fine_all_points.shape[1])), dim=0)
            mri_mri_dist = LA.norm((mri_fine_select_points.view(-1, mri_fine_select_points.shape[1], 1) - \
                                    mri_fine_all_points.view(-1, 1, mri_fine_all_points.shape[1])), dim=0)

        pos_match = torch.unique(torch.where(mri_ct_dist < self.train_cfg.inter_cfg.positive_distance)[0])

        if pos_match.shape[0] == 0:
            pos_match = torch.unique(torch.where(mri_ct_dist < mri_ct_dist.min() + 0.5)[0])
        if pos_match.shape[0] <= self.train_cfg.inter_cfg.after_select_pos_number:
            pos_match = pos_match
        else:
            pos_match = pos_match[torch.randperm(pos_match.shape[0])[:self.train_cfg.intra_cfg.after_select_pos_number]]

        mri_ct_dist = mri_ct_dist[pos_match, :]
        mri_mri_dist = mri_mri_dist[pos_match, :]

        mri_source = points_select[pos_match]
        ct_target = mri_ct_dist.min(dim=1)[1]

        mri_mri_ignore = torch.where(mri_mri_dist < self.train_cfg.inter_cfg.ignore_distance)
        mri_ct_ignore = torch.where(mri_ct_dist < self.train_cfg.inter_cfg.ignore_distance)
        mri_mri_ignore = torch.stack(mri_mri_ignore)
        mri_ct_ignore = torch.stack(mri_ct_ignore)

        mri_mri_mask = torch.ones_like(mri_mri_dist)
        mri_ct_mask = torch.ones_like(mri_ct_dist)

        mri_mri_mask[mri_mri_ignore[0, :], mri_mri_ignore[1, :]] = 0
        mri_ct_mask[mri_ct_ignore[0, :], mri_ct_ignore[1, :]] = 0

        neg_mask_1 = torch.cat([mri_mri_mask, mri_ct_mask], dim=1)

        mri_coarse_feat_resample = F.interpolate(mri_coarse_feat, mri_fine_feat.shape[2:], mode='trilinear')
        ct_coarse_feat_resample = F.interpolate(ct_coarse_feat, ct_fine_feat.shape[2:], mode='trilinear')
        mri_coarse_feat_resample = F.normalize(mri_coarse_feat_resample, dim=1)
        ct_coarse_feat_resample = F.normalize(ct_coarse_feat_resample, dim=1)

        mri_location = mri_fine_foregrond_points_loc[mri_source].type(torch.LongTensor)
        ct_location = ct_fine_points_loc[ct_target].type(torch.LongTensor)


        ct_fine_selection_points = ct_fine_mesh[0,:,ct_location[:,0],ct_location[:,1],ct_location[:,2]]

        with torch.no_grad():
            ct_mri_dist = LA.norm((ct_fine_selection_points.view(-1, ct_fine_selection_points.shape[1], 1) - \
                                   mri_fine_all_points.view(-1, 1, mri_fine_all_points.shape[1])), dim=0)
            ct_ct_dist = LA.norm((ct_fine_selection_points.view(-1, ct_fine_selection_points.shape[1], 1) - \
                                    ct_fine_all_points.view(-1, 1, ct_fine_all_points.shape[1])), dim=0)

        ct_ct_ignore = torch.where(ct_ct_dist < self.train_cfg.inter_cfg.ignore_distance)
        ct_mri_ignore = torch.where(ct_mri_dist < self.train_cfg.inter_cfg.ignore_distance)
        ct_ct_ignore = torch.stack(ct_ct_ignore)
        ct_mri_ignore = torch.stack(ct_mri_ignore)

        ct_ct_mask = torch.ones_like(ct_ct_dist)
        ct_mri_mask = torch.ones_like(ct_mri_dist)

        ct_ct_mask[ct_ct_ignore[0, :], ct_ct_ignore[1, :]] = 0
        ct_mri_mask[ct_mri_ignore[0, :], ct_mri_ignore[1, :]] = 0

        neg_mask_2 = torch.cat([ct_mri_mask,ct_ct_mask], dim=1)

        q_mri_fine_feat = mri_fine_feat[0, :, mri_location[:, 0], mri_location[:, 1],
                          mri_location[:, 2]].transpose(0, 1)
        k_ct_fine_feat = ct_fine_feat[0, :, ct_location[:, 0], ct_location[:, 1],
                         ct_location[:, 2]].transpose(0, 1)

        q_mri_coarse_feat = mri_coarse_feat_resample[0, :, mri_location[:, 0], mri_location[:, 1],
                            mri_location[:, 2]].transpose(0, 1)
        k_ct_coarse_feat = ct_coarse_feat_resample[0, :, ct_location[:, 0], ct_location[:, 1],
                         ct_location[:, 2]].transpose(0, 1)

        inner_view = torch.einsum("nc,nc->n", q_mri_fine_feat, k_ct_fine_feat).view(-1, 1)

        neg_fine_view_1 = torch.einsum("nc,ck->nk", q_mri_fine_feat, torch.cat(
            (mri_fine_feat[0, :, :, :, :].view(128, -1), ct_fine_feat[0, :, :, :, :].view(128, -1)), dim=1))
        neg_coarse_view_1 = torch.einsum("nc,ck->nk", q_mri_coarse_feat,
                                       torch.cat((mri_coarse_feat_resample[0, :, :, :, :].view(128, -1),
                                                  ct_coarse_feat_resample[0, :, :, :, :].view(128, -1)), dim=1))

        # print('compute pos and neg time:',time4-time3)

        neg_all_view_1 = neg_fine_view_1 + neg_coarse_view_1
        neg_all_view_1 = neg_all_view_1 * neg_mask_1

        neg_candidate_view_index_1 = neg_all_view_1.topk(self.train_cfg.inter_cfg.pre_select_neg_number, dim=1)[1]

        neg_use_view_1 = torch.zeros((q_mri_fine_feat.shape[0], self.train_cfg.inter_cfg.after_select_neg_number),
                                   device=neg_all_view_1.device)

        for i in range(q_mri_fine_feat.shape[0]):
            use_index = neg_candidate_view_index_1[i, torch.randperm(neg_candidate_view_index_1[i, :].shape[0])[
                                                    :self.train_cfg.inter_cfg.after_select_neg_number]]
            neg_use_view_1[i, :] = neg_all_view_1[i, use_index]

        logits_view_1 = torch.cat([inner_view, neg_use_view_1], dim=1)

        neg_fine_view_2 = torch.einsum("nc,ck->nk", k_ct_fine_feat, torch.cat(
            (mri_fine_feat[0, :, :, :, :].view(128, -1), ct_fine_feat[0, :, :, :, :].view(128, -1)), dim=1))
        neg_coarse_view_2 = torch.einsum("nc,ck->nk", k_ct_coarse_feat,
                                         torch.cat((mri_coarse_feat_resample[0, :, :, :, :].view(128, -1),
                                                    ct_coarse_feat_resample[0, :, :, :, :].view(128, -1)), dim=1))

        # print('compute pos and neg time:',time4-time3)

        neg_all_view_2 = neg_fine_view_2 + neg_coarse_view_2
        neg_all_view_2 = neg_all_view_2 * neg_mask_2

        neg_candidate_view_index_2 = neg_all_view_2.topk(self.train_cfg.inter_cfg.pre_select_neg_number, dim=1)[1]

        neg_use_view_2 = torch.zeros((k_ct_fine_feat.shape[0], self.train_cfg.inter_cfg.after_select_neg_number),
                                     device=neg_all_view_2.device)

        for i in range(q_mri_fine_feat.shape[0]):
            use_index = neg_candidate_view_index_2[i, torch.randperm(neg_candidate_view_index_2[i, :].shape[0])[
                                                      :self.train_cfg.inter_cfg.after_select_neg_number]]
            neg_use_view_2[i, :] = neg_fine_view_2[i, use_index]

        logits_view_2 = torch.cat([inner_view, neg_use_view_2], dim=1)




        logits = torch.cat([logits_view_1, logits_view_2], dim=0)
        logits = logits / self.train_cfg.inter_cfg.fine_temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = self.criterion(logits, labels)
        out['loss'] = loss
        return out

    def single_inter_coarse_loss(self, feat, mesh, mask):
        out = dict()
        ct_fine_feat = feat[0][0]
        ct_coarse_feat = feat[0][1]
        ct_mesh = mesh[0].to(ct_fine_feat.device)
        ct_mask = mask[0].to(ct_fine_feat.device)
        ct_local_grid = self.meshgrid3d(ct_fine_feat.shape[2:]).to(ct_fine_feat.device)
        mri_fine_feat = feat[1][0]
        mri_coarse_feat = feat[1][1]
        mri_mesh = mesh[1].to(mri_fine_feat.device)
        mri_mask = mask[1].to(mri_fine_feat.device)
        mri_local_grid = self.meshgrid3d(mri_fine_feat.shape[2:])

        ct_fine_mesh = F.interpolate(ct_mesh.unsqueeze(0), size=ct_fine_feat.shape[2:], mode='trilinear').type(
            torch.half)
        ct_coarse_mesh = F.interpolate(ct_mesh.unsqueeze(0), size=ct_coarse_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_fine_mesh = F.interpolate(mri_mesh.unsqueeze(0), size=mri_fine_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_coarse_mesh = F.interpolate(mri_mesh.unsqueeze(0), size=mri_coarse_feat.shape[2:], mode='trilinear').type(
            torch.half)
        mri_fine_mask = F.interpolate(mri_mask.type(mri_mesh.dtype).unsqueeze(0), size=mri_fine_feat.shape[2:]).type(
            torch.half)

        mri_coarse_mesh_flatten =mri_coarse_mesh.view(3,-1)
        ct_coarse_mesh_flatten = ct_coarse_mesh.view(3,-1)

        with torch.no_grad():
            dist_mri_ct = LA.norm((mri_coarse_mesh_flatten.view(-1, mri_coarse_mesh_flatten.shape[1], 1) - \
                            ct_coarse_mesh_flatten.view(-1, 1, ct_coarse_mesh_flatten.shape[1])), dim=0)
            dist_z = LA.norm((mri_coarse_mesh_flatten[2, :].view(-1, mri_coarse_mesh_flatten.shape[1], 1) - \
                                ct_coarse_mesh_flatten[2, :].view(-1, 1, ct_coarse_mesh_flatten.shape[1])), dim=0)

            dist_mri_mri = LA.norm((mri_coarse_mesh_flatten.view(-1, mri_coarse_mesh_flatten.shape[1], 1) - \
                                mri_coarse_mesh_flatten.view(-1, 1, mri_coarse_mesh_flatten.shape[1])), dim=0)
            dist_ct_ct = LA.norm((ct_coarse_mesh_flatten.view(-1, ct_coarse_mesh_flatten.shape[1], 1) - \
                                ct_coarse_mesh_flatten.view(-1, 1, ct_coarse_mesh_flatten.shape[1])), dim=0)
            dist_ct_mri = dist_mri_ct.transpose(1,0)
        pos_coarse_can = torch.unique(
            torch.where(dist_mri_ct < self.train_cfg.inter_cfg.coarse_positive_distance)[0])

        key_coarse_can = dist_mri_ct[pos_coarse_can].min(dim=1)[1]
        pos_coarse = pos_coarse_can[dist_z[pos_coarse_can, key_coarse_can] < 8]
        key_coarse = key_coarse_can[dist_z[pos_coarse_can, key_coarse_can] < 8]

        mri_coarse_feat_flatten = mri_coarse_feat.view(mri_coarse_feat.shape[1],-1)
        ct_coarse_feat_flatten = ct_coarse_feat.view(ct_coarse_feat.shape[1],-1)

        q_mri_coarse_feat = mri_coarse_feat_flatten[:, pos_coarse].transpose(0, 1)
        k_ct_coarse_feat = ct_coarse_feat_flatten[:, key_coarse].transpose(0, 1)


        dist_mri_ct = dist_mri_ct[pos_coarse, :]
        dist_mri_mri = dist_mri_mri[pos_coarse, :]
        dist_ct_ct = dist_ct_ct[key_coarse,:]
        dist_ct_mri = dist_ct_mri[key_coarse,:]

        ind_ignores_mri_mri = torch.where(dist_mri_mri < self.train_cfg.inter_cfg.coarse_ignore_distance)
        ind_ignores_mri_ct = torch.where(dist_mri_ct < self.train_cfg.inter_cfg.coarse_ignore_distance)
        ind_ignores_ct_ct = torch.where(dist_ct_ct < self.train_cfg.inter_cfg.coarse_ignore_distance)
        ind_ignores_ct_mri = torch.where(dist_ct_mri < self.train_cfg.inter_cfg.coarse_ignore_distance)

        ind_ignores_mri_mri = torch.stack(ind_ignores_mri_mri)
        ind_ignores_mri_ct = torch.stack(ind_ignores_mri_ct)
        ind_ignores_ct_ct = torch.stack(ind_ignores_ct_ct)
        ind_ignores_ct_mri = torch.stack(ind_ignores_ct_mri)

        use_mask_mri_mri = torch.ones_like(dist_mri_mri)
        use_mask_mri_ct = torch.ones_like(dist_mri_ct)
        use_mask_ct_ct = torch.ones_like(dist_ct_ct)
        use_mask_ct_mri = torch.ones_like(dist_ct_mri)

        use_mask_mri_mri[ind_ignores_mri_mri[0, :], ind_ignores_mri_mri[1, :]] = 0
        use_mask_mri_ct[ind_ignores_mri_ct[0, :], ind_ignores_mri_ct[1, :]] = 0
        use_mask_ct_ct[ind_ignores_ct_ct[0, :], ind_ignores_ct_ct[1, :]] = 0
        use_mask_ct_mri[ind_ignores_ct_mri[0,:],ind_ignores_ct_mri[1,:]] = 0
        # # neg_mask = torch.cat([use_mask_mri_mri, use_mask_mri_ct], dim=1)
        # neg_mask =  use_mask_mri_ct
        neg_mask_1 = torch.cat([use_mask_mri_mri, use_mask_mri_ct], dim=1)
        neg_mask_2 = torch.cat([use_mask_ct_mri, use_mask_ct_ct], dim=1)

        inner_view = torch.einsum("nc,nc->n", q_mri_coarse_feat, k_ct_coarse_feat).view(-1, 1)


        neg_coarse_view_1 = torch.einsum("nc,ck->nk", q_mri_coarse_feat,
                                        torch.cat((mri_coarse_feat_flatten, ct_coarse_feat_flatten), dim=1))

        neg_coarse_view_2 = torch.einsum("nc,ck->nk", k_ct_coarse_feat,
                                        torch.cat((mri_coarse_feat_flatten, ct_coarse_feat_flatten), dim=1))
        # neg_coarse_view = torch.einsum("nc,ck->nk", q_mri_coarse_feat,ct_coarse_feat_flatten)

        neg_coarse_view_1 = neg_coarse_view_1 * neg_mask_1
        neg_coarse_view_2 = neg_coarse_view_2 * neg_mask_2


        neg_candidate_coarse_index_1 = \
            neg_coarse_view_1.topk(self.train_cfg.inter_cfg.coarse_pre_select_neg_number, dim=1)[1]

        neg_use_coarse_1 = torch.zeros(
            (q_mri_coarse_feat.shape[0], self.train_cfg.inter_cfg.coarse_after_select_neg_number),device=neg_coarse_view_1.device)

        for i in range(q_mri_coarse_feat.shape[0]):
            use_index = neg_candidate_coarse_index_1[i,torch.randperm(neg_candidate_coarse_index_1.shape[1])][
                        :self.train_cfg.inter_cfg.coarse_after_select_neg_number]
            neg_use_coarse_1[i, :use_index.shape[0]] = neg_coarse_view_1[i, use_index]

        neg_candidate_coarse_index_2 = \
            neg_coarse_view_2.topk(self.train_cfg.inter_cfg.coarse_pre_select_neg_number, dim=1)[1]

        neg_use_coarse_2 = torch.zeros(
            (k_ct_coarse_feat.shape[0], self.train_cfg.inter_cfg.coarse_after_select_neg_number),
            device=neg_coarse_view_2.device)

        for i in range(q_mri_coarse_feat.shape[0]):
            use_index = neg_candidate_coarse_index_2[i, torch.randperm(neg_candidate_coarse_index_2.shape[1])][
                        :self.train_cfg.inter_cfg.coarse_after_select_neg_number]
            neg_use_coarse_2[i, :use_index.shape[0]] = neg_coarse_view_1[i, use_index]

        logits_view_1 = torch.cat([inner_view, neg_use_coarse_1], dim=1)
        logits_view_2 = torch.cat([inner_view, neg_use_coarse_2], dim=1)
        logits_view = torch.cat([logits_view_1,logits_view_2],dim=0)
        logits = logits_view / self.train_cfg.inter_cfg.coarse_temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = self.criterion(logits, labels)
        out['loss'] = loss
        return out

    def single_intra_fine_loss(self, fine_feat, coarse_feat, fine_grid, coarse_grid, fine_vaild, coarse_valid, *args):

        view_1_fine = fine_feat[0, :, :, :, :].view(128, -1)
        view_2_fine = fine_feat[1, :, :, :, :].view(128, -1)

        out = dict()

        fine_local_grid = self.meshgrid3d(fine_feat.shape[2:])  # z y x

        p1_y_min = fine_grid[0, 0, fine_vaild[0, 0] > 0].min()
        p1_y_max = fine_grid[0, 0, fine_vaild[0, 0] > 0].max()
        p1_x_min = fine_grid[0, 1, fine_vaild[0, 0] > 0].min()
        p1_x_max = fine_grid[0, 1, fine_vaild[0, 0] > 0].max()
        p1_z_min = fine_grid[0, 2, fine_vaild[0, 0] > 0].min()
        p1_z_max = fine_grid[0, 2, fine_vaild[0, 0] > 0].max()

        p2_y_min = fine_grid[1, 0, fine_vaild[1, 0] > 0].min()
        p2_y_max = fine_grid[1, 0, fine_vaild[1, 0] > 0].max()
        p2_x_min = fine_grid[1, 1, fine_vaild[1, 0] > 0].min()
        p2_x_max = fine_grid[1, 1, fine_vaild[1, 0] > 0].max()
        p2_z_min = fine_grid[1, 2, fine_vaild[1, 0] > 0].min()
        p2_z_max = fine_grid[1, 2, fine_vaild[1, 0] > 0].max()

        intersection_y = [max(p1_y_min, p2_y_min), min(p1_y_max, p2_y_max)]
        intersection_x = [max(p1_x_min, p2_x_min), min(p1_x_max, p2_x_max)]
        intersection_z = [max(p1_z_min, p2_z_min), min(p1_z_max, p2_z_max)]

        intersection_volume_sub1 = (fine_grid[0, 0] >= intersection_y[0]) * (
                fine_grid[0, 0] <= intersection_y[1]) \
                                   * (fine_grid[0, 1] >= intersection_x[0]) * (
                                           fine_grid[0, 1] <= intersection_x[1]) \
                                   * (fine_grid[0, 2] >= intersection_z[0]) * (
                                           fine_grid[0, 2] <= intersection_z[1]) * fine_vaild[0, 0]

        intersection_volume_sub2 = (fine_grid[1, 0] >= intersection_y[0]) * (
                fine_grid[1, 0] <= intersection_y[1]) \
                                   * (fine_grid[1, 1] >= intersection_x[0]) * (
                                           fine_grid[1, 1] <= intersection_x[1]) \
                                   * (fine_grid[1, 2] >= intersection_z[0]) * (
                                           fine_grid[1, 2] <= intersection_z[1]) * fine_vaild[1, 0]
        pos1_index_overlap = fine_local_grid[intersection_volume_sub1 > 0, :]
        pos2_index_overlap = fine_local_grid[intersection_volume_sub2 > 0, :]
        pos_index_all = fine_local_grid.view(-1, 3)
        # yxz pos_mm_overlap      zyx pos_index
        pos1_mm_overlap = fine_grid[0, :, intersection_volume_sub1 > 0]
        pos2_mm_overlap = fine_grid[1, :, intersection_volume_sub2 > 0]
        view1_num_points = pos1_mm_overlap.shape[1]
        view2_num_points = pos2_mm_overlap.shape[1]

        if intersection_y[0] > intersection_y[1] or intersection_x[0] > intersection_x[1] or intersection_z[0] > \
                intersection_z[1] or view1_num_points == 0 or view2_num_points==0:

            # no overlap
            index_use = torch.randperm(fine_feat.shape[2] * fine_feat.shape[3] * fine_feat.shape[4])
            q_index = index_use[:self.train_cfg.intra_cfg.after_select_pos_number]
            fine_local_grid_flatten = fine_local_grid.view(-1, 3)

            q_loc = fine_local_grid_flatten[q_index, :].type(torch.LongTensor)
            pos1_mm_use = fine_grid[0, :, q_loc[:, 0], q_loc[:, 1], q_loc[:, 2]]
            pos1_mm_all = fine_grid[0, :, :, :, :].view(3, -1)
            dist1_1 = LA.norm((pos1_mm_use.view(-1, pos1_mm_use.shape[1], 1) - \
                               pos1_mm_all.view(-1, 1, pos1_mm_all.shape[1])), dim=0)
            ind_ignores_1_1 = torch.where(dist1_1 < self.train_cfg.intra_cfg.ignore_distance)
            ind_ignores_1_1 = torch.stack(ind_ignores_1_1)
            use_mask_1_1 = torch.ones_like(dist1_1)
            use_mask_1_1[ind_ignores_1_1[0, :], ind_ignores_1_1[1, :]] = 0

            q_fine_feat = fine_feat[0, :, q_loc[:, 0], q_loc[:, 1], q_loc[:, 2]]
            q_fine_feat = q_fine_feat.permute(1, 0)

            neg_fine = torch.einsum("nc,ck->nk", q_fine_feat, view_1_fine)
            neg_fine = neg_fine * use_mask_1_1
            inner = torch.einsum("nc,nc->n", q_fine_feat, q_fine_feat).view(-1, 1)

            neg_fine_use_index = neg_fine.topk(self.train_cfg.intra_cfg.after_select_neg_number)[1]
            neg_fine_use = torch.zeros((neg_fine.shape[0], self.train_cfg.intra_cfg.after_select_neg_number),
                                       device=fine_feat.device)
            for i in range(neg_fine.shape[0]):
                neg_fine_use[i, :] = neg_fine[i, neg_fine_use_index[i, :]]
            logits = torch.cat([inner, neg_fine_use], dim=1)
            logits = logits / self.train_cfg.intra_cfg.fine_temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
            loss = self.criterion(logits, labels)
        else:
            # have overlap
            # intersection_volume_sub1 = (fine_grid[0, 0] >= intersection_y[0]) * (
            #         fine_grid[0, 0] <= intersection_y[1]) \
            #                            * (fine_grid[0, 1] >= intersection_x[0]) * (
            #                                    fine_grid[0, 1] <= intersection_x[1]) \
            #                            * (fine_grid[0, 2] >= intersection_z[0]) * (
            #                                    fine_grid[0, 2] <= intersection_z[1]) * fine_vaild[0, 0]
            #
            # intersection_volume_sub2 = (fine_grid[1, 0] >= intersection_y[0]) * (
            #         fine_grid[1, 0] <= intersection_y[1]) \
            #                            * (fine_grid[1, 1] >= intersection_x[0]) * (
            #                                    fine_grid[1, 1] <= intersection_x[1]) \
            #                            * (fine_grid[1, 2] >= intersection_z[0]) * (
            #                                    fine_grid[1, 2] <= intersection_z[1]) * fine_vaild[1, 0]
            # pos1_index_overlap = fine_local_grid[intersection_volume_sub1 > 0, :]
            # pos2_index_overlap = fine_local_grid[intersection_volume_sub2 > 0, :]
            # pos_index_all = fine_local_grid.view(-1, 3)
            # # yxz pos_mm_overlap      zyx pos_index
            # pos1_mm_overlap = fine_grid[0, :, intersection_volume_sub1 > 0]
            # pos2_mm_overlap = fine_grid[1, :, intersection_volume_sub2 > 0]
            # view1_num_points = pos1_mm_overlap.shape[1]
            # view2_num_points = pos2_mm_overlap.shape[1]

            if view1_num_points < self.train_cfg.intra_cfg.pre_select_pos_number:
                points1_select = torch.randperm(view1_num_points)
            else:
                points1_select = torch.randperm(view1_num_points)[:self.train_cfg.intra_cfg.pre_select_pos_number]
            if view2_num_points < self.train_cfg.intra_cfg.pre_select_pos_number:
                points2_select = torch.randperm(view2_num_points)
            else:
                points2_select = torch.randperm(view2_num_points)[:self.train_cfg.intra_cfg.pre_select_pos_number]

            pos1_use = pos1_mm_overlap[:, points1_select]
            pos2_use = pos2_mm_overlap[:, points2_select]

            pos1_mm_all = fine_grid[0, :, :, :, :].view(3, -1)
            pos2_mm_all = fine_grid[1, :, :, :, :].view(3, -1)

            # time1 = time.time()

            with torch.no_grad():

                dist1_1 = LA.norm((pos1_use.view(-1, pos1_use.shape[1], 1) - \
                                   pos1_mm_all.view(-1, 1, pos1_mm_all.shape[1])), dim=0)

                dist1_2 = LA.norm((pos1_use.view(-1, pos1_use.shape[1], 1) - \
                                   pos2_mm_all.view(-1, 1, pos2_mm_all.shape[1])), dim=0)
                dist2_1 = LA.norm((pos2_use.view(-1, pos2_use.shape[1], 1) - \
                                   pos1_mm_all.view(-1, 1, pos1_mm_all.shape[1])), dim=0)
                dist2_2 = LA.norm((pos2_use.view(-1, pos2_use.shape[1], 1) - \
                                   pos2_mm_all.view(-1, 1, pos2_mm_all.shape[1])), dim=0)

            # time2 = time.time()
            # print('dist time:',time2-time1)

            pos1 = torch.unique(torch.where(dist1_2 < self.train_cfg.intra_cfg.positive_distance)[0])
            pos2 = torch.unique(torch.where(dist2_1 < self.train_cfg.intra_cfg.positive_distance)[0])

            if pos1.shape[0] == 0:
                print(dist1_2.min())
                pos1 = torch.unique(torch.where(dist1_2 < dist1_2.min() + 0.5)[0])
                pos2 = torch.unique(torch.where(dist2_1 < dist2_1.min() + 0.5)[0])
            if pos1.shape[0] <= self.train_cfg.intra_cfg.after_select_pos_number:
                pos1 = pos1
            else:
                pos1 = pos1[torch.randperm(pos1.shape[0])[:self.train_cfg.intra_cfg.after_select_pos_number]]

            if pos2.shape[0] <= self.train_cfg.intra_cfg.after_select_pos_number:
                pos2 = pos2
            else:
                pos2 = pos2[torch.randperm(pos2.shape[0])[:self.train_cfg.intra_cfg.after_select_pos_number]]

            dist1_1 = dist1_1[pos1, :]
            dist1_2 = dist1_2[pos1, :]
            dist2_1 = dist2_1[pos2, :]
            dist2_2 = dist2_2[pos2, :]

            pos1_source = points1_select[pos1]
            pos2_source = points2_select[pos2]
            pos1_target = dist1_2.min(dim=1)[1]
            pos2_target = dist2_1.min(dim=1)[1]

            ind_ignores_1_1 = torch.where(dist1_1 < self.train_cfg.intra_cfg.ignore_distance)
            ind_ignores_1_2 = torch.where(dist1_2 < self.train_cfg.intra_cfg.ignore_distance)
            ind_ignores_2_1 = torch.where(dist2_1 < self.train_cfg.intra_cfg.ignore_distance)
            ind_ignores_2_2 = torch.where(dist2_2 < self.train_cfg.intra_cfg.ignore_distance)
            ind_ignores_1_1 = torch.stack(ind_ignores_1_1)
            ind_ignores_1_2 = torch.stack(ind_ignores_1_2)
            ind_ignores_2_1 = torch.stack(ind_ignores_2_1)
            ind_ignores_2_2 = torch.stack(ind_ignores_2_2)

            use_mask_1_1 = torch.ones_like(dist1_1)
            use_mask_1_2 = torch.ones_like(dist1_2)
            use_mask_2_1 = torch.ones_like(dist2_1)
            use_mask_2_2 = torch.ones_like(dist2_2)
            use_mask_1_1[ind_ignores_1_1[0, :], ind_ignores_1_1[1, :]] = 0
            use_mask_1_2[ind_ignores_1_2[0, :], ind_ignores_1_2[1, :]] = 0
            use_mask_2_1[ind_ignores_2_1[0, :], ind_ignores_2_1[1, :]] = 0
            use_mask_2_2[ind_ignores_2_2[0, :], ind_ignores_2_2[1, :]] = 0

            neg_mask1 = torch.cat([use_mask_1_1, use_mask_1_2], dim=1)
            neg_mask2 = torch.cat([use_mask_2_1, use_mask_2_2], dim=1)

            coarse_feat_resample = F.interpolate(coarse_feat, fine_feat.shape[2:], mode='trilinear')
            coarse_feat_resample = F.normalize(coarse_feat_resample, dim=1)
            view_1_coarse = coarse_feat_resample[0, :, :, :, :].view(128, -1)
            view_2_coarse = coarse_feat_resample[1, :, :, :, :].view(128, -1)

            q_view1_location = pos1_index_overlap[pos1_source].type(torch.LongTensor)
            k_view1_location = pos_index_all[pos1_target].type(torch.LongTensor)
            q_view2_location = pos2_index_overlap[pos2_source].type(torch.LongTensor)
            k_view2_location = pos_index_all[pos2_target].type(torch.LongTensor)

            q_view1_fine_feat = fine_feat[0, :, q_view1_location[:, 0], q_view1_location[:, 1],
                                q_view1_location[:, 2]].transpose(0, 1)
            k_view1_fine_feat = fine_feat[1, :, k_view1_location[:, 0], k_view1_location[:, 1],
                                k_view1_location[:, 2]].transpose(0, 1)
            q_view2_fine_feat = fine_feat[1, :, q_view2_location[:, 0], q_view2_location[:, 1],
                                q_view2_location[:, 2]].transpose(0, 1)
            k_view2_fine_feat = fine_feat[0, :, k_view2_location[:, 0], k_view2_location[:, 1],
                                k_view2_location[:, 2]].transpose(0, 1)

            q_view1_coarse_feat = coarse_feat_resample[0, :, q_view1_location[:, 0], q_view1_location[:, 1],
                                  q_view1_location[:, 2]].transpose(0, 1)
            q_view2_coarse_feat = coarse_feat_resample[1, :, q_view2_location[:, 0], q_view2_location[:, 1],
                                  q_view2_location[:, 2]].transpose(0, 1)

            inner_view1 = torch.einsum("nc,nc->n", q_view1_fine_feat, k_view1_fine_feat).view(-1, 1)
            inner_view2 = torch.einsum("nc,nc->n", q_view2_fine_feat, k_view2_fine_feat).view(-1, 1)

            neg_fine_view1 = torch.einsum("nc,ck->nk", q_view1_fine_feat, torch.cat((view_1_fine, view_2_fine), dim=1))
            neg_coarse_view1 = torch.einsum("nc,ck->nk", q_view1_coarse_feat,
                                            torch.cat((view_1_coarse, view_2_coarse), dim=1))
            neg_fine_view2 = torch.einsum("nc,ck->nk", q_view2_fine_feat, torch.cat((view_1_fine, view_2_fine), dim=1))
            neg_coarse_view2 = torch.einsum("nc,ck->nk", q_view2_coarse_feat,
                                            torch.cat((view_1_coarse, view_2_coarse), dim=1))

            # print('compute pos and neg time:',time4-time3)

            neg_all_view1 = neg_fine_view1 + neg_coarse_view1
            neg_all_view2 = neg_fine_view2 + neg_coarse_view2
            neg_all_view1 = neg_all_view1 * neg_mask1
            neg_all_view2 = neg_all_view2 * neg_mask2

            neg_candidate_view1_index = neg_all_view1.topk(self.train_cfg.intra_cfg.pre_select_neg_number, dim=1)[1]
            neg_candidate_view2_index = neg_all_view2.topk(self.train_cfg.intra_cfg.pre_select_neg_number, dim=1)[
                1]  # find diverse and hard negatives

            neg_use_view1 = torch.zeros((q_view1_fine_feat.shape[0], self.train_cfg.intra_cfg.after_select_neg_number),
                                        device=neg_all_view1.device)
            neg_use_view2 = torch.zeros((q_view2_fine_feat.shape[0], self.train_cfg.intra_cfg.after_select_neg_number),
                                        device=neg_all_view2.device)
            for i in range(q_view1_fine_feat.shape[0]):
                use_index = neg_candidate_view1_index[i, torch.randperm(neg_candidate_view1_index[i, :].shape[0])[
                                                         :self.train_cfg.intra_cfg.after_select_neg_number]]
                neg_use_view1[i, :] = neg_fine_view1[i, use_index]

            for i in range(q_view2_fine_feat.shape[0]):
                use_index = neg_candidate_view2_index[i, torch.randperm(neg_candidate_view2_index[i, :].shape[0])[
                                                         :self.train_cfg.intra_cfg.after_select_neg_number]]

                neg_use_view2[i, :] = neg_fine_view2[i, use_index]

            logits_view1 = torch.cat([inner_view1, neg_use_view1], dim=1)
            logits_view2 = torch.cat([inner_view2, neg_use_view2], dim=1)
            logits = torch.cat([logits_view1, logits_view2], dim=0)
            logits = logits / self.train_cfg.intra_cfg.fine_temperature

            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
            loss = self.criterion(logits, labels)
        out['loss'] = loss
        return out

    def single_intra_coarse_loss(self, coarse_feat, coarse_grid, coarse_valid, *args):  # coarse_grid[B,C,D,H,W] C:y,x,z
        coarse_feat_flatten = coarse_feat.view(coarse_feat.shape[0], 128, -1)
        coarse_feat_flatten = coarse_feat_flatten.permute(1, 0, 2)
        coarse_feat_flatten = coarse_feat_flatten.reshape(128, -1)

        out = dict()

        for i in range(int(coarse_feat.shape[0] / 2)):
            view1_feat_coarse = coarse_feat[2 * i, :, :, :, :].view(128, -1)
            view2_feat_coarse = coarse_feat[2 * i + 1, :, :, :, :].view(128, -1)
            view1_loc_coarse = coarse_grid[2 * i, :, :, :, :].view(3, -1)
            view2_loc_coarse = coarse_grid[2 * i + 1, :, :, :, :].view(3, -1)

            global_use_index_list = [j + 2 * i * view1_feat_coarse.shape[1] for j in
                                     range(0, 2 * view1_feat_coarse.shape[1])]
            global_index_list = [j for j in range(0, coarse_feat.shape[0] * view1_feat_coarse.shape[1])]
            global_search_index_list = list(set(global_index_list) - set(global_use_index_list))
            global_search_index = torch.tensor(global_search_index_list).to(view1_feat_coarse.device)

            with torch.no_grad():
                dist = LA.norm((view1_loc_coarse.view(-1, view1_loc_coarse.shape[1], 1) - \
                                view2_loc_coarse.view(-1, 1, view2_loc_coarse.shape[1])), dim=0)
                dist_z_1 = LA.norm((view1_loc_coarse[2, :].view(-1, view1_loc_coarse.shape[1], 1) - \
                                    view2_loc_coarse[2, :].view(-1, 1, view2_loc_coarse.shape[1])), dim=0)
                dist_z_2 = LA.norm((view2_loc_coarse[2, :].view(-1, view2_loc_coarse.shape[1], 1) - \
                                    view1_loc_coarse[2, :].view(-1, 1, view1_loc_coarse.shape[1])), dim=0)
                dist_1_1 = LA.norm((view1_loc_coarse.view(-1, view1_loc_coarse.shape[1], 1) - \
                                    view1_loc_coarse.view(-1, 1, view1_loc_coarse.shape[1])), dim=0)
                dist_2_2 = LA.norm((view2_loc_coarse.view(-1, view2_loc_coarse.shape[1], 1) - \
                                    view2_loc_coarse.view(-1, 1, view2_loc_coarse.shape[1])), dim=0)
            pos_view1_coarse_can = torch.unique(
                torch.where(dist < self.train_cfg.intra_cfg.coarse_positive_distance)[0])

            key_view1_coarse_can = dist[pos_view1_coarse_can].min(dim=1)[1]
            pos_view1_coarse = pos_view1_coarse_can[dist_z_1[pos_view1_coarse_can, key_view1_coarse_can] < 8]
            key_view1_coarse = key_view1_coarse_can[dist_z_1[pos_view1_coarse_can, key_view1_coarse_can] < 8]
            pos_view2_coarse_can = torch.unique(
                torch.where(dist.transpose(0, 1) < self.train_cfg.intra_cfg.coarse_positive_distance)[0])
            key_view2_coarse_can = dist.transpose(0, 1)[pos_view2_coarse_can].min(dim=1)[1]
            pos_view2_coarse = pos_view2_coarse_can[
                dist_z_2[pos_view2_coarse_can, key_view2_coarse_can] < self.train_cfg.intra_cfg.coarse_z_thres]
            key_view2_coarse = key_view2_coarse_can[
                dist_z_2[pos_view2_coarse_can, key_view2_coarse_can] < self.train_cfg.intra_cfg.coarse_z_thres]

            if pos_view1_coarse.shape[0] > 0:
                q_view1_coarse_feat = view1_feat_coarse[:, pos_view1_coarse].transpose(0, 1)
                k_view1_coarse_feat = view2_feat_coarse[:, key_view1_coarse].transpose(0, 1)
                q_view2_coarse_feat = view2_feat_coarse[:, pos_view2_coarse].transpose(0, 1)
                k_view2_coarse_feat = view1_feat_coarse[:, key_view2_coarse].transpose(0, 1)

                # all_nodes_view1 = []
                # all_nodes_view2 = []
                dist_1_2 = dist.clone()
                dist_2_1 = dist.transpose(0, 1).clone()
                dist_1_2 = dist_1_2[pos_view1_coarse, :]
                dist_1_1 = dist_1_1[pos_view1_coarse, :]
                dist_2_1 = dist_2_1[pos_view2_coarse, :]
                dist_2_2 = dist_2_2[pos_view2_coarse, :]

                ind_ignores_1_1 = torch.where(dist_1_1 < self.train_cfg.intra_cfg.coarse_ignore_distance)
                ind_ignores_1_2 = torch.where(dist_1_2 < self.train_cfg.intra_cfg.coarse_ignore_distance)
                ind_ignores_2_1 = torch.where(dist_2_1 < self.train_cfg.intra_cfg.coarse_ignore_distance)
                ind_ignores_2_2 = torch.where(dist_2_2 < self.train_cfg.intra_cfg.coarse_ignore_distance)
                ind_ignores_1_1 = torch.stack(ind_ignores_1_1)
                ind_ignores_1_2 = torch.stack(ind_ignores_1_2)
                ind_ignores_2_1 = torch.stack(ind_ignores_2_1)
                ind_ignores_2_2 = torch.stack(ind_ignores_2_2)

                use_mask_1_1 = torch.ones_like(dist_1_1)
                use_mask_1_2 = torch.ones_like(dist_1_2)
                use_mask_2_1 = torch.ones_like(dist_2_1)
                use_mask_2_2 = torch.ones_like(dist_2_2)

                use_mask_1_1[ind_ignores_1_1[0, :], ind_ignores_1_1[1, :]] = 0
                use_mask_1_2[ind_ignores_1_2[0, :], ind_ignores_1_2[1, :]] = 0
                use_mask_2_1[ind_ignores_2_1[0, :], ind_ignores_2_1[1, :]] = 0
                use_mask_2_2[ind_ignores_2_2[0, :], ind_ignores_2_2[1, :]] = 0

                neg_mask1 = torch.cat([use_mask_1_1, use_mask_1_2], dim=1)
                neg_mask2 = torch.cat([use_mask_2_1, use_mask_2_2], dim=1)

                inner_view1 = torch.einsum("nc,nc->n", q_view1_coarse_feat, k_view1_coarse_feat).view(-1, 1)
                inner_view2 = torch.einsum("nc,nc->n", q_view2_coarse_feat, k_view2_coarse_feat).view(-1, 1)

                neg_coarse_view1 = torch.einsum("nc,ck->nk", q_view1_coarse_feat,
                                                torch.cat((view1_feat_coarse, view2_feat_coarse), dim=1))
                neg_coarse_view2 = torch.einsum("nc,ck->nk", q_view2_coarse_feat,
                                                torch.cat((view1_feat_coarse, view2_feat_coarse), dim=1))

                neg_coarse_view1 = neg_coarse_view1 * neg_mask1
                neg_coarse_view2 = neg_coarse_view2 * neg_mask2

                neg_candidate_view1_coarse_index = \
                    neg_coarse_view1.topk(self.train_cfg.intra_cfg.coarse_pre_select_neg_number, dim=1)[1]
                neg_candidate_view2_coarse_index = \
                    neg_coarse_view2.topk(self.train_cfg.intra_cfg.coarse_pre_select_neg_number, dim=1)[1]

                neg_use_view1_coarse = torch.zeros(
                    (q_view1_coarse_feat.shape[0], self.train_cfg.intra_cfg.coarse_after_select_neg_number))
                neg_use_view2_coarse = torch.zeros(
                    (q_view2_coarse_feat.shape[0], self.train_cfg.intra_cfg.coarse_after_select_neg_number))

                for i in range(q_view1_coarse_feat.shape[0]):
                    use_index = neg_candidate_view1_coarse_index[i,torch.randperm(neg_candidate_view1_coarse_index.shape[1])][
                                :self.train_cfg.intra_cfg.coarse_after_select_neg_number]
                    neg_use_view1_coarse[i, :use_index.shape[0]] = neg_coarse_view1[i, use_index]

                for i in range(q_view2_coarse_feat.shape[0]):
                    use_index = neg_candidate_view2_coarse_index[i,torch.randperm(neg_candidate_view2_coarse_index.shape[1])][
                                :self.train_cfg.intra_cfg.coarse_after_select_neg_number]
                    neg_use_view2_coarse[i, :use_index.shape[0]] = neg_coarse_view2[i, use_index]

                neg_coarse_global_index_1 = global_search_index[
                    torch.randperm(global_search_index_list.__len__())[
                    :self.train_cfg.intra_cfg.coarse_global_select_number]]
                neg_coarse_global_1 = coarse_feat_flatten[:, neg_coarse_global_index_1]
                neg_coarse_global_index_2 = global_search_index[
                    torch.randperm(global_search_index_list.__len__())[
                    :self.train_cfg.intra_cfg.coarse_global_select_number]]
                neg_coarse_global_2 = coarse_feat_flatten[:, neg_coarse_global_index_2]

                neg_coarse_global_view1 = torch.einsum("nc,ck->nk", q_view1_coarse_feat,
                                                       neg_coarse_global_1)
                neg_coarse_global_view2 = torch.einsum("nc,ck->nk", q_view2_coarse_feat,
                                                       neg_coarse_global_2)

                neg_view1_coarse_concate = torch.cat(
                    [neg_use_view1_coarse.to(view1_feat_coarse.device), neg_coarse_global_view1], dim=1)
                neg_view2_coarse_concate = torch.cat(
                    [neg_use_view2_coarse.to(view1_feat_coarse.device), neg_coarse_global_view2], dim=1)

                logits_view1 = torch.cat([inner_view1, neg_view1_coarse_concate], dim=1)
                logits_view2 = torch.cat([inner_view2, neg_view2_coarse_concate], dim=1)
                logits = torch.cat([logits_view1, logits_view2], dim=0)
                logits = logits / self.train_cfg.intra_cfg.coarse_temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                loss = self.criterion(logits, labels)
                if out.keys().__len__() == 0:
                    out['loss'] = loss
                else:
                    out['loss'] += loss
            else:
                neg_coarse_global_index_1 = global_search_index[
                    torch.randperm(global_search_index_list.__len__())[
                    :self.train_cfg.intra_cfg.coarse_global_select_number]]
                neg_coarse_global_1 = coarse_feat_flatten[:, neg_coarse_global_index_1]
                neg_coarse_global_index_2 = global_search_index[
                    torch.randperm(global_search_index_list.__len__())[
                    :self.train_cfg.intra_cfg.coarse_global_select_number]]
                neg_coarse_global_2 = coarse_feat_flatten[:, neg_coarse_global_index_2]

                neg_coarse_global_view1 = torch.einsum("nc,ck->nk", view1_feat_coarse.transpose(0, 1),
                                                       neg_coarse_global_1)
                neg_coarse_global_view2 = torch.einsum("nc,ck->nk", view2_feat_coarse.transpose(0, 1),
                                                       neg_coarse_global_2)

                logits_view1 = torch.einsum("nc,ck->nk", view1_feat_coarse.transpose(0, 1),
                                            torch.cat([view1_feat_coarse, view2_feat_coarse], dim=1)).topk(
                    self.train_cfg.intra_cfg.coarse_after_select_neg_number + 1,
                    dim=1)[
                    0]
                logits_view2 = torch.einsum("nc,ck->nk", view2_feat_coarse.transpose(0, 1),
                                            torch.cat([view1_feat_coarse, view2_feat_coarse], dim=1)).topk(
                    self.train_cfg.intra_cfg.coarse_after_select_neg_number + 1,
                    dim=1)[
                    0]

                logits_view1 = torch.cat([logits_view1, neg_coarse_global_view1], dim=1)
                logits_view2 = torch.cat([logits_view2, neg_coarse_global_view2], dim=1)
                logits = torch.cat([logits_view1, logits_view2], dim=0)
                logits = logits / self.train_cfg.intra_cfg.coarse_temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                loss = self.criterion(logits, labels)
                if out.keys().__len__() == 0:
                    out['loss'] = loss
                else:
                    out['loss'] += loss

        out['loss'] = out['loss'] / (coarse_feat.shape[0] / 2)
        return out

    def meshgrid3d(self, shape):
        z_ = torch.linspace(0., shape[0] - 1, shape[0])
        y_ = torch.linspace(0., shape[1] - 1, shape[1])
        x_ = torch.linspace(0., shape[2] - 1, shape[2])
        z, y, x = torch.meshgrid(z_, y_, x_)
        return torch.stack((z, y, x), 3)

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False,**kwargs):
        """Test without augmentation."""
        is_mri = img_metas[0]['is_mri']
        x = self.extract_feat(img)
        outs = []
        out1 = x[0]#.data.cpu().numpy()
        out2 = x[1]#.data.cpu().numpy()
        # out3 = x[2].data.cpu().numpy()
        outs = [out1, out2, img#.data.cpu().numpy()
                , img_metas[0]['filename'].split('.', 1)[0]]
        output_embedding = self.test_cfg.get('output_embedding', True)
        if not output_embedding:
            if not os.path.exists(self.test_cfg.save_path):
                os.mkdir(self.test_cfg.save_path)
            outfilename = self.test_cfg.save_path + \
                          img_metas[0]['filename'].split('.', 1)[0] + '.pkl'
            f = open(outfilename, 'wb')
            pickle.dump(outs, f)
            return [
                x[0][0, 0, 0, 0, 0].data.cpu()]  # we have saved the data into harddisk, this is just for fit the code
        else:
            return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def sample_index(self, mask, num_points, num_bg_points=1000):
        labels = torch.unique(mask).tolist()
        labels_organ = [label for label in labels if label > 0]
        if labels_organ.__len__() > 15:
            labels_organ = random.shuffle(labels_organ)
            labels_organ = labels_organ[:15]
        labels_organ.append(0)
        count = 0
        for label in labels_organ:
            all_ind = torch.where(mask == label)[0]
            if label == 0:
                thres = num_bg_points
            else:
                thres = num_points
            if all_ind.shape[0] > thres:
                use_count = thres
            else:
                use_count = all_ind.shape[0]
            pert = torch.randperm(all_ind.shape[0])[:use_count]
            use_ind = all_ind[pert]
            if count == 0:
                sample_index = use_ind
            else:
                sample_index = torch.cat((sample_index, use_ind))
            count = count + 1
        return sample_index


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
