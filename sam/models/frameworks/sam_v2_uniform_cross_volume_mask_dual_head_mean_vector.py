# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Written by Xiaoyu Bai
# Rearrange the ignore computation into useindex, save computation cost
import os.path
import warnings

# import ipdb
import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from mmdet.models.detectors.base import BaseDetector
import torch.nn.functional as F
from torch import linalg as LA
import time
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from ..losses import SupConMeanLoss
import random


@DETECTORS.register_module()
class Sam_uniform_cross_volume_dual_head_mean_vector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 read_out_head=None,
                 sem_neck=None,
                 superloss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Sam_uniform_cross_volume_dual_head_mean_vector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if neck is not None:
            self.neck = build_neck(neck)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.supcriterion = build_loss(superloss)
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
        if p < 0.4:
            all_inter_feats = []
            for i, image in enumerate(all_inter_img):
                feat = self.extract_feat(image.unsqueeze(0))
                all_inter_feats.append(feat)
            loss = self.loss_inter(all_inter_feats, all_inter_info)
        else:
            feats = self.extract_feat(intra_imgs)
            loss = self.loss_intra(feats, mesh_grids, all_intra_info, valids)
        losses['loss'] = loss
        return losses

    def loss_intra(self, feats, meshgrid, img_metas, valid, *args):

        N, C, D, H, W = meshgrid.shape
        output_half = F.interpolate(meshgrid, size=(int(D / 2), int(H / 2), int(W / 2)), mode='trilinear', align_corners=False)
        output_416 = F.interpolate(meshgrid, size=(int(D / 4), int(H / 16), int(W / 16)), mode='trilinear', align_corners=False)
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

    def loss_inter(self, feats, img_metas, **kwargs):
        N = feats.__len__()

        for i in range(int(N / 2)):
            result = self.single_inter_loss([feats[2 * i], feats[2 * i + 1]],
                                            [img_metas[2 * i]['mask'], img_metas[2 * i + 1]['mask']])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 2)
        loss = fine_loss
        return loss

    def single_inter_loss(self, feat, mask):
        out = dict()
        view_1_fine = feat[0][2]
        view_2_fine = feat[1][2]
        view_1_fine = view_1_fine.view(view_1_fine.shape[1], -1).unsqueeze(1).permute(2, 1, 0)
        view_2_fine = view_2_fine.view(view_2_fine.shape[1], -1).unsqueeze(1).permute(2, 1, 0)
        mask_1 = mask[0].to(view_1_fine.device).type(torch.half)
        mask_2 = mask[1].to(view_2_fine.device).type(torch.half)
        mask_1_fine = F.interpolate(mask_1.unsqueeze(0), size=feat[0][2][0, :, :, :, :].shape[1:]).view(-1)
        mask_2_fine = F.interpolate(mask_2.unsqueeze(0), size=feat[1][2][0, :, :, :, :].shape[1:]).view(-1)
        # fine_1_means,fine_1_labels = self.get_mean_vector(mask_1_fine,view_1_fine)
        # fine_2_means,fine_2_labels = self.get_mean_vector(mask_2_fine,view_2_fine)
        fine_feats = torch.cat((view_1_fine, view_2_fine), dim=0)
        fine_labels = torch.cat((mask_1_fine, mask_2_fine), dim=0)
        fine_mean_feat, fine_mean_labels = self.get_mean_vector(fine_labels, fine_feats)
        # fine_mean_feat = torch.cat((fine_1_means,fine_2_means),dim=0)
        # fine_mean_labels = torch.cat((fine_1_labels,fine_2_labels),dim=0)
        loss = self.supcriterion(fine_feats, fine_labels, fine_mean_feat, fine_mean_labels)
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

        if intersection_y[0] > intersection_y[1] or intersection_x[0] > intersection_x[1] or intersection_z[0] > \
                intersection_z[1]:

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
            logits = logits / self.train_cfg.intra_cfg.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
            loss = self.criterion(logits, labels)
        else:
            # have overlap
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

            coarse_feat_resample = F.interpolate(coarse_feat, fine_feat.shape[2:], mode='trilinear', align_corners=False)
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
            logits = logits / self.train_cfg.intra_cfg.temperature

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
                    use_index = torch.randperm(neg_candidate_view1_coarse_index.shape[1])[
                                :self.train_cfg.intra_cfg.coarse_after_select_neg_number]
                    neg_use_view1_coarse[i, :use_index.shape[0]] = neg_coarse_view1[i, use_index]

                for i in range(q_view2_coarse_feat.shape[0]):
                    use_index = torch.randperm(neg_candidate_view2_coarse_index.shape[1])[
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
                logits = logits / self.train_cfg.intra_cfg.temperature
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
                logits = logits / self.train_cfg.intra_cfg.temperature
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        # outs = []
        out1 = x[0]#.data.cpu().numpy()
        out2 = x[1]#.data.cpu().numpy()
        out3 = x[2]#.data.cpu().numpy()
        outs = [out1, out2, out3, img.data,#.cpu().numpy(),
                img_metas[0]['filename'].split('.', 1)[0]]
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

    def get_mean_vector(self, mask, features):
        labels = torch.unique(mask).tolist()
        labels_organ = [label for label in labels if label > 0]
        mean_vectors = []
        for label in labels_organ:
            all_ind = torch.where(mask == label)[0]
            mean_vector = features[all_ind, 0, :].mean(dim=0)
            mean_vectors.append(mean_vector)
        mean_vectors = torch.stack(mean_vectors)
        labels_organ = torch.tensor(labels_organ)
        return mean_vectors, labels_organ


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
