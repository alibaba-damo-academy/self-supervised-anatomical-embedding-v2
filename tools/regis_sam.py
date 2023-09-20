# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Written by Xiaoyu Bai
# This code first uses SAM to do rigid registration on two images, and then uses DEEDS for deformable fine-tuning.
# The pipeline is:
# (1) compute SAM embeddings on 2x2x2 mm resampled images and find grid points matchings.
# (2) q,k are template points and their corresponding points on query image. q_o,k_o are points location
# remapped to original image space.
# (3) Using robust rigid estimator to compute the rigid transform matrix on resample images and original images (and do transform).
# (4) Do DEEDS deformation on rigid registered resampled images.
# (5) Apply the deformable field on original rigid registered images.

# for cross-modality training, it contains several stages of registration
# the first registration is based on the intra-training ckpts, and its results  are used for training the next stage SAM


import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from numpy.linalg import inv
from scipy import ndimage
from scipy import ndimage as ndi

from interfaces import get_sim_embed_loc_multi_embedding_space
from interfaces import init, get_embedding
from misc.deeds_preprocess_no_mask_origin_image import padding_img
from utils import read_image

np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation as ro

config_file = '/data/sdd/user/SAM/configs/samv2/samv2_ct_mri.py'
checkpoint_file = '/data/sdd/user/SAM/work_dirs/samv2_ct_mri/iter_20000_ct_mr_intra_only.pth'

im1_file = '/data/sdd/user/DEEDs/deedsBCV/mri.nii.gz'
im1_mask = '/data/sdd/user/DEEDs/deedsBCV/mri_mask.nii.gz'
im2_file = '/data/sdd/user/DEEDs/deedsBCV/ct.nii.gz'
im2_mask = '/data/sdd/user/DEEDs/deedsBCV/ct_mask.nii.gz'
deeds_path = '/data/sdd/user/DEEDs/deedsBCV/'
s_thres = 1.5


def compute_point(model, normed_im1, norm_info_1, normed_im2, norm_info_2):
    cuda_device = torch.device('cuda:' + str(0))
    emb1 = get_embedding(normed_im1, model)
    emb2 = get_embedding(normed_im2, model)

    shape1 = emb1[0].shape
    shape2 = emb2[0].shape
    img_1 = torch.tensor(emb1[2]).to(cuda_device)
    processed_1_img_downsample = F.interpolate(img_1, shape1[2:], mode='trilinear', align_corners=False)
    body_index = torch.where(processed_1_img_downsample > -25)
    z_min = body_index[2].min()
    z_max = body_index[2].max()
    y_min = body_index[3].min()
    y_max = body_index[3].max()
    x_min = body_index[4].min()
    x_max = body_index[4].max()

    x_strid = 12
    y_strid = 12
    z_strid = 8
    all_points_part1 = []  # 4 mini-batches because of  GPU memory limitation
    all_points_part2 = []
    all_points_part3 = []
    all_points_part4 = []
    for i in range(x_min, x_max, x_strid):
        for j in range(y_min, y_max, y_strid):
            for k in range(z_min+1, z_max, z_strid):
                all_points_part1.append([i, j, k])
    all_points_part1 = np.array(all_points_part1)
    for i in range(x_min + 5, x_max, x_strid):
        for j in range(y_min + 5, y_max, y_strid):
            for k in range(z_min + 4, z_max, z_strid):
                all_points_part2.append([i, j, k])
    all_points_part2 = np.array(all_points_part2)
    for i in range(x_min, x_max, x_strid):
        for j in range(y_min, y_max, y_strid):
            for k in range(z_min + 4, z_max, z_strid):
                all_points_part3.append([i, j, k])
    all_points_part3 = np.array(all_points_part3)
    for i in range(x_min + 5, x_max, x_strid):
        for j in range(y_min + 5, y_max, y_strid):
            for k in range(z_min+1, z_max, z_strid):
                all_points_part4.append([i, j, k])
    all_points_part4 = np.array(all_points_part4)

    its = 1
    for iter in range(its):
        if iter % 2 == 0:
            if iter == 0:
                q_loc_part1 = all_points_part1
            k_loc_part1, k_score_part1 = get_sim_embed_loc_multi_embedding_space(emb1, emb2, q_loc_part1, cuda_device)
            print(k_score_part1.mean())
        else:
            q_loc_part1, q_score_part1 = get_sim_embed_loc_multi_embedding_space(emb2, emb1, k_loc_part1, cuda_device)

    for iter in range(its):
        if iter % 2 == 0:
            if iter == 0:
                q_loc_part2 = all_points_part2
            k_loc_part2, k_score_part2 = get_sim_embed_loc_multi_embedding_space(emb1, emb2, q_loc_part2, cuda_device)
            print(k_score_part2.mean())
        else:
            q_loc_part2, q_score_part2 = get_sim_embed_loc_multi_embedding_space(emb2, emb1, k_loc_part2, cuda_device)

    for iter in range(its):
        if iter % 2 == 0:
            if iter == 0:
                q_loc_part3 = all_points_part3
            k_loc_part3, k_score_part3 = get_sim_embed_loc_multi_embedding_space(emb1, emb2, q_loc_part3, cuda_device)
            print(k_score_part3.mean())
        else:
            q_loc_part3, q_score_part3 = get_sim_embed_loc_multi_embedding_space(emb2, emb1, k_loc_part3, cuda_device)
    for iter in range(its):
        if iter % 2 == 0:
            if iter == 0:
                q_loc_part4 = all_points_part4
            k_loc_part4, k_score_part4 = get_sim_embed_loc_multi_embedding_space(emb1, emb2, q_loc_part4, cuda_device)
            print(k_score_part4.mean())
        else:
            q_loc_part4, q_score_part4 = get_sim_embed_loc_multi_embedding_space(emb2, emb1, k_loc_part4, cuda_device)

    q_loc = np.concatenate([q_loc_part1, q_loc_part2, q_loc_part3, q_loc_part4], axis=0)
    k_loc = np.concatenate([k_loc_part1, k_loc_part2, k_loc_part3, k_loc_part4], axis=0)
    k_score = np.concatenate([k_score_part1, k_score_part2, k_score_part3, k_score_part4], axis=0)
    print(k_score.mean(),k_score.max())
    q_loc = q_loc[k_score > s_thres, :]
    k_loc = k_loc[k_score > s_thres, :]
    q_loc = q_loc[:, ::-1].transpose(1, 0)
    k_loc = k_loc[:, ::-1].transpose(1, 0)
    q_loc = q_loc * 2 + 0.5  # zyx
    k_loc = k_loc * 2 + 0.5
    q_loc_origin = (q_loc.transpose(1, 0) / norm_info_1[::-1]).transpose(1, 0)
    k_loc_origin = (k_loc.transpose(1, 0) / norm_info_2[::-1]).transpose(1, 0)
    return q_loc, k_loc, q_loc_origin, k_loc_origin


def rigid_trans(q, k, im1, normed_im1, im2, normed_im2, DEEDS_fine_tune=True, with_mask=False,
                save_name=None, final_save_path=None, reverse=False):
    saveflag = False
    if not save_name == None:
        saveflag = True
    if reverse:
        querys = np.array(k)
        keys = np.array(q)
    else:
        querys = np.array(q)
        keys = np.array(k)
    querys = querys[::-1, :]  # xyz

    keys = keys[::-1, :]

    # find mean column wise
    centroid_querys = np.mean(querys, axis=1)
    centroid_keys = np.mean(keys, axis=1)

    # ensure centroids are 3x1
    centroid_querys = centroid_querys.reshape(-1, 1)
    centroid_keys = centroid_keys.reshape(-1, 1)

    # subtract mean
    querys_m = querys - centroid_querys
    keys_m = keys - centroid_keys
    querys_m = querys_m.astype(np.float32)
    keys_m = keys_m.astype(np.float32)
    H = querys_m @ np.transpose(keys_m)

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    rot_thres = 0.2
    r = ro.from_matrix(R)
    rot_est = r.as_rotvec()
    rot_est_flag = rot_est >= 0
    rot_est_binary = np.where(rot_est_flag, 1, -1)
    rot_est_larger_than_thres = np.abs(rot_est) > rot_thres
    rot_est_rect = np.where(np.abs(rot_est) <= rot_thres, rot_est, rot_thres)
    rot_est_rect_all = rot_est_rect * rot_est_binary
    rot_est_rect = np.where(rot_est_larger_than_thres, rot_est_rect_all, rot_est_rect)
    R_rect = ro.from_rotvec(rot_est_rect).as_matrix()
    t = -R_rect @ centroid_querys + centroid_keys
    m = np.zeros((4, 4), dtype=R.dtype)
    m[:3, :3] = R_rect
    m[:3, 3] = t.reshape(-1)
    m[3, :] = np.array([0, 0, 0, 1])
    print(m)
    m_xyz = np.copy(m)
    col0 = m[:, 0].copy()
    col1 = m[:, 1].copy()
    m[:, 0] = col1
    m[:, 1] = col0
    rol0 = m[0, :].copy()
    rol1 = m[1, :].copy()
    m[0, :] = rol1
    m[1, :] = rol0
    m1 = inv(m)
    # ------------------------resampled SAM rotation---------------------------------
    im1_resample_nd = normed_im1['img'][0][0, 0, :, :, :].numpy().transpose(1, 2, 0)
    im2_resample_nd = normed_im2['img'][0][0, 0, :, :, :].numpy().transpose(1, 2, 0)
    im1_resampled_shape = np.array(im1_resample_nd.shape)
    im2_resampled_shape = np.array(im2_resample_nd.shape)
    shapes = np.concatenate([im1_resampled_shape, im2_resampled_shape]).reshape(2, 3)
    covers = shapes.max(0)

    img_use = np.zeros(covers)
    if reverse:
        img_use[:] = im2_resample_nd.min()
        img_use[:im2_resampled_shape[0], :im2_resampled_shape[1], :im2_resampled_shape[2]] = im2_resample_nd
        img_use = img_use[:im1_resampled_shape[0], :im1_resampled_shape[1], :im1_resampled_shape[2]]
    else:
        img_use[:] = im1_resample_nd.min()
        img_use[:im1_resampled_shape[0], :im1_resampled_shape[1], :im1_resampled_shape[2]] = im1_resample_nd
        img_use = img_use[:im2_resampled_shape[0], :im2_resampled_shape[1], :im2_resampled_shape[2]]

    if with_mask == True:
        mask1_resample_nd = im1['processed_mask']
        mask1_resampled_shape = np.array(mask1_resample_nd.shape)
        mask_use = np.zeros(covers)
        mask_use[:mask1_resampled_shape[0], :mask1_resampled_shape[1], :mask1_resampled_shape[2]] = mask1_resample_nd
        mask2_resample_nd = im2['processed_mask']
        mask2_resampled_shape = np.array(mask2_resample_nd.shape)
        mask_use = mask_use[:mask2_resampled_shape[0], :mask2_resampled_shape[1], :mask2_resampled_shape[2]]

    rigid_nd_resampled = ndi.affine_transform(img_use, m1,
                                              cval=img_use.min())
    rigid_mask = rigid_nd_resampled > img_use.min() + 5
    rigid_mask = ndimage.binary_dilation(rigid_mask, structure=np.ones((15, 15, 1)))
    rigid_resampled = nib.Nifti1Image(rigid_nd_resampled, affine=(im2['affine_resampled']))

    fixed_resample_array = normed_im2['img'][0][0, 0, :, :, :].numpy().transpose(1, 2, 0) * rigid_mask
    fixed_resample_array[rigid_mask == 0] = fixed_resample_array.min()

    fixed_resample = nib.Nifti1Image(fixed_resample_array,
                                     affine=(im2['affine_resampled']))
    nib.save(fixed_resample, deeds_path + '/fixed_resample_sam.nii.gz')

    fixed_resample_no_crop = nib.Nifti1Image(normed_im2['img'][0][0, 0, :, :, :].numpy().transpose(1, 2, 0),
                                             affine=(im2['affine_resampled']))
    if saveflag:
        nib.save(rigid_resampled, os.path.join(final_save_path, save_name + '_0001.nii.gz'))
        nib.save(fixed_resample_no_crop, os.path.join(final_save_path, save_name + '_0000.nii.gz'))
    else:
        nib.save(rigid_resampled, deeds_path + 'affined_resample_sam.nii.gz')
        nib.save(fixed_resample_no_crop, deeds_path + 'fixed_resample_nocrop.nii.gz')

    if with_mask == True:
        rigid_mask_nd_resampled = ndi.affine_transform(mask_use, m1, order=0,
                                                       cval=0)
        rigid_mask_nd_resampled = rigid_mask_nd_resampled.astype('int32')
        rigid_mask_resampled = nib.Nifti1Image(rigid_mask_nd_resampled, affine=(im2['affine_resampled']))
        if saveflag:
            nib.save(rigid_mask_resampled, os.path.join(final_save_path, save_name + '_mask_0001.nii.gz'))
        else:
            nib.save(rigid_mask_resampled, deeds_path + 'affined_mask_resample_sam.nii.gz')

        fixed_mask_resample_array = mask2_resample_nd

        fixed_mask_resample = nib.Nifti1Image(fixed_mask_resample_array,
                                              affine=(im2['affine_resampled']))
        if saveflag:
            nib.save(fixed_mask_resample, os.path.join(final_save_path, save_name + '_mask_0000.nii.gz'))
        else:
            nib.save(fixed_mask_resample, deeds_path + 'fixed_mask_resample_sam.nii.gz')
    # -----------------------------resampled DEEDS affine fine tuning------------------------------
    if DEEDS_fine_tune:
        cmd = '%s/linearBCV -F %s/fixed_resample_nocrop.nii.gz -M %s/affined_resample_sam.nii.gz -l 3 -G 5x4x3 -L  3x2x2 -Q 2x2x2 -O %s/mri_ct_affine_resample' % (
            deeds_path, deeds_path, deeds_path, deeds_path)
        os.system(cmd)

        deeds_affine = []
        f = open(os.path.join(deeds_path, 'mri_ct_affine_resample_matrix.txt'), 'r')
        for i in range(4):
            deeds_affine.append(f.readline())
        f.close()

        all_affines_number = []
        for lines in deeds_affine:
            all_in_line = lines.split(' ', -1)
            all_in_line = [part for part in all_in_line if part.__len__() > 0]
            all_in_line.pop(-1)
            all_in_line = [float(part) for part in all_in_line]
            all_affines_number.append(all_in_line)
        all_affines_number = np.array(all_affines_number)
        all_affines_number = all_affines_number.transpose(1, 0)

        shifts = all_affines_number[:3, 3:].copy()
        ro_part = all_affines_number[:3, :3].copy()
        # yxz -> xyz
        ro_part = np.matmul(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), ro_part)
        ro_part = np.matmul(ro_part, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        all_affines_number[:3, :3] = ro_part  # xyz
        all_affines_number[0, 3] = shifts[1, 0]
        all_affines_number[1, 3] = shifts[0, 0]
        all_affines_number[2, 3] = shifts[2, 0]
        all_affines_number = np.concatenate([all_affines_number, np.array([0, 0, 0, 1]).reshape(-1, 4)], axis=0)
        all_affines_number = np.linalg.inv(all_affines_number)  # left multiply or right inverse multiply
        m_xyz_1 = np.matmul(m_xyz, all_affines_number)
        m_xyz_1_xyz = m_xyz_1.copy()

        col0 = m_xyz_1[:, 0].copy()
        col1 = m_xyz_1[:, 1].copy()
        m_xyz_1[:, 0] = col1
        m_xyz_1[:, 1] = col0
        rol0 = m_xyz_1[0, :].copy()
        rol1 = m_xyz_1[1, :].copy()
        m_xyz_1[0, :] = rol1
        m_xyz_1[1, :] = rol0
        m_xyz_2 = inv(m_xyz_1)

        rigid_nd_resampled = ndi.affine_transform(img_use, m_xyz_2,
                                                  cval=img_use.min())
        rigid_mask = rigid_nd_resampled > img_use.min() + 5
        rigid_mask = ndimage.binary_dilation(rigid_mask, structure=np.ones((15, 15, 1)))
        rigid_resampled = nib.Nifti1Image(rigid_nd_resampled, affine=(im2['affine_resampled']))
        nib.save(rigid_resampled, deeds_path + '/affined_resample_sam_deeds.nii.gz')
        fixed_resample_array = normed_im2['img'][0][0, 0, :, :, :].numpy().transpose(1, 2, 0) * rigid_mask
        fixed_resample_array[rigid_mask == 0] = fixed_resample_array.min()

        fixed_resample = nib.Nifti1Image(fixed_resample_array,
                                         affine=(im2['affine_resampled']))
        nib.save(fixed_resample, deeds_path + '/fixed_resample_sam_deeds.nii.gz')
    # ----------------------use to original image---------------------------------------------------------------------------
    im1_affine = im1['affine'].copy()
    col0 = im1_affine[:, 0].copy()
    col1 = im1_affine[:, 1].copy()
    im1_affine[:, 0] = col1
    im1_affine[:, 1] = col0

    im1_affine_resampled = im1['affine_resampled'].copy()
    col0 = im1_affine_resampled[:, 0].copy()
    col1 = im1_affine_resampled[:, 1].copy()
    im1_affine_resampled[:, 0] = col1
    im1_affine_resampled[:, 1] = col0

    im2_affine = im2['affine'].copy()
    col0 = im2_affine[:, 0].copy()
    col1 = im2_affine[:, 1].copy()
    im2_affine[:, 0] = col1
    im2_affine[:, 1] = col0

    im2_affine_resampled = im2['affine_resampled'].copy()
    col0 = im2_affine_resampled[:, 0].copy()
    col1 = im2_affine_resampled[:, 1].copy()
    im2_affine_resampled[:, 0] = col1
    im2_affine_resampled[:, 1] = col0

    L = np.matmul(im2_affine, np.linalg.inv(im2_affine_resampled))
    L[:3, 3] = 0
    B = np.matmul(im2_affine_resampled, np.linalg.inv(im2_affine))
    B[:3, 3] = 0
    if DEEDS_fine_tune:
        m_origin = np.matmul(np.linalg.inv(L), m_xyz_1_xyz)
        m_origin = np.matmul(m_origin, np.linalg.inv(B))

        col0 = m_origin[:, 0].copy()
        col1 = m_origin[:, 1].copy()
        m_origin[:, 0] = col1
        m_origin[:, 1] = col0
        rol0 = m_origin[0, :].copy()
        rol1 = m_origin[1, :].copy()
        m_origin[0, :] = rol1
        m_origin[1, :] = rol0
    else:
        m_origin = np.matmul(np.linalg.inv(L), m)  # m has transported the pivots so no change cols and raws
        m_origin = np.matmul(m_origin, np.linalg.inv(B))

    m_origin_sam = np.matmul(np.linalg.inv(L), m)
    m_origin_sam = np.matmul(m_origin_sam, np.linalg.inv(B))

    m_origin_1 = inv(m_origin)

    rigid_nd = ndi.affine_transform(im1['img'], m_origin_1, cval=im1['img'].min())
    fixed = nib.Nifti1Image(im2['img'], affine=im2['affine'])
    nib.save(fixed, deeds_path + 'fixed.nii.gz')
    rigid = nib.Nifti1Image(rigid_nd, affine=(im2['affine']))
    nib.save(rigid, deeds_path + 'affined.nii.gz')
    rigid_nd_sam = ndi.affine_transform(im1['img'], inv(m_origin_sam), cval=im1['img'].min())
    rigid_sam = nib.Nifti1Image(rigid_nd_sam, affine=(im2['affine']))
    nib.save(rigid_sam, deeds_path + 'affined_sam.nii.gz')
    # else:


if __name__ == '__main__':
    print(checkpoint_file)
    print(s_thres)
    model = init(config_file, checkpoint_file)
    root_path = '/data/sdd/user/rawdata/CT-MRI-Abd/nii/'
    print(root_path)
    all_cases = os.listdir(root_path)
    all_cases = [case.split('_', 1)[0] for case in all_cases if '_0000.nii.gz' in case]
    final_save_path = '/data/sde/user/abd_registered/abd_registered_0_'+str(s_thres)+'_double_check'
    print(final_save_path)
    if not os.path.exists(final_save_path):
        os.mkdir(final_save_path)
    s_name = 'ct.nii.gz'
    t_name = 'mri.nii.gz'
    count = 0
    for index, caseinfo in enumerate(all_cases):
        print(count, caseinfo)
        count += 1
        # if count > 1:
        #     break
        # if '1059783' not in caseinfo:
        #     continue
        source_image = os.path.join(root_path, caseinfo + '_0000.nii.gz')
        target_image = os.path.join(root_path, caseinfo + '_0001.nii.gz')
        sou_shape = tio.ScalarImage(source_image).data.shape
        tar_shape = tio.ScalarImage(target_image).data.shape

        padding_img(source_image=source_image, target_image=target_image, save_path=deeds_path, s_name=s_name,
                    t_name=t_name)
        im1, normed_im1, norm_info_1 = read_image(im1_file, is_MRI=True)  # im1 yxz normed_im1 zyx
        im2, normed_im2, norm_info_2 = read_image(im2_file, is_MRI=False, remove_bed=True)
        q, k, q_o, k_o = compute_point(model, normed_im1, norm_info_1, normed_im2, norm_info_2)
        DEEDS_fine_tune = False
        rigid_trans(q, k, im1, normed_im1, im2, normed_im2, DEEDS_fine_tune=DEEDS_fine_tune,
                    with_mask=False, save_name=None, final_save_path=None, reverse=False)

        # # ---------------------------------------------------------------
        if DEEDS_fine_tune:
            cmd = '%s/deedsBCV -F %s/fixed_resample_sam_deeds.nii.gz -M %s/affined_resample_sam_deeds.nii.gz -l 3 -G 5x4x3 -L 5x4x3 -Q 2x2x1 -O %s/MRI_SAM_resample_sam_deeds' % (
                deeds_path, deeds_path, deeds_path, deeds_path)
            os.system(cmd)
            cmd = '%s/applyBCVfloat_aniso -F %s/affined_resample_sam_deeds.nii.gz  -M %s/affined.nii.gz   -O %s/MRI_SAM_resample_sam_deeds -D %s/MRI_SAM_deform.nii.gz' % (
                deeds_path, deeds_path, deeds_path, deeds_path, deeds_path)
            os.system(cmd)
        else:
            cmd = '%s/deedsBCV -F %s/fixed_resample_sam.nii.gz -M %s/affined_resample_sam.nii.gz -l 4 -G 6x5x4x3 -L 6x5x4x3 -Q 3x2x2x1 -O %s/MRI_SAM_resample_sam' % (
                deeds_path, deeds_path, deeds_path, deeds_path)
            os.system(cmd)
            cmd = '%s/applyBCVfloat_aniso -F %s/affined_resample_sam.nii.gz  -M %s/affined_sam.nii.gz   -O %s/MRI_SAM_resample_sam -D %s/MRI_SAM_deform.nii.gz' % (
                deeds_path, deeds_path, deeds_path, deeds_path, deeds_path)
            # cmd = '%s/deedsBCV -F %s/fixed_resample_sam.nii.gz -M %s/affined_resample_sam.nii.gz -l 1 -G 3 -L 0 -Q 3 -O %s/MRI_SAM_resample_sam' % (
            #     deeds_path, deeds_path, deeds_path, deeds_path)
            # os.system(cmd)
            # cmd = '%s/applyBCVfloat_aniso -F %s/affined_resample_sam.nii.gz  -M %s/affined_sam.nii.gz   -O %s/MRI_SAM_resample_sam -D %s/MRI_SAM_deform.nii.gz' % (
            #     deeds_path, deeds_path, deeds_path, deeds_path, deeds_path)
            os.system(cmd)
        # # --------------------------------------------------------------
        ct = tio.ScalarImage(os.path.join(deeds_path, 'fixed.nii.gz'))
        ct_data = ct.data[0, :sou_shape[1], :sou_shape[2], :sou_shape[3]].numpy().transpose(2, 0, 1)
        ct_affine = ct.affine
        col1 = ct_affine[:, 1].copy()
        col0 = ct_affine[:, 0].copy()
        ct_affine[:, 1] = col0
        ct_affine[:, 0] = col1
        mri = tio.ScalarImage(os.path.join(deeds_path, 'MRI_SAM_deform.nii.gz'))
        mri_data = mri.data[0, :sou_shape[1], :sou_shape[2], :sou_shape[3]].numpy().transpose(2, 0, 1)
        mri_affine = mri.affine
        col1 = mri_affine[:, 1].copy()
        col0 = mri_affine[:, 0].copy()
        mri_affine[:, 1] = col0
        mri_affine[:, 0] = col1

        ct_nii = sitk.GetImageFromArray(ct_data)
        ct_nii.SetSpacing(ct.spacing)
        ct_nii.SetOrigin((-ct_affine[0, 3], -ct_affine[1, 3], ct_affine[2, 3]))
        ct_nii.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        sitk.WriteImage(ct_nii, os.path.join(final_save_path,
                                             caseinfo + '_0000.nii.gz'))

        mri_nii = sitk.GetImageFromArray(mri_data)
        mri_nii.SetSpacing(mri.spacing)
        mri_nii.SetOrigin((-mri_affine[0, 3], -mri_affine[1, 3], mri_affine[2, 3]))
        mri_nii.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        sitk.WriteImage(mri_nii, os.path.join(final_save_path,
                                              caseinfo + '_0001.nii.gz'))
        # # -------------------------------
