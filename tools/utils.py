# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import numpy as np
from sam.datasets.piplines import Resample, RescaleIntensity, GenerateMetaInfo, Collect3d, CropBackground, RemoveCTBed, \
    Crop, DynamicRescaleIntensity
from sam.datasets.collect import collate
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import torch

import SimpleITK as sitk


def read_image(path, mask_path=None, norm_spacing=(2., 2., 2.), crop_loc=None, is_MRI=False, to_canonical=True,
               remove_bed=False):
    data = {}
    origin_info = {}
    image_fn = path.split('/', -1)[-1]
    data_path = path.replace(image_fn, '')

    img_tio = tio.ScalarImage(data_path + image_fn)
    if to_canonical:
        ToCanonical = tio.ToCanonical()
        img_tio = ToCanonical(img_tio)
    # print(img_tio.spacing)
    if img_tio.orientation == ('R', 'A', 'S'):
        img_data = img_tio.data
        img_tio.data = torch.flip(img_data, (1, 2))
        img_tio.affine = np.array(
            [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    assert img_tio.orientation == ('L', 'P', 'S'), print('right now the image orientation need to be LPS+ ')
    img_data = img_tio.data
    img_tio.data = img_data.permute(0, 2, 1, 3)
    img_tio.affine = np.array(
        [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])

    if remove_bed:
        remove_bed = RemoveCTBed(thres=-800)
        data = {}
        data['subject'] = tio.Subject(image=img_tio)
        data = remove_bed(data)
        body_mask = data['subject']['body_mask'].data
        img_tio.data[body_mask == 0] = img_tio.data.min()
        data = {}
        print('remove_bed')
    if not mask_path is None:
        # mask_info = mask_path.split('/', -1)[-1]
        mask_tio = tio.LabelMap(mask_path)
        if to_canonical:
            mask_tio = ToCanonical(mask_tio)
        if mask_tio.orientation == ('R', 'A', 'S'):
            mask_data = mask_tio.data
            mask_tio.data = torch.flip(mask_data, (1, 2))
            mask_tio.affine = np.array(
                [-mask_tio.affine[0, :], -mask_tio.affine[1, :], mask_tio.affine[2, :], mask_tio.affine[3, :]])
        assert mask_tio.orientation == ('L', 'P', 'S'), print('right now the image orientation need to be LPS+ ')
        mask_data = mask_tio.data
        # mask_data[mask_data>1] = 1
        mask_tio.data = mask_data.permute(0, 2, 1, 3)
        mask_tio.affine = np.array(
            [mask_tio.affine[1, :], mask_tio.affine[0, :], mask_tio.affine[2, :], mask_tio.affine[3, :]])
        mask_origin_data = mask_tio.data.numpy().copy()

    origin_data = img_tio.data.numpy().copy()
    img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
    img_tio_spacing = img_tio.spacing
    img_tio_affine = img_tio.affine
    norm_ratio = np.array(img_tio_spacing) / np.array(norm_spacing)
    if not mask_path is None:
        subject = tio.Subject(
            image=img_tio,
            mask=mask_tio
        )
    else:
        subject = tio.Subject(
            image=img_tio
        )
    data['image_fn'] = image_fn
    data['subject'] = subject
    if not crop_loc is None:
        crop_filter = Crop(by_view=False)
        data['crop_matrix_origin'] = crop_loc
        data = crop_filter(data)

    resample_filter = Resample(norm_spacing=norm_spacing, crop_artifacts=False)
    data = resample_filter(data)
    affine_resampled = data['subject'].image.affine
    if is_MRI:
        rescale_filter = DynamicRescaleIntensity()
    else:
        rescale_filter = RescaleIntensity()
    data = rescale_filter(data)
    meta_collect_filter = GenerateMetaInfo(is_train=False)
    data = meta_collect_filter(data)
    data['is_mri'] = is_MRI
    collects = Collect3d(keys=['img'], meta_keys=['filename', 'is_mri'])
    input = collects(data)
    batch = collate([input])
    # batch['img_metas'] =  batch['img_metas'].data
    origin_info['img'] = origin_data[0, :, :, :]  # yxz
    origin_info['shape'] = img_tio_shape
    origin_info['spacing'] = img_tio_spacing
    origin_info['affine'] = img_tio_affine
    origin_info['affine_resampled'] = affine_resampled
    if not mask_path is None:
        origin_info['processed_mask'] = data['subject'].mask.data.numpy()[0, :, :, :]
        origin_info['origin_mask'] = mask_origin_data[0, :, :, :]
    return origin_info, batch, norm_ratio


def read_image_regis(path):
    data = {}
    origin_info = {}
    image_fn = path.split('/', -1)[-1]
    data_path = path.replace(image_fn, '')

    img_tio = tio.ScalarImage(data_path + image_fn)
    if img_tio.orientation == ('R', 'A', 'S'):
        img_data = img_tio.data
        img_tio.data = torch.flip(img_data, (1, 2))
        img_tio.affine = np.array(
            [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    assert img_tio.orientation == ('L', 'P', 'S'), print('right now the image orientation need to be LPS+ ')
    img_data = img_tio.data
    img_tio.data = img_data.permute(0, 2, 1, 3)
    img_tio.affine = np.array(
        [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    # origin_data = img_tio.data.numpy().copy()

    subject = tio.Subject(
        image=img_tio
    )
    data['image_fn'] = image_fn
    data['subject'] = subject
    remove_bed = RemoveCTBed(thres=-300)
    data = remove_bed(data)
    # mask = data['body_mask'].astype('int16')
    # img = data['subject'].image.data[0,:,:,:].numpy()
    # mask_sitk = sitk.GetImageFromArray(mask)
    # sitk.WriteImage(mask_sitk,'mask.nii.gz')
    # img_sitk = sitk.GetImageFromArray(img)
    # sitk.WriteImage(img_sitk,'img.nii.gz')

    crop = CropBackground(thres=1, by_mask=True)
    data = crop(data)
    img_tio_shape = data[
        'subject'].image.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
    img_tio_spacing = data['subject'].image.spacing
    img_tio_affine = data['subject'].image.affine
    origin_data = data['subject'].image.data.numpy().copy()
    origin_mask = data['subject']['body_mask'].data[0, :, :, :].numpy().copy()
    norm_ratio = np.array(img_tio_spacing) / np.array([2., 2., 2.])
    resample = Resample()
    data = resample(data)
    rescale = RescaleIntensity()
    data = rescale(data)
    body_mask = data['subject']['body_mask'].data[0, :, :, :]
    meta_collect = GenerateMetaInfo()
    data = meta_collect(data)
    collects = Collect3d(keys=['img'])
    input = collects(data)
    batch = collate([input])
    origin_info['img'] = origin_data[0, :, :, :]  # yxz
    origin_info['shape'] = img_tio_shape
    origin_info['spacing'] = img_tio_spacing
    origin_info['affine'] = img_tio_affine
    origin_info['body_mask'] = body_mask  # this is the processed data body mask not for original image
    origin_info['origin_mask'] = origin_mask
    return origin_info, batch, norm_ratio


def visualize(im1, im2, norm_ratio_1, norm_ratio_2, pt1, pt2, score,
              lung_window=False, save_folder='', savename=None, im1_isMRI=False, im2_isMRI=False):
    print('visualizing ...')
    if lung_window:
        q_window_low, q_window_high = -1350, 150
        k_window_low, k_window_high = -1350, 150
    else:
        q_window_low, q_window_high = -100, 200
        k_window_low, k_window_high = -100, 200
    if im1_isMRI:
        q_window_low, q_window_high = im1.min(), im1.max()
    if im2_isMRI:
        k_window_low, k_window_high = im2.min(), im2.max()
    markersize = 26

    fig, ax = plt.subplots(3, 2, figsize=(20, 30))
    q_img = im1.transpose(2, 0, 1)
    q_img[q_img < q_window_low] = q_window_low
    q_img[q_img > q_window_high] = q_window_high
    q_img = (q_img - q_window_low) / (q_window_high - q_window_low)

    q_img = q_img.astype(np.float32)

    slice = q_img[pt1[2], :, :]
    ax[0, 0].set_title('query')
    ax[0, 0].imshow(slice, cmap='gray')
    ax[0, 0].plot((pt1[0]), (pt1[1]), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    slice = q_img[:, pt1[1], :]
    slice = slice[::-1, :]
    ax[1, 0].set_title('query')
    ax[1, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[0])
    ax[1, 0].plot((pt1[0]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)

    slice = q_img[:, :, pt1[0]]
    slice = slice[::-1, :]
    ax[2, 0].set_title('query')
    ax[2, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[1])
    ax[2, 0].plot((pt1[1]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    k_img = im2.transpose(2, 0, 1)

    k_img[k_img < k_window_low] = k_window_low
    k_img[k_img > k_window_high] = k_window_high
    k_img = (k_img - k_window_low) / (k_window_high - k_window_low)
    k_img = k_img.astype(np.float32)

    slice = k_img[pt2[2], :, :]
    ax[0, 1].set_title('key')
    ax[0, 1].imshow(slice, cmap='gray')
    ax[0, 1].plot((pt2[0]), (pt2[1]), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)

    slice = k_img[:, pt2[1], :]
    slice = slice[::-1, :]

    ax[1, 1].set_title('key')
    ax[1, 1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[0])
    ax[1, 1].plot((pt2[0]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    slice = k_img[:, :, pt2[0]]
    slice = slice[::-1, :]
    ax[2, 1].set_title('key')
    ax[2, 1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[1])
    ax[2, 1].plot((pt2[1]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    plt.suptitle(f'score:{score}')
    plt.tight_layout()
    if not savename is None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder,
            f'{savename}_{pt1[0]}_{pt1[1]}_{pt1[2]}_{pt2[0]}_{pt2[1]}_{pt2[2]}.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def get_contour_points(mask):
    # mask = ndimage.binary_erosion(mask, structure=np.array(
    #     [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]))
    # mask_erosion = ndimage.binary_erosion(mask, structure=np.array(
    #     [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]))

    mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    mask_erosion = ndimage.binary_erosion(mask, structure=np.array(
        [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]))

    mask_contour = mask.astype(int) - mask_erosion.astype(int)
    all_points = np.where(mask_contour == 1)
    all_points = np.array(all_points)
    z_range = [all_points[2, :].min(), all_points[2, :].max()]
    all_points = all_points[:, all_points[2, :] < z_range[1]]
    all_points = all_points[:, all_points[2, :] > z_range[0]]
    x = all_points[1, :].copy()
    y = all_points[0, :].copy()
    all_points[0, :] = x
    all_points[1, :] = y
    return all_points


def visualize_seg(img, points, im_fn):
    window_high = 200
    window_low = -200
    if not os.path.exists('/data/sdb/user/mmdet/liver_test/' + str(im_fn) + '/'):
        os.mkdir('/data/sdb/user/mmdet/liver_test/' + str(im_fn) + '/')

    q_img = img.transpose(2, 0, 1)
    q_img[q_img < window_low] = window_low
    q_img[q_img > window_high] = window_high
    q_img = (q_img - window_low) / (window_high - window_low)
    q_img = q_img.astype(np.float32)
    zs = np.unique(points[:, 2])
    for z in zs:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(q_img[z, :, :], cmap='gray')
        slice_points = points[points[:, 2] == z, :]
        for i in range(slice_points.shape[0]):
            ax.plot((slice_points[i, 0]), slice_points[i, 1], 'o', markerfacecolor='none',
                    markeredgecolor="red",
                    markersize=12, markeredgewidth=2)
        plt.savefig('/data/sdb/user/mmdet/liver_test/' + str(im_fn) + '/' + str(z) + '.png')
        # plt.show()
        plt.close()


def make_local_grid_patch(pt, margin_x, margin_y, margin_z, imshape_x, imshape_y, imshape_z):
    pt = np.asarray(pt).astype('int')
    pt_patch = np.zeros((2 * margin_z + 1, 2 * margin_y + 1, 2 * margin_x + 1, 3), dtype='int')
    for j in range(-margin_z, margin_z + 1):
        for k in range(-margin_y, margin_y + 1):
            for w in range(-margin_x, margin_x + 1):
                pt_patch[j + margin_z, k + margin_y, w + margin_x, :] = pt + np.array(
                    [w, k, j])
    pt_patch = pt_patch.reshape(-1, 3)

    pt_patch[:, 0] = np.where(pt_patch[:, 0] < imshape_x, pt_patch[:, 0], imshape_x - 1)
    pt_patch[:, 1] = np.where(pt_patch[:, 1] < imshape_y, pt_patch[:, 1], imshape_y - 1)
    pt_patch[:, 2] = np.where(pt_patch[:, 2] < imshape_z, pt_patch[:, 2], imshape_z - 1)
    pt_patch[:, 0] = np.where(pt_patch[:, 0] > 0, pt_patch[:, 0], 0)
    pt_patch[:, 1] = np.where(pt_patch[:, 1] > 0, pt_patch[:, 1], 0)
    pt_patch[:, 2] = np.where(pt_patch[:, 2] > 0, pt_patch[:, 2], 0)
    pt_patch = np.unique(pt_patch, axis=0)
    return pt_patch


def IOX(box1, gts, dim='3D', metric='IOU'):
    """
    Compute the IOU, IOBB, IOGT, or IOBBGT between a box and a list of boxes. We assume the box is a predicted bounding-box and
    the box list is a list of ground-truths of the same image. IOU is intersection over union; IOBB is intersection over
    predicted bounding-box; IOGT is intersection over ground-truths; IOBBGT is max(IOBB, IOGT)
    :param box1: A numpy array of size d. d = 4 for 2D box (x1, y1, x2, y2), and 6 for 3D box (x1, y1, z1, x2, y2, z2).
    :param gts: A numpy array of size n x d. n is the number of gt boxes.
    :param dim: One of the 3 strings: 2D, 3D, P3D. P3D means detections are in 3D but gts are in 2D. If a gt is on one
     slice of the box, we compute the 2D overlap to decide if the box hits the gt. Note that gt is only annotated on one
     slice, but its format should still in 3D (x1, y1, z, x2, y2, z).
    :param metric: One of the 4 strings: IOU, IOBB, IOGT, IOBBGT.
    :return: A numpy array with size n.
    """
    if dim == '3D':
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        izmin = np.maximum(gts[:, 2], box1[2])
        ixmax = np.minimum(gts[:, 3], box1[3])
        iymax = np.minimum(gts[:, 4], box1[4])
        izmax = np.minimum(gts[:, 5], box1[5])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        id = np.maximum(izmax - izmin + 1., 0.)
        inters = iw * ih * id
        union = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.) +
                 (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.) - inters)
        box_size = (box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.)
        gt_sizes = (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.)

    elif dim == 'P3D':
        assert np.all(gts[:, 2] == gts[:, 5])
        z_inters = (box1[2] <= gts[:, 2]) & (gts[:, 2] <= box1[5])
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        ixmax = np.minimum(gts[:, 3], box1[3])
        iymax = np.minimum(gts[:, 4], box1[4])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih * z_inters
        union = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) +
                 (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) - inters)
        box_size = (box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.)
        gt_sizes = (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.)

    elif dim == '2D':
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        ixmax = np.minimum(gts[:, 2], box1[2])
        iymax = np.minimum(gts[:, 3], box1[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        union = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
                 (gts[:, 2] - gts[:, 0] + 1.) * (gts[:, 3] - gts[:, 1] + 1.) - inters)
        box_size = (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)
        gt_sizes = (gts[:, 2] - gts[:, 0] + 1.) * (gts[:, 3] - gts[:, 1] + 1.)

    if metric == 'IOU':
        return inters / union
    elif metric == 'IOBB':
        return inters / box_size
    elif metric == 'IOGT':
        return inters / gt_sizes
    elif metric == 'IOBBGT':
        return np.maximum(inters / box_size, inters / gt_sizes)


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability， alpha is the bin_score or Z in paper"""
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
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


if __name__ == '__main__':
    data_path = '/data/sdb/user/Deeplesion/Images_nifti/'
    image_fn = '004450_01_02_089-149.nii.gz'

    _, batch, _ = read_image(data_path + image_fn)
