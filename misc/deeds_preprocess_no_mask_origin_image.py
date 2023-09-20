# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import nibabel as nib
import os
import torch
import numpy as np

target_image = '/data/sdd/user/testt'

source_image = '/data/sdd/user/tests'

save_path = '/data/sdd/user/DEEDs/deedsBCV'
s_name = 'ct.nii.gz'
s_mask_name = 'ct_mask.nii.gz'
t_name = 'mri.nii.gz'
t_mask_name = 'mri_mask.nii.gz'


def padding_img(source_image=source_image, target_image=target_image, save_path=save_path, s_name=s_name,
                t_name=t_name, source_mask=None, target_mask=None):
    s_img = tio.ScalarImage(source_image)
    t_img = tio.ScalarImage(target_image)
    if not source_mask is None:
        s_mask = tio.LabelMap(source_mask)
        t_mask = tio.LabelMap(target_mask)

    # s_mask_data = s_mask.data
    # s_img.data[s_mask_data>4] = -1024
    # s_img.data[s_mask_data<4] = -1024
    #
    # t_mask_data = t_mask.data
    # t_img.data[t_mask_data>4] = -1024
    # t_img.data[t_mask_data<4] = -1024

    ToCanonical = tio.ToCanonical()  # this will cause bias shift, please check it
    s_img = ToCanonical(s_img)
    s_img.data = torch.flip(s_img.data, (1, 2))
    s_img.affine = np.array([-s_img.affine[0, :], -s_img.affine[1, :], s_img.affine[2, :], s_img.affine[3, :]])
    t_img = ToCanonical(t_img)
    t_img.data = torch.flip(t_img.data, (1, 2))
    t_img.affine = np.array([-t_img.affine[0, :], -t_img.affine[1, :], t_img.affine[2, :], t_img.affine[3, :]])
    if not source_mask is None:
        s_mask = ToCanonical(s_mask)
        s_mask.data = torch.flip(s_mask.data, (1, 2))
        s_mask.affine = np.array([-s_mask.affine[0, :], -s_mask.affine[1, :], s_mask.affine[2, :], s_mask.affine[3, :]])
        t_mask = ToCanonical(t_mask)
        t_mask.data = torch.flip(t_mask.data, (1, 2))
        t_mask.affine = np.array([-t_mask.affine[0, :], -t_mask.affine[1, :], t_mask.affine[2, :], t_mask.affine[3, :]])

    resampletrans = tio.Resample(s_img.spacing)
    # resampletrans = tio.Resample((2.,2.,4.))
    # s_img = resampletrans(s_img)
    # s_img_resample.data[s_img_resample.data > 3000] = 3000
    t_img_resample = resampletrans(t_img)
    if not source_mask is None:
        t_mask_resample = resampletrans(t_mask)

    # organ_ind = torch.where(s_img_resample.data>0)
    # organ_ind = torch.stack(organ_ind[1:])

    # organ_ind.min(dim=1)

    s_img_shape = np.array(s_img.shape[1:])
    t_img_shape = np.array(t_img_resample.shape[1:])

    all_shape = np.stack([s_img_shape, t_img_shape])
    max_shape = all_shape.max(axis=0)

    max_shape_normed = np.ceil(np.array(s_img.spacing) / 2. * max_shape)
    max_shape_normed = max_shape_normed + (3 - max_shape_normed % 3)
    max_shape_padding = np.round(max_shape_normed / np.array(s_img.spacing) * 2.)
    max_shape = max_shape_padding.astype(np.int)

    s_img_paddings = max_shape - s_img_shape
    s_img_paddings_half = np.floor(s_img_paddings / 2).astype('int')

    t_img_paddings = max_shape - t_img_shape
    t_img_paddings_half = np.floor(t_img_paddings / 2).astype('int')

    s_padded_img = np.zeros(max_shape, dtype=s_img.data[0, :, :, :].numpy().dtype)
    t_padded_img = np.zeros(max_shape, dtype=t_img_resample.data[0, :, :, :].numpy().dtype)

    s_padded_img[:] = -1024
    t_padded_img[:] = t_img_resample.data.numpy().min()

    if not source_mask is None:
        s_padding_mask = np.zeros(max_shape, dtype=s_mask.data[0, :, :, :].numpy().dtype)
        t_padding_mask = np.zeros(max_shape, dtype=t_mask_resample.data[0, :, :, :].numpy().dtype)

    # s_padded_img[s_img_paddings_half[0]:s_img_paddings_half[0] + s_img_shape[0],
    # s_img_paddings_half[1]:s_img_paddings_half[1] + s_img_shape[1],
    # s_img_paddings_half[2]:s_img_paddings_half[2] + s_img_shape[2]] = s_img_resample.data[0, :, :, :].numpy()
    # s_padded_mask[s_mask_paddings_half[0]:s_mask_paddings_half[0] + s_mask_shape[0],
    # s_mask_paddings_half[1]:s_mask_paddings_half[1] + s_mask_shape[1],
    # s_mask_paddings_half[2]:s_mask_paddings_half[2] + s_mask_shape[2]] = s_mask_resample.data[0, :, :, :].numpy()
    #
    # t_padded_img[t_img_paddings_half[0]:t_img_paddings_half[0] + t_img_shape[0],
    # t_img_paddings_half[1]:t_img_paddings_half[1] + t_img_shape[1],
    # t_img_paddings_half[2]:t_img_paddings_half[2] + t_img_shape[2]] = t_img_resample.data[0, :, :, :].numpy()
    # t_padded_mask[t_mask_paddings_half[0]:t_mask_paddings_half[0] + t_mask_shape[0],
    # t_mask_paddings_half[1]:t_mask_paddings_half[1] + t_mask_shape[1],
    # t_mask_paddings_half[2]:t_mask_paddings_half[2] + t_mask_shape[2]] = t_mask_resample.data[0, :, :, :].numpy()

    s_padded_img[:s_img_shape[0],
    : s_img_shape[1],
    : s_img_shape[2]] = s_img.data[0, :, :, :].numpy()

    t_padded_img[: t_img_shape[0],
    : t_img_shape[1],
    :t_img_shape[2]] = t_img_resample.data[0, :, :, :].numpy()

    tmp1 = s_img.affine[0, 3].copy()
    tmp2 = s_img.affine[1, 3].copy()
    tmp3 = s_img.affine[2, 3].copy()
    s_img.affine[0, 3] = tmp2
    s_img.affine[1, 3] = tmp1
    nii_img = nib.Nifti1Image(s_padded_img, affine=s_img.affine)
    nib.save(nii_img, os.path.join(save_path, s_name))

    tmp1 = t_img_resample.affine[0, 3].copy()
    tmp2 = t_img_resample.affine[1, 3].copy()
    t_img_resample.affine[0, 3] = tmp2
    t_img_resample.affine[1, 3] = tmp1
    nii_img = nib.Nifti1Image(t_padded_img, affine=t_img_resample.affine)
    nib.save(nii_img, os.path.join(save_path, t_name))

    if not source_mask is None:
        s_padding_mask[:s_img_shape[0],
        : s_img_shape[1],
        : s_img_shape[2]] = s_mask.data[0, :, :, :].numpy()

        t_padding_mask[: t_img_shape[0],
        : t_img_shape[1],
        :t_img_shape[2]] = t_mask_resample.data[0, :, :, :].numpy()

        tmp1 = s_mask.affine[0, 3].copy()
        tmp2 = s_mask.affine[1, 3].copy()
        tmp3 = s_img.affine[2, 3].copy()
        s_mask.affine[0, 3] = tmp2
        s_mask.affine[1, 3] = tmp1
        nii_img = nib.Nifti1Image(s_padding_mask, affine=s_mask.affine)
        nib.save(nii_img, os.path.join(save_path, s_mask_name))

        tmp1 = t_mask_resample.affine[0, 3].copy()
        tmp2 = t_mask_resample.affine[1, 3].copy()
        t_mask_resample.affine[0, 3] = tmp2
        t_mask_resample.affine[1, 3] = tmp1
        nii_img = nib.Nifti1Image(t_padding_mask, affine=t_mask_resample.affine)
        nib.save(nii_img, os.path.join(save_path, t_mask_name))


if __name__ == '__main__':
    padding_img(source_image=source_image, target_image=target_image, save_path=save_path, s_name=s_name,
                t_name=t_name)
