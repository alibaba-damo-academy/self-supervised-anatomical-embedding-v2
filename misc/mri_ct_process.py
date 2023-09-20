# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import torchio as tio
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from skimage import morphology
from skimage import segmentation


data_path = '/data/sde/user/abd_registered/abd_registered_0/'
img_save_path = '/data/sde/user/abd_registered/processed_0/image-regis_0/'
mask_save_path = '/data/sde/user/abd_registered/processed_0/mask-regis_0/'

if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)
if not os.path.exists(mask_save_path):
    os.mkdir(mask_save_path)

all_file_name = os.listdir(data_path)

prefix_name = [name.split('_', 1)[0] for name in all_file_name]
prefix_name = np.unique(prefix_name)

for name in prefix_name:
    if '970107' not in name:
        continue
    print(name)
    ct_img = tio.ScalarImage(os.path.join(data_path, name + '_0000.nii.gz'))
    t1_img = tio.ScalarImage(os.path.join(data_path, name + '_0001.nii.gz'))
    # t2_img = tio.ScalarImage(os.path.join(data_path, name + '_0002.nii.gz'))

    t1_data = t1_img.data.numpy()
    t1_data = t1_data[0, :, :, :]
    t1_data = t1_data.transpose(1, 0, 2)
    # t2_data = t2_img.data.numpy()
    ct_data = ct_img.data.numpy()
    ct_data = ct_data[0, :, :, :]
    ct_data = ct_data.transpose(1, 0, 2)
    assert ct_img.orientation == ('L', 'P', 'S')
    t1_min_max = np.percentile(t1_data, (2, 99.8))
    # t2_min_max = np.percentile(t2_data, (2, 99.8))
    t1_mask_data = t1_data > (t1_min_max[0] + 50)
    # t2_mask_data = t2_data > (t2_min_max[0] + 10)
    ct_mask_data = (ct_data > -300)
    ct_mask_processed = np.copy(ct_mask_data)
    t1_mask_processed = np.copy(t1_mask_data)

    for i in range(ct_mask_data.shape[2]):
        ct_mask_processed[:, :, i] = morphology.closing(ct_mask_data[:, :, i], morphology.disk(3))
        ct_mask_processed[:, :, i] = morphology.opening(ct_mask_processed[:, :, i], morphology.disk(3))
        ct_mask_processed[:, :, i] = morphology.remove_small_holes(
            morphology.remove_small_objects(ct_mask_processed[:, :, i], min_size=2000), area_threshold=25000)
    # ct_slic = segmentation.slic(ct_data, n_segments=15, mask=ct_mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.01)

    for i in range(t1_mask_data.shape[2]):
        t1_mask_processed[:, :, i] = morphology.closing(t1_mask_data[:, :, i], morphology.disk(3))
        t1_mask_processed[:, :, i] = morphology.opening(t1_mask_processed[:, :, i], morphology.disk(3))
        t1_mask_processed[:, :, i] = morphology.remove_small_holes(
            morphology.remove_small_objects(t1_mask_processed[:, :, i], min_size=2000), area_threshold=25000)
    # t1_slic = segmentation.slic(t1_data, n_segments=15, mask=t1_mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.05)

    t1_mask = tio.LabelMap(tensor=t1_mask_processed.transpose(1, 0, 2)[np.newaxis, :, :, :], affine=t1_img.affine)
    # t2_mask = tio.LabelMap(tensor=t2_mask_data, affine=t2_img.affine)
    ct_mask = tio.LabelMap(tensor=ct_mask_processed.transpose(1, 0, 2)[np.newaxis, :, :, :], affine=ct_img.affine)

    subject = tio.Subject(ct=ct_img,
                          t1=t1_img,
                          ct_mask=ct_mask,
                          t1_mask=t1_mask, )

    # ToCanonical = tio.ToCanonical()
    # subject = ToCanonical(subject)
    resampletrans = tio.Resample((1.6, 1.6, 1.6))
    subject = resampletrans(subject)

    nii_img = nib.Nifti1Image(subject.ct.data[0, :, :, :].numpy(), affine=subject.ct.affine)
    nib.save(nii_img, os.path.join(img_save_path, name + '_0000.nii.gz'))
    nii_img = nib.Nifti1Image(subject.t1.data[0, :, :, :].numpy(), affine=subject.t1.affine)
    nib.save(nii_img, os.path.join(img_save_path, name + '_0001.nii.gz'))
    # nii_img = nib.Nifti1Image(subject.t2.data[0, :, :, :].numpy(), affine=subject.t2.affine)
    # nib.save(nii_img, os.path.join(img_save_path, name+'_0002.nii.gz'))

    nii_img = nib.Nifti1Image(subject.ct_mask.data[0, :, :, :].numpy().astype(np.int16), affine=subject.ct_mask.affine)
    nib.save(nii_img, os.path.join(mask_save_path, name + '_0000.nii.gz'))
    nii_img = nib.Nifti1Image(subject.t1_mask.data[0, :, :, :].numpy().astype(np.int16), affine=subject.t1_mask.affine)
    nib.save(nii_img, os.path.join(mask_save_path, name + '_0001.nii.gz'))
    # nii_img = nib.Nifti1Image(subject.t2_mask.data[0, :, :, :].numpy(), affine=subject.t2_mask.affine)
    # nib.save(nii_img, os.path.join(mask_save_path, name + '_0002.nii.gz'))

    print('done')

print('done')
