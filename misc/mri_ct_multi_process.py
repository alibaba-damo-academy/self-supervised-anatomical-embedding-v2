# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import torchio as tio
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from sam.datasets.piplines.process3d import Resample
from multiprocessing import Pool, set_start_method
import csv
from skimage import morphology
from skimage import segmentation

data_path = '/data/sde/user/abd_registered/abd_registered_1_1.8/'
img_save_path = '/data/sde/user/abd_registered/processed_1_1.8/image-regis_1/'
mask_save_path = '/data/sde/user/abd_registered/processed_1_1.8/mask-regis_1/'

if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)
if not os.path.exists(mask_save_path):
    os.mkdir(mask_save_path)


def process(name):
    print(name)
    ct_img = tio.ScalarImage(os.path.join(data_path, name + '_0000.nii.gz'))
    t1_img = tio.ScalarImage(os.path.join(data_path, name + '_0001.nii.gz'))
    # t2_img = tio.ScalarImage(os.path.join(data_path, name + '_0002.nii.gz'))

    t1_data = t1_img.data.numpy()
    # t2_data = t2_img.data.numpy()
    ct_data = ct_img.data.numpy()
    t1_data = t1_data[0, :, :, :]
    t1_data = t1_data.transpose(1, 0, 2)
    # t2_data = t2_data[0, :, :, :]
    # t2_data = t2_data.transpose(1, 0, 2)
    ct_data = ct_data[0, :, :, :]
    ct_data = ct_data.transpose(1, 0, 2)
    t1_min_max = np.percentile(t1_data, (0.2, 99.8))
    # t2_min_max = np.percentile(t2_data, (2, 99.8))
    t1_mask_data = t1_data > (t1_min_max[0] + 30)
    # t2_mask_data = t2_data > (t2_min_max[0] + 50)
    ct_mask_data = (ct_data > -300)

    assert ct_img.orientation == ('L', 'P', 'S')

    ct_mask_processed = np.copy(ct_mask_data)
    t1_mask_processed = np.copy(t1_mask_data)
    # t2_mask_processed = np.copy(t2_mask_data)

    for i in range(ct_mask_data.shape[2]):
        ct_mask_processed[:, :, i] = morphology.closing(ct_mask_data[:, :, i], morphology.disk(3))
        ct_mask_processed[:, :, i] = morphology.opening(ct_mask_processed[:, :, i], morphology.disk(3))
        ct_mask_processed[:, :, i] = morphology.remove_small_holes(
            morphology.remove_small_objects(ct_mask_processed[:, :, i], min_size=2000), area_threshold=30000)
    # ct_slic = segmentation.slic(ct_data, n_segments=1, mask=ct_mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.01)

    for i in range(t1_mask_data.shape[2]):
        t1_mask_processed[:, :, i] = morphology.closing(t1_mask_data[:, :, i], morphology.disk(3))
        t1_mask_processed[:, :, i] = morphology.opening(t1_mask_processed[:, :, i], morphology.disk(3))
        t1_mask_processed[:, :, i] = morphology.remove_small_holes(
            morphology.remove_small_objects(t1_mask_processed[:, :, i], min_size=2000), area_threshold=30000)
    # t1_slic = segmentation.slic(t1_data, n_segments=1, mask=t1_mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.05)

    # for i in range(t2_mask_data.shape[2]):
    #     t2_mask_processed[:, :, i] = morphology.closing(t2_mask_data[:, :, i], morphology.disk(3))
    #     t2_mask_processed[:, :, i] = morphology.opening(t2_mask_processed[:, :, i], morphology.disk(3))
    #     t2_mask_processed[:, :, i] = morphology.remove_small_holes(
    #         morphology.remove_small_objects(t2_mask_processed[:, :, i], min_size=2000), area_threshold=25000)
    # t2_slic = segmentation.slic(t1_data, n_segments=15, mask=t2_mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.05)

    t1_mask = tio.LabelMap(tensor=t1_mask_processed.transpose(1, 0, 2)[np.newaxis, :, :, :], affine=t1_img.affine)
    # t2_mask = tio.LabelMap(tensor=t2_slic.transpose(1, 0, 2)[np.newaxis, :, :, :], affine=t2_img.affine)
    ct_mask = tio.LabelMap(tensor=ct_mask_processed.transpose(1, 0, 2)[np.newaxis, :, :, :], affine=ct_img.affine)

    subject = tio.Subject(image=ct_img,
                          t1=t1_img,
                          # t2=t2_img,
                          ct_mask=ct_mask,
                          t1_mask=t1_mask, )
    # t2_mask=t2_mask)

    # ToCanonical = tio.ToCanonical()
    # subject = ToCanonical(subject)
    # print(subject.orientation)
    # if subject.orientation == ('R', 'A', 'S'):
    #     img_data = img_tio.data
    #     img_tio.data = torch.flip(img_data, (1, 2))
    #     img_tio.affine = np.array(
    #         [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    data = {}
    resampletrans = Resample(norm_spacing=(1.6, 1.6, 1.6), crop_artifacts=False)
    data['subject'] = subject
    data = resampletrans(data)
    subject = data['subject']

    nii_img = nib.Nifti1Image(subject.image.data[0, :, :, :].numpy(), affine=subject.image.affine)
    nib.save(nii_img, os.path.join(img_save_path, name + '_0000.nii.gz'))
    nii_img = nib.Nifti1Image(subject.t1.data[0, :, :, :].numpy(), affine=subject.t1.affine)
    nib.save(nii_img, os.path.join(img_save_path, name + '_0001.nii.gz'))
    # nii_img = nib.Nifti1Image(subject.t2.data[0, :, :, :].numpy(), affine=subject.t2.affine)
    # nib.save(nii_img, os.path.join(img_save_path, name+'_0002.nii.gz'))

    nii_img = nib.Nifti1Image(subject.ct_mask.data[0, :, :, :].numpy().astype(np.int16), affine=subject.ct_mask.affine)
    nib.save(nii_img, os.path.join(mask_save_path, name + '_0000.nii.gz'))
    nii_img = nib.Nifti1Image(subject.t1_mask.data[0, :, :, :].numpy().astype(np.int16), affine=subject.t1_mask.affine)
    nib.save(nii_img, os.path.join(mask_save_path, name + '_0001.nii.gz'))
    zs = np.where(subject.t1_mask.data[0, :, :, :] == 1)[2]
    # print(zs.max()-zs.min())
    # nii_img = nib.Nifti1Image(subject.t2_mask.data[0, :, :, :].numpy().astype(np.int16), affine=subject.t2_mask.affine)
    # nib.save(nii_img, os.path.join(mask_save_path, name + '_0002.nii.gz'))
    return name


def run_pool():
    set_start_method('spawn')
    all_elements = []
    # deeplesion_nii_path = '/data/sdb/user/Deeplesion/Images_nifti'
    # all_niis = os.listdir(deeplesion_nii_path)
    all_file_name = os.listdir(data_path)
    prefix_name = [name.split('_', 1)[0] for name in all_file_name]
    prefix_name = np.unique(prefix_name)
    with Pool(processes=20) as pool:
        allresults = pool.map(process, prefix_name)
    return allresults


if __name__ == '__main__':
    results = run_pool()
    # with open("/data/sdd/user/processed_data/ind_files/CT_MRI_abd_inter_0_filename.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["name"])
    #     for nii_name in results:
    #         if not nii_name:
    #             continue
    #         else:
    #             writer.writerow([nii_name])
