# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import os
import numpy as np
import pickle
import nibabel as nib
import torch
from multiprocessing import Pool, set_start_method
import csv

data_path = '/data/sde/user/Totalsegmentator/Totalsegmentator_dataset'
save_path = '/data/sdd/user/processed_data/totolsegmentaion/nii-resample'
mask_save_path = '/data/sdd/user/processed_data/totolsegmentaion/nii-resample-mask'

#this contains all mask name, you need to generate it
seg_labels = pickle.load(open('/data/sdd/user/processed_data/ind_files/seg_labels.pkl', 'rb'))
# all_seg_files = os.listdir(os.path.join(data_path,case,'segmentations'))
# all_seg_name = [seg_name.split('.',1)[0] for seg_name in all_seg_files]
# f = open('/data/sdd/user/processed_data/ind_files/seg_labels.pkl','wb')
# pickle.dump(all_seg_name,f)


if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(mask_save_path):
    os.mkdir(mask_save_path)


def process(case):
    print(case)
    img = tio.ScalarImage(os.path.join(data_path, case, 'ct.nii.gz'))
    data = img.data
    # assert img.orientation == ('R','A','S')
    ToCanonical = tio.ToCanonical()
    img = ToCanonical(img)
    img_data = img.data
    mask_all = torch.zeros_like(img_data)
    for label, seg_mask in enumerate(seg_labels):
        mask = tio.LabelMap(os.path.join(data_path, case, 'segmentations/', seg_mask + '.nii.gz'))
        mask = ToCanonical(mask)
        mask_data = mask.data
        mask_all[mask_data == 1] = label + 1
    img.data = torch.flip(img.data, (1, 2))
    img.affine = np.array([-img.affine[0, :], -img.affine[1, :], img.affine[2, :], img.affine[3, :]])
    mask_all = torch.flip(mask_all, (1, 2))

    nii_img = nib.Nifti1Image(img.data[0, :, :, :].numpy(), affine=img.affine)
    nib.save(nii_img, os.path.join(save_path, case + '.nii.gz'))
    nii_img = nib.Nifti1Image(mask_all[0, :, :, :].numpy(), affine=img.affine)
    nib.save(nii_img, os.path.join(mask_save_path, case + '.nii.gz'))
    return case + '.nii.gz'


def run_pool():
    set_start_method('spawn')
    all_cases = os.listdir(data_path)
    all_cases = [case for case in all_cases if '.csv' not in case]
    with Pool(processes=20) as pool:
        allresults = pool.map(process, all_cases)
    return allresults


if __name__ == '__main__':
    results = run_pool()
    with open("/data/sdd/user/processed_data/ind_files/totolseg_filename.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name"])
        for nii_name in results:
            if not nii_name:
                continue
            else:
                writer.writerow([nii_name])
