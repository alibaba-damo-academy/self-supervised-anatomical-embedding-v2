# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import csv
import os
import os.path as osp
import nibabel as nib
from sam.datasets.piplines.process3d import RemoveCTBed, CropBackground, Resample
from multiprocessing import Pool, set_start_method


root_fd = '/data/user/codes/SAM_release/data/'
dataset_name = 'NIH_lymph_node'
data_root = osp.join(root_fd, 'raw_data', dataset_name)
save_path = osp.join(root_fd, 'processed_data', dataset_name, 'nii-crop')
mask_save_path = osp.join(root_fd, 'processed_data', dataset_name, 'nii-crop-mask')
ind_save_path = osp.join(root_fd, 'processed_data', dataset_name, 'ind_files')
os.makedirs(save_path, exist_ok=True)
os.makedirs(mask_save_path, exist_ok=True)
os.makedirs(ind_save_path, exist_ok=True)

norm_spacing = 1.6


def process(name):
    print(name)
    filepath = os.path.join(data_root, name)
    img = tio.ScalarImage(filepath)
    subject = tio.Subject(image=img)
    data = {}
    data['subject'] = subject
    removebedtrans = RemoveCTBed()
    data = removebedtrans(data)
    resampletrans = Resample(norm_spacing=(norm_spacing,)*3)
    data = resampletrans(data)
    # croptrans = CropBackground(by_mask=True)  # whether crop background
    # data = croptrans(data)
    nii_img = nib.Nifti1Image(data['subject'].image.data[0, :, :, :].numpy(), affine=data['subject'].image.affine)
    nib.save(nii_img, os.path.join(save_path, name))

    mask_img = nib.Nifti1Image(data['subject']['body_mask'].data[0, :, :, :].numpy(),
                               affine=data['subject'].image.affine)
    nib.save(mask_img, os.path.join(mask_save_path, name))
    return name


def run_pool():
    set_start_method('spawn')
    all_file_name = os.listdir(data_root)
    with Pool(processes=20) as pool:
        allresults = pool.map(process, all_file_name)
    return allresults


if __name__ == '__main__':
    results = run_pool()
    with open(osp.join(ind_save_path, "lymphnode_abd_filename.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name"])
        for nii_name in results:
            if not nii_name:
                continue
            else:
                writer.writerow([nii_name])
    print('done')
