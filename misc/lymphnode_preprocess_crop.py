# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import csv
import os
import nibabel as nib
from sam.datasets.piplines.process3d import RemoveCTBed,CropBackground,Resample

data_root = '/data/sdd/user/rawdata/lymphnode/Abdomen/images_nii/'

save_path = '/data/sdd/user/processed_data/lymphnode/nii-crop/'
mask_save_path = '/data/sdd/user/processed_data/lymphnode/nii-crop-mask/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(mask_save_path):
    os.mkdir(mask_save_path)

all_files = os.listdir(data_root)
print('test')


for name in all_files:
    print(name)
    filepath = os.path.join(data_root, name)
    img = tio.ScalarImage(filepath)
    subject = tio.Subject(image = img)
    data = {}
    data['subject'] = subject
    removebedtrans = RemoveCTBed()
    data = removebedtrans(data)
    resampletrans = Resample(norm_spacing=(1.6, 1.6, 1.6))
    data = resampletrans(data)
    croptrans = CropBackground(by_mask=True)
    data = croptrans(data)

    nii_img = nib.Nifti1Image(data['subject'].image.data[0,:,:,:].numpy(), affine=data['subject'].image.affine)
    nib.save(nii_img, os.path.join(save_path, name))

    mask_img = nib.Nifti1Image(data['subject']['body_mask'].data[0,:,:,:].numpy(), affine=data['subject'].image.affine)
    nib.save(mask_img, os.path.join(mask_save_path, name))

