# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torchio as tio
import csv
import os
import nibabel as nib

data_root = '/data/sdd/user/rawdata/lymphnode/Mediastinum/images_nii/'

save_path = '/data/sdd/user/processed_data/lymphnode/nii/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

all_files = os.listdir(data_root)
print('test')

with open("/data/sdd/user/processed_data/lymphnode_m_filename.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name"])
    i = 0
    for name in all_files:
        filepath = os.path.join(data_root, name)
        img = tio.ScalarImage(filepath)
        resampletrans = tio.Resample((1.6, 1.6, 1.6))
        img_resample = resampletrans(img)

        nii_img = nib.Nifti1Image(img_resample.data[0, :, :, :].numpy(), affine=img_resample.affine)
        nib.save(nii_img, os.path.join(save_path, name))
        writer.writerow([name])
        print(i)
        i = i + 1
