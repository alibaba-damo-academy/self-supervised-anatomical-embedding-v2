# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import torchio as tio
import torch
import pickle as pkl

# test_path = '/data/sdd/user/rawdata/CT-MRI-Abd/nii_regis_final_test'
test_path = '/data/sde/user/CT-MRI-ABD-resample_padding-test/'
all_cases = os.listdir(test_path)

all_cases = [name.split('_', 1)[0] for name in all_cases if '_c2f_affine_mask.nii.gz' in name]
dis_all = []
dis_x = []
dis_y = []
dis_z = []
all = []
for case in all_cases:
    print(case)
    dis_case = []
    mr_mask = tio.LabelMap(os.path.join(test_path, case + '_0001_mask.nii.gz'))
    ct_mask = tio.LabelMap(os.path.join(test_path, case + '_0000_mask.nii.gz'))
    mr_data = mr_mask.data[0, :, :, :]
    ct_data = ct_mask.data[0, :, :, :]
    # ct_data = torch.round(ct_data)
    ct_labels = torch.unique(ct_data).tolist()
    # mr_data = torch.round(mr_data)
    # mr_labels = torch.unique(mr_data).tolist()
    labels = torch.unique(mr_data).tolist()
    print(labels, ct_labels)
    labels = [label for label in labels if label > 0]
    for label in labels:
        mr_loc = torch.where(mr_data == label)
        mr_loc = torch.stack(mr_loc)
        mr_loc = torch.mean(mr_loc.type(torch.float32), dim=1)
        ct_loc = torch.where(ct_data == label)
        ct_loc = torch.stack(ct_loc)
        if ct_loc.shape[1] == 0:
            continue
        ct_loc = torch.mean(ct_loc.type(torch.float32), dim=1)
        all.append([label, mr_loc, ct_loc])
        dis_all.append(torch.linalg.norm(mr_loc - ct_loc))
        dis_case.append(torch.linalg.norm(mr_loc - ct_loc))
        dis_x.append(torch.linalg.norm(mr_loc[0] - ct_loc[0]))
        dis_y.append(torch.linalg.norm(mr_loc[1] - ct_loc[1]))
        dis_z.append(torch.linalg.norm(mr_loc[2] - ct_loc[2]))
f = open('/data/sde/user/init_abd_results.pkl', 'wb')
pkl.dump(all, f)
dis_all = torch.tensor(dis_all)
dis_x = torch.tensor(dis_x)
dis_y = torch.tensor(dis_y)
dis_z = torch.tensor(dis_z)
print(dis_all.mean(), dis_all.std())
print(dis_x.mean(), dis_x.std())
print(dis_y.mean(), dis_y.std())
print(dis_z.mean(), dis_z.std())
