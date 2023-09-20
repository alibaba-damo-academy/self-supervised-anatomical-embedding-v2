import torchio as tio
import os
import numpy as np

ct_landmark_path = '/data/sdd/user/DEEDs/deedsBCV/ct_mask.nii.gz'
mri_landmark_path = '/data/sdd/user/DEEDs/deedsBCV/nonlinear_mri_ct_deformed_seg.nii.gz'

ct_landmark = tio.LabelMap(ct_landmark_path)
t1_landmark = tio.LabelMap(mri_landmark_path)

ct_landmark_numpy = ct_landmark.data.numpy()[0, :, :, :]
ct_landmark_numpy = ct_landmark_numpy.transpose(1, 0, 2)
all_ct_landmarks = []
for j in range(1, ct_landmark_numpy.max() + 1):
    loc = np.where(ct_landmark_numpy == j)
    loc = np.stack(loc)
    loc = loc.mean(axis=1)
    all_ct_landmarks.append(loc)
all_ct_landmarks = np.stack(all_ct_landmarks)

all_t1_landmarks = []
t1_landmark_numpy = t1_landmark.data.numpy()[0, :, :, :]
t1_landmark_numpy = t1_landmark_numpy.transpose(1, 0, 2)
all_t1_landmarks = []
for j in range(1, ct_landmark_numpy.max() + 1):
    loc = np.where(t1_landmark_numpy == j)
    loc = np.stack(loc)
    loc = loc.mean(axis=1)
    all_t1_landmarks.append(loc)
all_t1_landmarks = np.stack(all_t1_landmarks)

ct_not_nan_index = np.unique(np.where(np.isnan(all_ct_landmarks) == False)[0])
all_ct_landmarks = all_ct_landmarks[ct_not_nan_index, :]
all_t1_landmarks = all_t1_landmarks[ct_not_nan_index, :]
dis_case = []
for j in range(all_ct_landmarks.shape[0]):
    gth = all_ct_landmarks[j,:]
    pre = all_t1_landmarks[j,:]
    if np.isnan(pre).any():
        continue
    dis = np.linalg.norm(gth - pre)
    # print(str(j)+','+str(dis))
    dis_case.append(dis)
dis_here = np.array(dis_case)
# print(dis_case)
print(dis_here.mean(), dis_here.std(), dis_here.max())

print('done')