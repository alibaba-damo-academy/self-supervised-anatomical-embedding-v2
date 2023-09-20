# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
import torchio as tio
import nibabel as nib
import os
import torch


class RandHistogramShift():
    """
    Apply random nonlinear transform to the image's intensity histogram.
    Args:
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
    """

    def __init__(self, num_control_points=(5, 15), prob: float = 1) -> None:

        if isinstance(num_control_points, int):
            if num_control_points <= 2:
                raise ValueError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (num_control_points, num_control_points)
        else:
            if len(num_control_points) != 2:
                raise ValueError("num_control points should be a number or a pair of numbers")
            if min(num_control_points) <= 2:
                raise ValueError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (min(num_control_points), max(num_control_points))
        self.prob = prob

    def interp(self, x, xp, fp):
        ns = torch if isinstance(x, torch.Tensor) else np
        if isinstance(x, np.ndarray):
            # approx 2x faster than code below for ndarray
            return np.interp(x, xp, fp)

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indices = ns.searchsorted(xp.reshape(-1), x.reshape(-1)) - 1
        indices = ns.clip(indices, 0, len(m) - 1)

        f = (m[indices] * x.reshape(-1) + b[indices]).reshape(x.shape)
        f[x < xp[0]] = fp[0]
        f[x > xp[-1]] = fp[-1]
        return f

    def randomize(self, flag) -> None:
        num_control_point = np.random.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        if flag == 0:
            reference_control_points = np.linspace(0.17, 0.5, num_control_point)
            reference_control_points_start = np.linspace(0., 0.17, 5)
            reference_control_points_end = np.linspace(0.5, 1., 5)
        else:
            reference_control_points = np.linspace(0.5, 0.67, num_control_point)
            reference_control_points_start = np.linspace(0., 0.5, 5)
            reference_control_points_end = np.linspace(0.67, 1., 5)
        self.reference_control_points = np.concatenate(
            [reference_control_points_start[:-1], reference_control_points, reference_control_points_end[1:]])
        self.floating_control_points = np.copy(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = np.random.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )

    def __call__(self, data):
        if np.random.rand() < self.prob:
            img_t = data['subject'].image.data
            img_mask = data['subject'].mask.data
            all_parts = img_mask.max()
            img_t[img_t > 2048] = 2048
            img_t[img_t < -1024] = -1024
            img_t = (img_t + 1024) / 3072.
            # img_t[img_t > 300] = 300
            # img_t[img_t < -200] = -200
            # img_t = (img_t + 200) / 500.

            img_t_flip = 1 - img_t
            img_min, img_max = img_t.min(), img_t.max()
            img_flip_min, img_flip_max = img_t_flip.min(), img_t_flip.max()
            xps = []
            yps = []
            reference_control_points_scaled_all = []
            floating_control_points_scaled_all = []
            flip_flags = []
            for i in range(all_parts):
                if np.random.rand() < 0.3:
                    flag = 1
                    use_min = img_flip_min
                    use_max = img_flip_max
                else:
                    flag = 0
                    use_min = img_min
                    use_max = img_max
                flip_flags.append(flag)
                self.randomize(flag)
                xp = torch.tensor(self.reference_control_points, dtype=img_t.dtype)
                yp = torch.tensor(self.floating_control_points, dtype=img_t.dtype)
                xps.append(xp)
                yps.append(yp)

                # if np.random.rand() < 0.01:  # contrast strength
                #     reference_control_points_scaled = xp * (use_max - use_min) + use_min
                #     floating_control_points_scaled = yp * (use_max - use_min) + use_min
                # else:
                reference_control_points_scaled = xp
                floating_control_points_scaled = yp
                reference_control_points_scaled_all.append(reference_control_points_scaled)
                floating_control_points_scaled_all.append(floating_control_points_scaled)

            img_final = torch.zeros_like(img_t)
            img_final = torch.where(img_mask > 0, img_final, img_t)
            for i in range(all_parts):
                if flip_flags[i] == 0:
                    img_ted = self.interp(img_t, reference_control_points_scaled_all[i],
                                          floating_control_points_scaled_all[i])
                else:
                    img_ted = self.interp(img_t_flip, reference_control_points_scaled_all[i],
                                          floating_control_points_scaled_all[i])
                img_final = torch.where(img_mask == i + 1, img_ted, img_final)
            # import matplotlib.pyplot as plt
            # plt.imshow(img_t[0, :, :, 90], cmap='gray', vmin=0, vmax=1)
            # plt.show()
            data['subject'].image.data = img_final
        return data


if __name__ == '__main__':
    image_path = '/data/sdd/user/rawdata/CT-MRI-Abd/nii/'

    save_path = '/data/sdd/user/rawdata/CT-MRI-Abd/nii-mask-slic'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    all_img = os.listdir(image_path)
    all_ct_img = [img for img in all_img if '_0000' in img]
    all_mri_img = [img for img in all_img if '_0001' in img]

    for img in all_ct_img:
        print(img)
        image = tio.ScalarImage(os.path.join(image_path, img))
        numpy_img = image.data.numpy()
        numpy_img = numpy_img[0, :, :, :]
        numpy_img = numpy_img.transpose(1, 0, 2)
        mask = numpy_img > -300
        mask_processed = np.copy(mask)
        for i in range(mask.shape[2]):
            mask_processed[:, :, i] = morphology.closing(mask[:, :, i], morphology.disk(3))
            mask_processed[:, :, i] = morphology.opening(mask_processed[:, :, i], morphology.disk(3))
            mask_processed[:, :, i] = morphology.remove_small_holes(
                morphology.remove_small_objects(mask_processed[:, :, i], min_size=2000), area_threshold=25000)
        m_slic = segmentation.slic(numpy_img, n_segments=15, mask=mask_processed, multichannel=False, start_label=1,
                                   compactness=0.01)
        m_slic = m_slic.transpose(1, 0, 2)
        m_slic = m_slic.astype(np.int32)
        nii_img = nib.Nifti1Image(m_slic, affine=image.affine)
        nib.save(nii_img, os.path.join(save_path, img))
