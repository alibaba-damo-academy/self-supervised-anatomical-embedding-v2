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

    def __init__(self, num_control_points=(5, 15), prob: float = 0.9) -> None:

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

    def randomize(self) -> None:
        num_control_point = np.random.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        self.reference_control_points = np.linspace(0, 1, num_control_point)
        self.floating_control_points = np.copy(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = np.random.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )

    def __call__(self, data):
        if np.random.rand() < self.prob:
            img_t = data['subject'].image.data
            img_mask = data['subject'].mask.data
            self.randomize()
            if self.reference_control_points is None or self.floating_control_points is None:
                raise RuntimeError("please call the `randomize()` function first.")
            if np.random.rand() < 0.5:  # flip intensity
                img_t2 = 1 - img_t
            else:
                img_t2 = img_t

            img_min, img_max = img_t2.min(), img_t2.max()
            xp = torch.tensor(self.reference_control_points, dtype=img_t.dtype)
            yp = torch.tensor(self.floating_control_points, dtype=img_t.dtype)
            if np.random.rand() < 0.5:  # contrast strength
                reference_control_points_scaled = xp * (img_max - img_min) + img_min
                floating_control_points_scaled = yp * (img_max - img_min) + img_min
            else:
                reference_control_points_scaled = xp
                floating_control_points_scaled = yp
            img_ted = self.interp(img_t2, reference_control_points_scaled, floating_control_points_scaled)
            img_t = torch.where(img_mask > 0, img_ted, img_t)
            # import matplotlib.pyplot as plt
            # plt.imshow(img_t[0, :, :, 90], cmap='gray', vmin=0, vmax=1)
            # plt.show()
            data['subject'].image.data = img_t
        return data


if __name__ == '__main__':
    image_path = '/data/sdd/user/rawdata/CT-MRI-Abd/nii/1058150_0000.nii.gz'
    save_path = '/data/sdd/user/rawdata/CT-MRI-Abd/'
    image = tio.ScalarImage(image_path)
    numpy_img = image.data.numpy()
    numpy_img = numpy_img[0, :, :, :]
    numpy_img = numpy_img.transpose(1, 0, 2)

    body_thres = (600, 50)
    # img_min_max = np.percentile(numpy_img, (2, 99.8)) # for MRI image
    img_min_max = (-1024, 3071)  # for CT image
    mask = numpy_img > (img_min_max[0] + body_thres[0])
    mask_processed = np.copy(mask)
    numpy_img[numpy_img < img_min_max[0]] = img_min_max[0]
    numpy_img[numpy_img > img_min_max[1]] = img_min_max[1]

    numpy_img = (numpy_img - img_min_max[0]) / (img_min_max[1] - img_min_max[0])
    for i in range(mask.shape[2]):
        mask_processed[:, :, i] = morphology.closing(mask[:, :, i], morphology.disk(3))
        mask_processed[:, :, i] = morphology.opening(mask_processed[:, :, i], morphology.disk(3))
        mask_processed[:, :, i] = morphology.remove_small_holes(
            morphology.remove_small_objects(mask_processed[:, :, i], min_size=2000), area_threshold=25000)
    # Superpixel
    # m_slic = segmentation.slic(numpy_img, n_segments=15, mask=mask_processed, multichannel=False, start_label=1,
    #                            compactness=0.05)
    # m_slic = m_slic.transpose(1, 0, 2)
    # m_slic = m_slic.astype(np.int32)
    # nii_img = nib.Nifti1Image(m_slic, affine=image.affine)
    # nib.save(nii_img, os.path.join(save_path, '820691_0000_slic' + '.nii.gz'))

    mask_processed = mask_processed.transpose(1, 0, 2)
    mask_processed = mask_processed.astype(np.int32)
    nii_img = nib.Nifti1Image(mask_processed, affine=image.affine)
    nib.save(nii_img, os.path.join(save_path, '1058150_0000_mask' + '.nii.gz'))

    numpy_img = numpy_img.transpose(1, 0, 2)
    subject = tio.Subject(image=tio.ScalarImage(tensor=numpy_img[np.newaxis, :, :, :], affine=image.affine),
                          mask=tio.LabelMap(tensor=mask_processed[np.newaxis, :, :, :], affine=image.affine))
    ranodmlize = RandHistogramShift()
    data = {}
    data['subject'] = subject
    data = ranodmlize(data)
    processed_img = data['subject'].image.data.numpy()[0, :, :, :]
    nii_img = nib.Nifti1Image(processed_img, affine=image.affine)
    nib.save(nii_img, os.path.join(save_path, '1058150_0000_aug' + '.nii.gz'))

    print('done')
