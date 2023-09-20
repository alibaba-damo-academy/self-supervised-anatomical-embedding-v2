# Xiaoyu Bai, Medical AI Lab, Alibaba DAMO Academy

from mmdet.datasets.builder import PIPELINES
import torchio as tio
import numpy as np
from mmdet.datasets.pipelines import Compose
import copy
import torch
import os
import nibabel as nib
from skimage import measure
from scipy import ndimage
from skimage import morphology
from skimage import segmentation


class Crop_mod(tio.Crop):
    def apply_transform(self, sample):
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = high
        # index_fin = np.array(sample.spatial_shape) - high
        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, index_ini)
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            i0, j0, k0 = index_ini
            i1, j1, k1 = index_fin
            image.set_data(image.data[:, i0:i1, j0:j1, k0:k1].clone())
            image.affine = new_affine
        return sample


def get_range(mask, margin=0):
    """Get up, down, left, right extreme coordinates of a binary mask"""
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return [u, d, l, r]


def meshgrid3d(shape, spacing):
    y_ = np.linspace(0., (shape[1] - 1) * spacing[0], shape[1])
    x_ = np.linspace(0., (shape[2] - 1) * spacing[1], shape[2])
    z_ = np.linspace(0., (shape[3] - 1) * spacing[2], shape[3])

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
    return [y, x, z]


def makelabel(shape):
    all_ = np.ones((1, shape[1] * shape[2] * shape[3]))
    all = all_.reshape(-1, shape[1], shape[2], shape[3])
    all[0, int(shape[1] / 2), int(shape[2] / 2), int(shape[3] / 2)] = 0
    return all


@PIPELINES.register_module()
class RemoveCTBed():
    def __init__(self, thres=-900):
        self.thres = thres

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data
        # img = img[:,:,:,:-1] # tio resample may generate allzero slice at the end
        mask = img > self.thres
        mask_np = mask[0, :, :, :].data.numpy()
        mask_processed = np.copy(mask_np)
        for i in range(mask_np.shape[2]):
            mask_processed[:, :, i] = morphology.closing(mask_np[:, :, i], morphology.disk(3))
            mask_processed[:, :, i] = morphology.opening(mask_processed[:, :, i], morphology.disk(3))
            mask_processed[:, :, i] = morphology.remove_small_holes(
                morphology.remove_small_objects(mask_processed[:, :, i], min_size=2000), area_threshold=40000)

        # mask_np = ndimage.binary_opening(mask_np, structure=np.ones((3, 3, 3)))
        # # mask_np = ndimage.binary_erosion(mask_np, structure=np.ones((5, 5, 5)))
        # mask_np = ndimage.binary_fill_holes(mask_np)
        # l, n = measure.label(mask_np, return_num=True)
        # regions = measure.regionprops(l)
        # max_area = 0
        # max_l = 0
        # for region in regions:
        #     if region.area > max_area:
        #         max_area = region.area
        #         max_l = region.label
        # ind = 1 - (l == max_l)
        # mask_np[ind == 1] = 0

        mask_np = mask_processed[np.newaxis, :, :, :]

        data['body_mask'] = mask_np

        # subject_new = tio.Subject(
        #     image=subject.image,
        #     label=tio.LabelMap(tensor=mask_np, affine=subject.image.affine)
        # )
        subject['body_mask'] = tio.LabelMap(tensor=mask_np, affine=subject.image.affine)
        #
        # l,n = measure.label(mask_np,return_num =True)
        # regions = measure.regionprops(l)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomElasticDeformation():
    def __call__(self, data):
        subject = data['subject']
        random_elastic_transform = tio.RandomElasticDeformation(locked_borders=1)
        subject = random_elastic_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: Compose(v) for k, v in transform_group.items()}

    def __call__(self, data):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(data))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results


@PIPELINES.register_module()
class LoadTestTioImage:
    """Load image as torchio subject.
    """

    def __init__(self, re_orient=True, landmark=False, suffix=None):
        self.re_orient = re_orient
        self.landmark = landmark
        if suffix is not None:
            self.suffix = suffix
        else:
            self.suffix = None

    def __call__(self, data):
        # time1 = time.time()
        data_path = data['data_path']
        image_fn = data['image_fn']
        if self.landmark:  # the fold structure of ChestCT landmark dataset is /fold/casefold/casenii
            if self.suffix is not None:
                img_tio = tio.ScalarImage(os.path.join(data_path, image_fn.split('.', 1)[0], image_fn + self.suffix))
                data['image_fn'] = image_fn + self.suffix
            else:
                img_tio = tio.ScalarImage(os.path.join(data_path, image_fn.split('.', 1)[0], image_fn))
        else:
            img_tio = tio.ScalarImage(data_path + image_fn)
        ToCanonical = tio.ToCanonical()
        img_tio = ToCanonical(img_tio)
        img_tio.data = torch.flip(img_tio.data, (1, 2))
        img_tio.affine = np.array(
            [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])

        if self.re_orient:
            img_data = img_tio.data
            img_tio.data = img_data.permute(0, 2, 1, 3)
            img_tio.affine = np.array(
                [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
        img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
        img_tio_spacing = img_tio.spacing
        subject = tio.Subject(
            image=img_tio
        )
        data['subject'] = subject
        # time2 = time.time()
        # print('load image info time cost:',time2-time1)
        return data


# @PIPELINES.register_module()
# class LoadTioImageWithMask:
#     """Load image as torchio subject.
#     """
#
#     def __init__(self, re_orient=True):
#         self.re_orient = re_orient
#
#     def __call__(self, data):
#         # time1 = time.time()
#         data_path = data['data_path']
#         image_fn = data['image_fn']
#         if 'mask' in data.keys():
#             mask_path = data['mask_path']
#             mask_tio = tio.LabelMap(os.path.join(mask_path, image_fn))
#
#         img_tio = tio.ScalarImage(os.path.join(data_path, image_fn))
#
#         # for NIH lymph node dataset covert orientation to PLS+
#         if self.re_orient:
#             img_data = img_tio.data
#             img_tio.data = img_data.permute(0, 2, 1, 3)
#             img_tio.affine = np.array(
#                 [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
#             if 'mask' in data.keys():
#                 mask_data = mask_tio.data
#                 mask_tio.data = mask_data.permute(0, 2, 1, 3)
#                 mask_tio.affine = np.array(
#                     [mask_tio.affine[1, :], mask_tio.affine[0, :], mask_tio.affine[2, :], mask_tio.affine[3, :]])
#         img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
#         img_tio_spacing = img_tio.spacing
#         meshs = meshgrid3d(img_tio_shape, img_tio_spacing)
#         labels = makelabel(img_tio_shape)
#         if 'mask' in data.keys():
#             subject = tio.Subject(
#                 image=img_tio,
#                 mask=mask_tio,
#                 label_tio_y=tio.ScalarImage(
#                     tensor=meshs[0].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 label_tio_x=tio.ScalarImage(
#                     tensor=meshs[1].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 label_tio_z=tio.ScalarImage(
#                     tensor=meshs[2].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 labels_map=tio.LabelMap(tensor=labels, affine=img_tio.affine)
#                 # labels_map=tio.ScalarImage(tensor=labels, affine=img_tio.affine)
#             )
#         else:
#             subject = tio.Subject(
#                 image=img_tio,
#                 label_tio_y=tio.ScalarImage(
#                     tensor=meshs[0].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 label_tio_x=tio.ScalarImage(
#                     tensor=meshs[1].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 label_tio_z=tio.ScalarImage(
#                     tensor=meshs[2].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
#                     affine=img_tio.affine),
#                 labels_map=tio.LabelMap(tensor=labels, affine=img_tio.affine)
#                 # labels_map=tio.ScalarImage(tensor=labels, affine=img_tio.affine)
#             )
#         data['subject'] = subject
#         return data


@PIPELINES.register_module()
class GeneratePairedImagesInfo:
    def __call__(self, data):
        data1 = {'data_path': data['data_path'], 'image_fn': data['image_fn1']}
        data2 = {'data_path': data['data_path'], 'image_fn': data['image_fn2']}
        data['paired'] = [data['image_fn1'], data['image_fn2']]
        data['data1'] = data1
        data['data2'] = data2
        return data


@PIPELINES.register_module()
class SeperateInfoByImage:
    def __call__(self, data):
        assert 'tag' in data.keys()
        paired = data['paired']
        tag = data['tag']
        anchor = data['anchor']
        mask_path = data['mask_path']
        if data['tag'] == 'im1':
            data = data['data1']
        elif data['tag'] == 'im2':
            data = data['data2']
        else:
            return NotImplementedError
        data['paired'] = paired
        data['tag'] = tag
        data['anchor'] = anchor
        data['mask_path'] = mask_path
        return data


@PIPELINES.register_module()
class LoadTioImage:
    """Load image as torchio subject.
    """

    def __init__(self, re_orient=True, with_mesh=True):
        self.re_orient = re_orient
        self.with_mesh = with_mesh

    def __call__(self, data):
        # time1 = time.time()
        data_path = data['data_path']
        image_fn = data['image_fn']
        img_tio = tio.ScalarImage(os.path.join(data_path, image_fn))
        # for NIH lymph node dataset covert orientation to PLS+
        if self.re_orient:
            img_data = img_tio.data
            img_tio.data = img_data.permute(0, 2, 1, 3)
            img_tio.affine = np.array(
                [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
        img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
        img_tio_spacing = img_tio.spacing
        if self.with_mesh:
            meshs = meshgrid3d(img_tio_shape, img_tio_spacing)
            labels = makelabel(img_tio_shape)
            subject = tio.Subject(
                image=img_tio,
                label_tio_y=tio.ScalarImage(
                    tensor=meshs[0].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                    affine=img_tio.affine),
                label_tio_x=tio.ScalarImage(
                    tensor=meshs[1].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                    affine=img_tio.affine),
                label_tio_z=tio.ScalarImage(
                    tensor=meshs[2].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                    affine=img_tio.affine),
                labels_map=tio.LabelMap(tensor=labels, affine=img_tio.affine)
                # labels_map=tio.ScalarImage(tensor=labels, affine=img_tio.affine)
            )
        else:
            subject = tio.Subject(
                image=img_tio,
            )
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, data):
        for k, v in self.attrs.items():
            assert k not in data
            data[k] = v
        return data


@PIPELINES.register_module()
class RandomBlur3d():
    def __init__(self, std=(0, 4)):
        self.std = std

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randomblur_transform = tio.RandomBlur(std=self.std)
        image = randomblur_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomBiasField():
    def __init__(self, coefficients=0.5, order=3):
        self.coefficients = coefficients
        self.order = order

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randombias_transform = tio.RandomBiasField(coefficients=self.coefficients, order=self.order)
        image = randombias_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandHistogramShift_block():
    """
    Apply random nonlinear transform to the image's intensity histogram.
    Args:
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
    """

    def __init__(self, num_control_points=(5, 15), prob: float = 0.6) -> None:

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

    def randomize(self, flag, is_mri=False) -> None:
        num_control_point = np.random.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        random_s = np.random.uniform(0.1, 0.5)
        random_e = np.random.uniform(0.6, 0.8)
        if flag == 0:
            if is_mri:
                reference_control_points = np.linspace(0., 1., num_control_point)
            else:
                # reference_control_points = np.linspace(0.17, 0.5, num_control_point)
                # reference_control_points_start = np.linspace(0., 0.17, 5)
                # reference_control_points_end = np.linspace(0.5, 1., 5)
                reference_control_points = np.linspace(random_s, random_e, num_control_point)
                reference_control_points_start = np.linspace(0., random_s, 5)
                reference_control_points_end = np.linspace(random_e, 1., 5)
        else:
            if is_mri:
                reference_control_points = np.linspace(0., 1., num_control_point)
            else:
                # reference_control_points = np.linspace(0.5, 0.67, num_control_point)
                # reference_control_points_start = np.linspace(0., 0.5, 5)
                # reference_control_points_end = np.linspace(0.67, 1., 5)
                reference_control_points = np.linspace(random_s, random_e, num_control_point)
                reference_control_points_start = np.linspace(0., random_s, 5)
                reference_control_points_end = np.linspace(random_e, 1., 5)
        if is_mri:
            self.reference_control_points = reference_control_points
        else:
            self.reference_control_points = np.concatenate(
                [reference_control_points_start[:-1], reference_control_points, reference_control_points_end[1:]])
        self.floating_control_points = np.copy(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = np.random.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )

    def __call__(self, data, is_mri=False):
        if np.random.rand() < self.prob:
            img_t = data['subject'].image.data
            img_mask = data['subject'].mask.data
            modality = data['mod']
            if modality == 'mri':
                is_mri = True
            all_parts = img_mask.max()
            if all_parts < 1:
                return data
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
                self.randomize(flag, is_mri=is_mri)
                xp = torch.tensor(self.reference_control_points, dtype=img_t.dtype)
                yp = torch.tensor(self.floating_control_points, dtype=img_t.dtype)
                xps.append(xp)
                yps.append(yp)

                if np.random.rand() < 0.2:  # contrast strength
                    reference_control_points_scaled = xp * (use_max - use_min) + use_min
                    floating_control_points_scaled = yp * (use_max - use_min) + use_min
                else:
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


@PIPELINES.register_module()
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
            if np.random.rand() < 0.3:  # flip intensity
                img_t2 = 1 - img_t
            else:
                img_t2 = img_t

            img_min, img_max = img_t2.min(), img_t2.max()
            xp = torch.tensor(self.reference_control_points, dtype=img_t.dtype)
            yp = torch.tensor(self.floating_control_points, dtype=img_t.dtype)
            if np.random.rand() < 0.3:  # contrast strength
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


@PIPELINES.register_module()
class RandomGamma():
    def __init__(self, log_gamma=(-1, 1)):
        self.log_gamma = log_gamma

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randomgamma_transform = tio.RandomGamma(log_gamma=self.log_gamma)
        image = randomgamma_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomNoise3d():
    def __init__(self, std=(0, 0.001)):
        self.std = std

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randomnoise_transform = tio.RandomNoise(mean=0, std=self.std)
        image = randomnoise_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class Resample():
    def __init__(self, norm_spacing=(2., 2., 2.), intra_volume=True, crop_artifacts=True, dynamic_spacing=False):
        self.norm_spacing = norm_spacing
        self.intra_volume = intra_volume
        self.crop_artifacts = crop_artifacts
        self.dynamic_spacing = dynamic_spacing

    def __call__(self, data):
        subject = data['subject']
        keys = list(data.keys())
        if 'tag' in keys and self.intra_volume:
            tag = data['tag']
            spacing_matrix = data['spacing_matrix']
            if tag == 'view1':
                spacing = spacing_matrix[0]
            else:
                spacing = spacing_matrix[1]
        elif self.dynamic_spacing:
            spacing = data['resample_spacing']
        else:
            spacing = np.array(self.norm_spacing)
        resample_transform = tio.Resample(target=spacing)
        subject = resample_transform(subject)
        if self.crop_artifacts:
            img_data = subject['image'].data
            crop_flag = 0
            crop_info = np.array(subject.shape[1:]).astype(int)
            if (img_data[0, :, :, -1] == 0).all():
                crop_flag = 1
                # crop_info[-1] = crop_info[-1] - 1
                crop_info[2] = crop_info[2] - 1
            if (img_data[0, :, -1, :] == 0).all():
                crop_flag = 1
                # crop_info[-1] = crop_info[-1] - 1
                crop_info[1] = crop_info[1] - 1
            if (img_data[0, -1, :, :] == 0).all():
                crop_flag = 1
                # crop_info[-1] = crop_info[-1] - 1
                crop_info[0] = crop_info[0] - 1
            if crop_flag == 1:
                crop_transform = Crop_mod([0, crop_info[0], 0, crop_info[1], 0, crop_info[2]])  # y1y2x1x2z1z2
                subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class GenerateMeshGrid():
    def __call__(self, data):
        sub = data['subject']
        image = sub.image.data
        y_sub = sub.label_tio_y.data
        x_sub = sub.label_tio_x.data
        z_sub = sub.label_tio_z.data
        pos_mm = torch.cat((y_sub, x_sub, z_sub), axis=0)
        # pos1_mm = pos1_mm.transpose((1, 2, 3, 0))
        sub_valid = sub.labels_map.data
        data['img'] = image.permute(0, 3, 1, 2)
        data['meshgrid'] = pos_mm.permute(0, 3, 1, 2)
        data['valid'] = sub_valid.permute(0, 3, 1, 2)
        data['spacing'] = (sub.image.spacing[2],sub.image.spacing[0],sub.image.spacing[1])
        return data


@PIPELINES.register_module()
class GenerateMetaInfo():
    def __init__(self, is_train=True, with_mask=False):
        self.is_train = is_train
        self.with_mask = with_mask

    def __call__(self, data):
        data['filename'] = data['image_fn']
        keys = list(data.keys())
        if 'tag' in keys and 'crop_matrix_origin' in keys:
            tag = data['tag']
            if tag == 'view1':
                data['crop_info'] = data['crop_matrix_origin'][0:2, :]
            else:
                data['crop_info'] = data['crop_matrix_origin'][2:, :]
        if 'img' not in keys:
            sub = data['subject']
            image = sub.image.data
            if self.is_train:
                data['img'] = image.permute(0, 3, 1, 2)
            else:
                data['img'] = [image.permute(0, 3, 1, 2)]
        if self.with_mask:
            sub = data['subject']
            mask = sub.mask.data
            data['mask'] = mask.permute(0, 3, 1, 2)
        data['volume_size'] = data['img'][0].shape

        return data


@PIPELINES.register_module()
class ComputeCorrespond():
    def __init__(self, shape=(96, 96, 24)):
        self.shape = shape

    def __call__(self, data):
        subject = data['subject']
        resize_transform = tio.Resize(self.shape)
        subject = resize_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class Resize3d():
    def __init__(self, shape=(96, 96, 24)):
        self.shape = shape

    def __call__(self, data):
        subject = data['subject']
        resize_transform = tio.Resize(self.shape)
        subject = resize_transform(subject)
        data['subject'] = subject

        return data


@PIPELINES.register_module()
class Cropz():
    def __init__(self, x_slice=100, y_slice=100, z_slice=100):
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.z_slice = z_slice

    def __call__(self, data):
        subject = data['subject']
        shape = subject.image.shape
        flag = False
        crop_info = [0, shape[1],
                     0, shape[2],
                     0, shape[3]]
        if shape[3] > self.z_slice + 10:
            seedz = np.random.randint(0, shape[3] - self.z_slice - 1)
            crop_info[4] = seedz
            crop_info[5] = seedz + self.z_slice
            flag = True
        if shape[1] > self.y_slice + 10:
            seedy = np.random.randint(0, shape[1] - self.y_slice - 1)
            crop_info[0] = seedy
            crop_info[1] = seedy + self.y_slice
            flag = True
        if shape[2] > self.x_slice + 10:
            seedx = np.random.randint(0, shape[2] - self.x_slice - 1)
            crop_info[2] = seedx
            crop_info[3] = seedx + self.x_slice
            flag = True
        if flag:
            crop_info = np.array(crop_info).astype(int)
            crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
            subject = crop_transform(subject)
            data['subject'] = subject
        return data


@PIPELINES.register_module()
class CropBackground():
    def __init__(self, thres=-500, by_mask=False):
        self.thres = thres
        self.by_mask = by_mask

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data
        if not self.by_mask:
            # img = img[:,:,:,:-1] # tio resample may generate allzero slice at the end
            mask = img > self.thres
        else:
            mask = subject['body_mask'].data
        foreground_region = torch.where(mask == 1)
        crop_info = (foreground_region[1].min(), foreground_region[1].max(),
                     foreground_region[2].min(), foreground_region[2].max(),
                     foreground_region[3].min(),
                     foreground_region[3].max())

        crop_info = np.array(crop_info).astype(int)
        crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
        subject = crop_transform(subject)
        data['subject'] = subject
        data['crop_info'] = crop_info
        return data


@PIPELINES.register_module()
class RandomCrop3d():
    def __init__(self, thres=0.99, min_z=18, max_z=32):
        self.thres = thres
        self.min_z = min_z
        self.max_z = max_z

    def __call__(self, data):
        subject = data['subject']
        img_shape = subject.image.shape
        p = np.random.uniform(0, 1)
        z_use = np.random.randint(self.min_z, self.max_z)
        if p < self.thres and (img_shape[3] - z_use - 1) > 0:
            seed = np.random.randint(0, img_shape[3] - z_use - 1)
            crop_info = np.array([0, img_shape[1], 0, img_shape[2], seed, seed + z_use])
            crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
            subject = crop_transform(subject)
            data['subject'] = subject
        return data


@PIPELINES.register_module()
class Crop():
    def __init__(self, switch='origin', by_view=True, fix_size=(96, 96, 32)):
        self.switch = switch
        self.fix_size = fix_size
        self.by_view = by_view
        assert self.switch in ['origin', 'fix']

    def __call__(self, data):
        # time1 = time.time()
        subject = data['subject']
        if 'crop_matrix_origin' in data.keys():
            crop_matrix_origin = data['crop_matrix_origin']
        if 'tag' in data.keys():
            tag = data['tag']
            if tag == 'view1':
                view = 0
            else:
                view = 2
        if self.switch == 'origin':
            if self.by_view:
                crop_info = (crop_matrix_origin[view][0],
                             crop_matrix_origin[view + 1][0],
                             crop_matrix_origin[view][1],
                             crop_matrix_origin[view + 1][1],
                             crop_matrix_origin[view][2],
                             crop_matrix_origin[view + 1][2])
            else:
                crop_info = (crop_matrix_origin[0][0],
                             crop_matrix_origin[1][0],
                             crop_matrix_origin[0][1],
                             crop_matrix_origin[1][1],
                             crop_matrix_origin[0][2],
                             crop_matrix_origin[1][2])
        else:
            subject_shape = subject.shape[1:]
            assert subject_shape >= self.fix_size, print(data['image_fn'])

            crop_info = (int(subject_shape[0] / 2) - self.fix_size[0] / 2,
                         int(subject_shape[0] / 2) + self.fix_size[0] / 2,
                         int(subject_shape[1] / 2) - self.fix_size[1] / 2,
                         int(subject_shape[1] / 2) + self.fix_size[1] / 2,
                         int(subject_shape[2] / 2) - self.fix_size[2] / 2,
                         int(subject_shape[2] / 2) + self.fix_size[2] / 2)
        crop_info = np.array(crop_info).astype(int)
        crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
        subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomAffine3d():
    def __init__(self, scales=0, degrees=(-10, 10, -10, 10, -10, 10), translation=0):
        self.scales = scales
        self.degrees = degrees
        self.translation = translation

    def __call__(self, data):
        subject = data['subject']
        random_affine_transform = tio.RandomAffine(scales=self.scales, degrees=self.degrees,
                                                   translation=self.translation)
        subject = random_affine_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class DynamicRescaleIntensity():
    def __init__(self, out_min_max=(0, 255), bias=50):
        self.out_min_max = out_min_max
        self.bias = bias

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data.type(torch.FloatTensor)
        img_numpy = img.numpy()
        # in_min_max = torch.quantile(img.type(torch.FloatTensor), torch.tensor([0.05,0.99])) # torch quantile like np.quantile not np.percentail
        in_min_max = np.percentile(img_numpy, (0.2, 99.8))
        if in_min_max[1] < 1:  # we found for some normalized MR, we should not re-normalize the intensity
            in_min_max[1] = 1
        img[img < in_min_max[0]] = in_min_max[0]
        img[img > in_min_max[1]] = in_min_max[1]
        subject.image.data = img
        rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max,
                                                 in_min_max=(int(in_min_max[0]), int(in_min_max[1])))
        image = rescale_transform(subject.image)
        subject.image.data = image.data - self.bias
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RescaleIntensity_modality():
    def __init__(self, out_min_max=(0, 255), in_min_max=(-1024, 3071), bias=50):
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max
        self.bias = bias

    def __call__(self, data):
        subject = data['subject']
        modality = data['mod']
        img = subject.image.data.type(torch.FloatTensor)
        img_numpy = img.numpy()
        if modality == 'ct':
            img[img < self.in_min_max[0]] = self.in_min_max[0]
            img[img > self.in_min_max[1]] = self.in_min_max[1]
            subject.image.data = img
            rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max, in_min_max=self.in_min_max)

        elif modality == 'mri':
            in_min_max = np.percentile(img_numpy, (0.2, 99.8))
            img[img < in_min_max[0]] = in_min_max[0]
            img[img > in_min_max[1]] = in_min_max[1]
            subject.image.data = img
            rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max,
                                                     in_min_max=(int(in_min_max[0]), int(in_min_max[1])))
        image = rescale_transform(subject.image)
        subject.image.data = image.data - self.bias
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RescaleIntensity():
    def __init__(self, out_min_max=(0, 255), in_min_max=(-1024, 3071), bias=50):
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max
        self.bias = bias

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data
        img[img < self.in_min_max[0]] = self.in_min_max[0]
        img[img > self.in_min_max[1]] = self.in_min_max[1]
        subject.image.data = img
        rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max, in_min_max=self.in_min_max)
        image = rescale_transform(subject.image)
        subject.image.data = image.data - self.bias
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class ComputeAugParam_sample():
    def __init__(self, scale=(0.8, 1.2), standard_spacing=(2., 2., 2.), patch_size=(96, 96, 32)):
        self.scale = scale
        self.standard_spacing = standard_spacing
        self.patch_size = patch_size
        # self.sample = sample

    def __call__(self, data):
        if 'anno' in data.keys():
            sample = True
        else:
            sample = False
        if 'mask' in data['subject'].keys():
            sample_by_mask = True
        else:
            sample_by_mask = False
        view1_scale = np.random.uniform(self.scale[0], self.scale[1])
        view2_scale = np.random.uniform(self.scale[0], self.scale[1])
        # view2_scale = view1_scale
        origin_spacing = data['subject'].image.spacing
        origin_size = data['subject'].image.shape
        # print(origin_size)
        standard_size = [np.floor(origin_spacing[0] / self.standard_spacing[0] * origin_size[1]).astype(int),
                         np.floor(origin_spacing[1] / self.standard_spacing[1] * origin_size[2]).astype(int),
                         np.floor(origin_spacing[2] / self.standard_spacing[2] * origin_size[3]).astype(int)]
        view1_patch_size = np.ceil(np.array(self.patch_size) * view1_scale).astype(int)  # y,x,z
        view2_patch_size = np.ceil(np.array(self.patch_size) * view2_scale).astype(int)
        affine_standard_to_origin = np.array([self.standard_spacing[0] / origin_spacing[0], 0, 0, 0,
                                              0, self.standard_spacing[1] / origin_spacing[1], 0, 0,
                                              0, 0, self.standard_spacing[2] / origin_spacing[2], 0,
                                              0, 0, 0, 1]).reshape(4, 4).transpose()
        view1_crop_size = np.ceil(view1_patch_size * 1.05).astype(int)
        view2_crop_size = np.ceil(view2_patch_size * 1.05).astype(int)

        margin_x = 1
        margin_y = 1
        margin_z = 1
        rangex_min = margin_x
        rangex_max = standard_size[1] - margin_x
        rangey_min = margin_y
        rangey_max = standard_size[0] - margin_y
        rangez_min = margin_z
        rangez_max = standard_size[2] - margin_z

        p_intersect = np.random.rand(1)
        if p_intersect <= 0.2:
            view1_left_up = (np.random.randint(rangey_min, rangey_max - view1_crop_size[0]),
                             np.random.randint(rangex_min, rangex_max - view1_crop_size[1]),
                             np.random.randint(rangez_min, rangez_max - view1_crop_size[2]))
            view1_left_up = np.asarray(view1_left_up)
            view2_left_up = (np.random.randint(rangey_min, rangey_max - view2_crop_size[0]),
                             np.random.randint(rangex_min, rangex_max - view2_crop_size[1]),
                             np.random.randint(rangez_min, rangez_max - view2_crop_size[2]))
            view2_left_up = np.asarray(view2_left_up)
        else:
            if sample == True:
                crop_info = data['crop_info']
                p_sample = np.random.rand(1)
                anno = data['anno']
                anno_cropped = {}
                for key in anno.keys():
                    anno_cropped[int((key - crop_info[4]) * origin_spacing[0] / self.standard_spacing[0])] = []
                    for bbx in anno[key]:
                        bbx_tmp = np.array(bbx)
                        bbx_tmp = bbx_tmp - [crop_info[2], crop_info[0]] * 2
                        bbx_tmp = bbx_tmp * ([origin_spacing[1] / self.standard_spacing[1],
                                              origin_spacing[0] / self.standard_spacing[0]] * 2)
                        anno_cropped[int((key - crop_info[4]) * origin_spacing[0] / self.standard_spacing[0])].append(
                            bbx_tmp.tolist())

                zs = list(anno_cropped.keys())
                zs = [z for z in zs if z > 1 and z < standard_size[2] - 1]
                seed_ind = np.random.randint(0, zs.__len__())
                boxes_len = anno_cropped[zs[seed_ind]].__len__()
                box_ind = np.random.randint(0, boxes_len)
            else:
                p_sample = 0.1
            if p_sample < 0.5:
                seed_margin_xy = 10
                seed_margin_z = 4
                seed_point = (np.random.randint(rangey_min + seed_margin_xy, rangey_max - seed_margin_xy),
                              np.random.randint(rangex_min + seed_margin_xy, rangex_max - seed_margin_xy),
                              np.random.randint(rangez_min + seed_margin_z, rangez_max - seed_margin_z))
                seed_point = np.array(seed_point)
            else:
                bbx = anno_cropped[zs[seed_ind]][box_ind]
                seed_point = ((bbx[1] + bbx[3]) / 2, (bbx[0] + bbx[2]) / 2, zs[seed_ind])
                seed_point = np.array(seed_point, dtype=int)

            assert max(seed_point[2] - view1_crop_size[2], rangez_min) < min(rangez_max - view1_crop_size[2],
                                                                             seed_point[2]), data['image_fn']

            view1_left_up = (np.random.randint(max(seed_point[0] - view1_crop_size[0], rangey_min),
                                               min(rangey_max - view1_crop_size[0], seed_point[0])),
                             np.random.randint(max(seed_point[1] - view1_crop_size[1], rangex_min),
                                               min(rangex_max - view1_crop_size[1], seed_point[1])),
                             np.random.randint(max(seed_point[2] - view1_crop_size[2], rangez_min),
                                               min(rangez_max - view1_crop_size[2], seed_point[2])))
            view1_left_up = np.asarray(view1_left_up)

            view2_left_up = (np.random.randint(max(seed_point[0] - view2_crop_size[0], rangey_min),
                                               min(rangey_max - view2_crop_size[0], seed_point[0])),
                             np.random.randint(max(seed_point[1] - view2_crop_size[1], rangex_min),
                                               min(rangex_max - view2_crop_size[1], seed_point[1])),
                             np.random.randint(max(seed_point[2] - view2_crop_size[2], rangez_min),
                                               min(rangez_max - view2_crop_size[2], seed_point[2])))
            view2_left_up = np.asarray(view2_left_up)

        view1_y1x1z1_crop = view1_left_up
        view1_y2x2z2_crop = view1_left_up + view1_crop_size
        view2_y1x1z1_crop = view2_left_up
        view2_y2x2z2_crop = view2_left_up + view2_crop_size

        crop_matrix = np.array([view1_y1x1z1_crop, view1_y2x2z2_crop, view2_y1x1z1_crop, view2_y2x2z2_crop])

        crop_matrix_origin = np.dot(crop_matrix, affine_standard_to_origin[:3, :3])
        crop_matrix_origin[::2, :] = np.floor(crop_matrix_origin[::2, :])
        crop_matrix_origin[1::2, :] = np.ceil(crop_matrix_origin[1::2, :])
        crop_matrix_origin = crop_matrix_origin.astype(int)
        view1_spacing = np.array(self.standard_spacing) * view1_scale
        view2_spacing = np.array(self.standard_spacing) * view2_scale
        spacing_matrix = np.array([view1_spacing, view2_spacing])

        data['crop_matrix_origin'] = crop_matrix_origin
        data['spacing_matrix'] = spacing_matrix
        return data


@PIPELINES.register_module()
class ComputeAugParam_masksample():
    def __init__(self, scale=(0.8, 1.2), standard_spacing=(2., 2., 2.), patch_size=(96, 96, 32)):
        self.scale = scale
        self.standard_spacing = standard_spacing
        self.patch_size = patch_size
        # self.sample = sample

    def __call__(self, data):
        if 'mask' in data['subject'].keys():
            sample_by_mask = True
        else:
            sample_by_mask = False
        view1_scale = np.random.uniform(self.scale[0], self.scale[1])
        view2_scale = np.random.uniform(self.scale[0], self.scale[1])
        # view2_scale = view1_scale
        origin_spacing = data['subject'].image.spacing
        origin_size = data['subject'].image.shape
        # print(origin_size)
        standard_size = [np.floor(origin_spacing[0] / self.standard_spacing[0] * origin_size[1]).astype(int),
                         np.floor(origin_spacing[1] / self.standard_spacing[1] * origin_size[2]).astype(int),
                         np.floor(origin_spacing[2] / self.standard_spacing[2] * origin_size[3]).astype(int)]
        view1_patch_size = np.ceil(np.array(self.patch_size) * view1_scale).astype(int)  # y,x,z
        view2_patch_size = np.ceil(np.array(self.patch_size) * view2_scale).astype(int)
        affine_standard_to_origin = np.array([self.standard_spacing[0] / origin_spacing[0], 0, 0, 0,
                                              0, self.standard_spacing[1] / origin_spacing[1], 0, 0,
                                              0, 0, self.standard_spacing[2] / origin_spacing[2], 0,
                                              0, 0, 0, 1]).reshape(4, 4).transpose()
        view1_crop_size = np.ceil(view1_patch_size * 1.05).astype(int)
        view2_crop_size = np.ceil(view2_patch_size * 1.05).astype(int)

        margin_x = 1
        margin_y = 1
        margin_z = 1
        rangex_min = margin_x
        rangex_max = standard_size[1] - margin_x
        rangey_min = margin_y
        rangey_max = standard_size[0] - margin_y
        rangez_min = margin_z
        rangez_max = standard_size[2] - margin_z


        p_intersect = np.random.rand(1)
        if p_intersect <= 0.1:
            view1_left_up = (np.random.randint(rangey_min, rangey_max - view1_crop_size[0]),
                             np.random.randint(rangex_min, rangex_max - view1_crop_size[1]),
                             np.random.randint(rangez_min, rangez_max - view1_crop_size[2]))
            view1_left_up = np.asarray(view1_left_up)
            view2_left_up = (np.random.randint(rangey_min, rangey_max - view2_crop_size[0]),
                             np.random.randint(rangex_min, rangex_max - view2_crop_size[1]),
                             np.random.randint(rangez_min, rangez_max - view2_crop_size[2]))
            view2_left_up = np.asarray(view2_left_up)
        else:
            if sample_by_mask == True:
                p_sample = np.random.rand(1)
                mask = data['subject']['mask'].data[0, :, :, :].clone().numpy()
                # print(mask.shape)
            else:
                p_sample = 0.1
            if p_sample <= 0.1:
                seed_margin_xy = 10
                seed_margin_z = 4
                seed_point = (np.random.randint(rangey_min + seed_margin_xy, rangey_max - seed_margin_xy),
                              np.random.randint(rangex_min + seed_margin_xy, rangex_max - seed_margin_xy),
                              np.random.randint(rangez_min + seed_margin_z, rangez_max - seed_margin_z))
                seed_point = np.array(seed_point)
            else:
                mask[:, :, :5] = 0
                mask[:, :, -5::1] = 0
                mask[:5, :, :] = 0
                mask[-5::1, :, :] = 0
                mask[:, :5, :] = 0
                mask[:, -5::1, :] = 0
                ys, xs, zs = np.where(mask > 0)
                body_len = ys.shape[0]
                ind = np.random.randint(0, body_len - 1)
                seed_point = np.array([ys[ind], xs[ind], zs[ind]], dtype=int)
                seed_point = np.dot(seed_point, np.linalg.inv(affine_standard_to_origin[:3, :3])).astype(int)

            assert max(seed_point[2] - view1_crop_size[2], rangez_min) < min(rangez_max - view1_crop_size[2],
                                                                             seed_point[2]), [rangez_max, seed_point[2],
                                                                                              view1_crop_size[2],data['image_fn']]
            assert max(seed_point[1] - view1_crop_size[1], rangex_min) < min(rangex_max - view1_crop_size[1],
                                                                             seed_point[1]), [rangex_max,data['image_fn']]
            assert max(seed_point[0] - view1_crop_size[0], rangey_min) < min(rangey_max - view1_crop_size[0],
                                                                             seed_point[0]), [rangey_max,view1_crop_size,data['image_fn']]

            assert max(seed_point[2] - view2_crop_size[2], rangez_min) < min(rangez_max - view2_crop_size[2],
                                                                             seed_point[2]), [rangez_max, seed_point[2],
                                                                                              [view2_crop_size[2]],data['image_fn']]
            assert max(seed_point[1] - view2_crop_size[1], rangex_min) < min(rangex_max - view2_crop_size[1],
                                                                             seed_point[1]), [rangex_max,data['image_fn']]
            assert max(seed_point[0] - view2_crop_size[0], rangey_min) < min(rangey_max - view2_crop_size[0],
                                                                             seed_point[0]), [rangey_max,view1_crop_size,data['image_fn']]

            view1_left_up = (np.random.randint(max(seed_point[0] - view1_crop_size[0], rangey_min),
                                               min(rangey_max - view1_crop_size[0], seed_point[0])),
                             np.random.randint(max(seed_point[1] - view1_crop_size[1], rangex_min),
                                               min(rangex_max - view1_crop_size[1], seed_point[1])),
                             np.random.randint(max(seed_point[2] - view1_crop_size[2], rangez_min),
                                               min(rangez_max - view1_crop_size[2], seed_point[2])))
            view1_left_up = np.asarray(view1_left_up)

            view2_left_up = (np.random.randint(max(seed_point[0] - view2_crop_size[0], rangey_min),
                                               min(rangey_max - view2_crop_size[0], seed_point[0])),
                             np.random.randint(max(seed_point[1] - view2_crop_size[1], rangex_min),
                                               min(rangex_max - view2_crop_size[1], seed_point[1])),
                             np.random.randint(max(seed_point[2] - view2_crop_size[2], rangez_min),
                                               min(rangez_max - view2_crop_size[2], seed_point[2])))
            view2_left_up = np.asarray(view2_left_up)

        view1_y1x1z1_crop = view1_left_up
        view1_y2x2z2_crop = view1_left_up + view1_crop_size
        view2_y1x1z1_crop = view2_left_up
        view2_y2x2z2_crop = view2_left_up + view2_crop_size

        crop_matrix = np.array([view1_y1x1z1_crop, view1_y2x2z2_crop, view2_y1x1z1_crop, view2_y2x2z2_crop])

        crop_matrix_origin = np.dot(crop_matrix, affine_standard_to_origin[:3, :3])
        crop_matrix_origin[::2, :] = np.floor(crop_matrix_origin[::2, :])
        crop_matrix_origin[1::2, :] = np.ceil(crop_matrix_origin[1::2, :])
        crop_matrix_origin = crop_matrix_origin.astype(int)
        view1_spacing = np.array(self.standard_spacing) * view1_scale
        view2_spacing = np.array(self.standard_spacing) * view2_scale
        spacing_matrix = np.array([view1_spacing, view2_spacing])

        data['crop_matrix_origin'] = crop_matrix_origin
        data['spacing_matrix'] = spacing_matrix
        return data


@PIPELINES.register_module()
class ComputeAugParam():
    def __init__(self, scale=(0.8, 1.2), standard_spacing=(2., 2., 2.), patch_size=(96, 96, 32)):
        self.scale = scale
        self.standard_spacing = standard_spacing
        self.patch_size = patch_size
        # self.sample = sample

    def __call__(self, data):
        if 'anno' in data.keys():
            sample = True
        else:
            sample = False
        view1_scale = np.random.uniform(self.scale[0], self.scale[1])
        view2_scale = np.random.uniform(self.scale[0], self.scale[1])
        # view2_scale = view1_scale
        origin_spacing = data['subject'].image.spacing
        origin_size = data['subject'].image.shape
        # print(origin_size)
        standard_size = [np.floor(origin_spacing[0] / self.standard_spacing[0] * origin_size[1]).astype(int),
                         np.floor(origin_spacing[1] / self.standard_spacing[1] * origin_size[2]).astype(int),
                         np.floor(origin_spacing[2] / self.standard_spacing[2] * origin_size[3]).astype(int)]
        view1_patch_size = np.ceil(np.array(self.patch_size) * view1_scale).astype(int)  # y,x,z
        view2_patch_size = np.ceil(np.array(self.patch_size) * view2_scale).astype(int)
        affine_standard_to_origin = np.array([self.standard_spacing[0] / origin_spacing[0], 0, 0, 0,
                                              0, self.standard_spacing[1] / origin_spacing[1], 0, 0,
                                              0, 0, self.standard_spacing[2] / origin_spacing[2], 0,
                                              0, 0, 0, 1]).reshape(4, 4).transpose()
        view1_crop_size = np.ceil(view1_patch_size * 1.05).astype(int)
        view2_crop_size = np.ceil(view2_patch_size * 1.05).astype(int)

        margin_x = 1
        margin_y = 1
        margin_z = 1
        rangex_min = margin_x
        rangex_max = standard_size[1] - margin_x
        rangey_min = margin_y
        rangey_max = standard_size[0] - margin_y
        rangez_min = margin_z
        rangez_max = standard_size[2] - margin_z

        y_interval = rangey_max - rangey_min - (view1_crop_size[0] + view2_crop_size[0])
        x_interval = rangex_max - rangex_min - (view1_crop_size[1] + view2_crop_size[1])
        z_interval = rangez_max - rangez_min - (view1_crop_size[2] + view2_crop_size[2])

        if y_interval >= 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_y = np.random.randint(20, min(view1_crop_size[0], view2_crop_size[0]) - 2)
            else:
                intersection_y = np.random.randint(-y_interval + 1, -y_interval + 10)
        else:
            intersection_y = np.random.randint(-y_interval + 1, min(view1_crop_size[0], view2_crop_size[0]) - 1)

        if x_interval >= 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_x = np.random.randint(20, min(view1_crop_size[1], view2_crop_size[1]) - 2)
            else:
                intersection_x = np.random.randint(-x_interval + 1, -x_interval + 10)
        else:
            intersection_x = np.random.randint(-x_interval + 5, min(view1_crop_size[1], view2_crop_size[1]) - 2)

        if z_interval > 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_z = np.random.randint(5, min(view1_crop_size[2], view2_crop_size[2]) - 2)
            else:
                intersection_z = np.random.randint(-z_interval + 1, -1)
        else:
            intersection_z = np.random.randint(-z_interval + 1, min(view1_crop_size[2], view2_crop_size[2]))

        overall_crop_size_y = (view1_crop_size[0] + view2_crop_size[0]) - intersection_y
        overall_crop_size_x = (view1_crop_size[1] + view2_crop_size[1]) - intersection_x
        overall_crop_size_z = (view1_crop_size[2] + view2_crop_size[2]) - intersection_z

        overall_x_min_left_up_corner = rangex_min
        overall_x_max_left_up_corner = rangex_max - overall_crop_size_x
        overall_y_min_left_up_corner = rangey_min
        overall_y_max_left_up_corner = rangey_max - overall_crop_size_y
        overall_z_min_left_up_corner = rangez_min
        overall_z_max_left_up_corner = rangez_max - overall_crop_size_z
        assert overall_x_min_left_up_corner <= overall_x_max_left_up_corner and \
               overall_y_min_left_up_corner <= overall_y_max_left_up_corner and \
               overall_z_min_left_up_corner <= overall_z_max_left_up_corner, \
            data['image_fn']

        overall_crop_left_up = (np.random.randint(overall_y_min_left_up_corner, overall_y_max_left_up_corner),
                                np.random.randint(overall_x_min_left_up_corner, overall_x_max_left_up_corner),
                                np.random.randint(overall_z_min_left_up_corner, overall_z_max_left_up_corner))
        overall_crop_left_up = np.asarray(overall_crop_left_up)

        view1_x_min = 0
        view1_x_max = overall_crop_size_x - view1_crop_size[1]
        view1_y_min = 0
        view1_y_max = overall_crop_size_y - view1_crop_size[0]
        view1_z_min = 0
        view1_z_max = overall_crop_size_z - view1_crop_size[2]

        assert view1_z_min < view1_z_max and view1_y_min < view1_y_max and view1_x_min < view1_x_max, data[
            'image_fn']

        view2_x_min = 0
        view2_x_max = overall_crop_size_x - view2_crop_size[1]
        view2_y_min = 0
        view2_y_max = overall_crop_size_y - view2_crop_size[0]
        view2_z_min = 0
        view2_z_max = overall_crop_size_z - view2_crop_size[2]

        assert view2_z_min < view2_z_max and view2_y_min < view2_y_max and view2_x_min < view2_x_max, data[
            'image_fn']

        view1_left_up = (np.random.randint(view1_y_min, view1_y_max),
                         np.random.randint(view1_x_min, view1_x_max),
                         np.random.randint(view1_z_min, view1_z_max))
        view1_left_up = np.asarray(view1_left_up)

        view2_left_up = (np.random.randint(view2_y_min, view2_y_max),
                         np.random.randint(view2_x_min, view2_x_max),
                         np.random.randint(view2_z_min, view2_z_max))
        view2_left_up = np.asarray(view2_left_up)
        view1_y1x1z1_crop = view1_left_up + overall_crop_left_up
        view1_y2x2z2_crop = view1_left_up + overall_crop_left_up + view1_crop_size
        view2_y1x1z1_crop = view2_left_up + overall_crop_left_up
        view2_y2x2z2_crop = view2_left_up + overall_crop_left_up + view2_crop_size

        crop_matrix = np.array([view1_y1x1z1_crop, view1_y2x2z2_crop, view2_y1x1z1_crop, view2_y2x2z2_crop])

        crop_matrix_origin = np.dot(crop_matrix, affine_standard_to_origin[:3, :3])
        crop_matrix_origin[::2, :] = np.floor(crop_matrix_origin[::2, :])
        crop_matrix_origin[1::2, :] = np.ceil(crop_matrix_origin[1::2, :])
        crop_matrix_origin = crop_matrix_origin.astype(int)
        view1_spacing = np.array(self.standard_spacing) * view1_scale
        view2_spacing = np.array(self.standard_spacing) * view2_scale
        spacing_matrix = np.array([view1_spacing, view2_spacing])

        data['crop_matrix_origin'] = crop_matrix_origin
        data['spacing_matrix'] = spacing_matrix
        return data


@PIPELINES.register_module()
class compute_crop_patch():
    def __init__(self, standard_spacing=(2., 2., 2.), patch_size=(96, 96, 96), aspect_ratio=[0.5, 1.5]):  # yxz
        self.standard_spacing = standard_spacing
        self.standard_patch_size = patch_size
        self.aspect_ratio = aspect_ratio

    def __call__(self, data):
        origin_spacing = data['subject'].image.spacing
        origin_size = data['subject'].image.shape
        standard_size = [np.floor(origin_spacing[0] / self.standard_spacing[0] * origin_size[1]).astype(int),
                         np.floor(origin_spacing[1] / self.standard_spacing[1] * origin_size[2]).astype(int),
                         np.floor(origin_spacing[2] / self.standard_spacing[2] * origin_size[3]).astype(int)]
        # print(standard_size)
        affine_standard_to_origin = np.array([self.standard_spacing[0] / origin_spacing[0], 0, 0, 0,
                                              0, self.standard_spacing[1] / origin_spacing[1], 0, 0,
                                              0, 0, self.standard_spacing[2] / origin_spacing[2], 0,
                                              0, 0, 0, 1]).reshape(4, 4).transpose()

        patchsize_y = self.standard_patch_size[0] * np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        patchsize_x = self.standard_patch_size[1] * np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        patchsize_z = self.standard_patch_size[0] * self.standard_patch_size[1] * self.standard_patch_size[
            2] / patchsize_y / patchsize_x
        patchsize_y = int(patchsize_y)
        patchsize_x = int(patchsize_x)
        patchsize_z = int(patchsize_z)
        patchsize_y = min(standard_size[0] - 12, patchsize_y)
        patchsize_x = min(standard_size[1] - 12, patchsize_x)
        patchsize_z = min(standard_size[2] - 12, patchsize_z)
        use_patch_size = (patchsize_y, patchsize_x, patchsize_z)

        left_up = np.array((np.random.randint(5, standard_size[0] - patchsize_y - 5),
                            np.random.randint(5, standard_size[1] - patchsize_x - 5),
                            np.random.randint(5, standard_size[2] - patchsize_z - 5)))
        right_bottom = left_up + np.array(use_patch_size)
        crop_matrix = np.array([left_up, right_bottom])
        crop_matrix_remap = np.dot(crop_matrix, affine_standard_to_origin[:3, :3])
        crop_matrix_remap[0, :] = np.floor(crop_matrix_remap[0, :]).astype('int')
        crop_matrix_remap[1, :] = np.ceil(crop_matrix_remap[1, :]).astype('int')
        data['crop_matrix_origin'] = crop_matrix_remap
        center = np.array((crop_matrix_remap[0, :] + crop_matrix_remap[1, :]) / 2).astype(int)
        data['subject']['labels_map'].data[0, center[0], center[1], center[2]] = 0
        return data


@PIPELINES.register_module()
class ComputeAugParamSAMV2():
    def __init__(self, scale=(0.8, 1.2), standard_spacing=(4., 4., 4.)):
        self.scale = scale
        self.standard_spacing = standard_spacing

    def __call__(self, data):
        view1_scale = np.random.uniform(self.scale[0], self.scale[1])
        view2_scale = np.random.uniform(self.scale[0], self.scale[1])
        # view1_scale = 0.8
        # view2_scale = 1.2
        view1_spacing = np.array(self.standard_spacing) * view1_scale
        view2_spacing = np.array(self.standard_spacing) * view2_scale
        spacing_matrix = np.array([view1_spacing, view2_spacing])
        data['spacing_matrix'] = spacing_matrix
        return data
