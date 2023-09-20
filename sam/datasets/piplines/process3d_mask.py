# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Written by Xiaoyu Bai

from mmdet.datasets.builder import PIPELINES
from .process3d import meshgrid3d, makelabel
import torchio as tio
import numpy as np
from mmdet.datasets.pipelines import Compose
import copy
import torch
import os
import nibabel as nib
from skimage import measure
from scipy import ndimage


@PIPELINES.register_module()
class GeneratePairedImagesMaskInfo:
    def __call__(self, data):
        anchor = 1  # select anchor organ for crop
        data1 = {'data_path': data['data_path'], 'image_fn': data['image_fn1']}
        data2 = {'data_path': data['data_path'], 'image_fn': data['image_fn2']}
        data['paired'] = [data['image_fn1'], data['image_fn2']]
        data['data1'] = data1
        data['data2'] = data2
        data['anchor'] = anchor
        return data


@PIPELINES.register_module()
class GenerateCTMRIPairInfo:
    def __call__(self, data):
        data1 = {'data_path': data['data_path'], 'image_fn': data['image_fn']}
        data2 = {'data_path': data['data_path'], 'image_fn': data['image_fn']}
        data['data1'] = data1
        data['data2'] = data2
        return data



@PIPELINES.register_module()
class AddSuffix:
    def __init__(self, suffix: str):
        self.suffix = suffix

    def __call__(self, data):
        data['image_fn'] = data['image_fn'] + self.suffix
        return data


@PIPELINES.register_module()
class LoadTioImageWithMask:
    """Load image as torchio subject.
    """

    def __init__(self, re_orient=True, with_mesh=False):
        self.re_orient = re_orient
        self.with_mesh = with_mesh

    def __call__(self, data):
        # time1 = time.time()
        data_path = data['data_path']
        image_fn = data['image_fn']
        if 'mask_path' in data.keys():
            mask_path = data['mask_path']
            mask_tio = tio.LabelMap(os.path.join(mask_path, image_fn))

        img_tio = tio.ScalarImage(os.path.join(data_path, image_fn))

        # print(img_tio.orientation,image_fn)

        # for NIH lymph node dataset covert orientation to PLS+
        if self.re_orient:
            img_data = img_tio.data
            img_tio.data = img_data.permute(0, 2, 1, 3)
            img_tio.affine = np.array(
                [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
            if 'mask_path' in data.keys():
                mask_data = mask_tio.data
                mask_tio.data = mask_data.permute(0, 2, 1, 3)
                mask_tio.affine = np.array(
                    [mask_tio.affine[1, :], mask_tio.affine[0, :], mask_tio.affine[2, :], mask_tio.affine[3, :]])
        img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
        img_tio_spacing = img_tio.spacing
        if self.with_mesh:
            meshs = meshgrid3d(img_tio_shape, img_tio_spacing) #3 channels y x z
            labels = makelabel(img_tio_shape)
        if 'mask_path' in data.keys():
            if self.with_mesh:
                subject = tio.Subject(
                    image=img_tio,
                    mask=mask_tio,
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
                    mask=mask_tio)

        else:
            if self.with_mesh:
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
                    image=img_tio
                )

        data['subject'] = subject
        return data
