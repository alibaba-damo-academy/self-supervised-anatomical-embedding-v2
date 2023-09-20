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
# import ipdb


@PIPELINES.register_module()
class DynamicSpacing:
    def __init__(self, norm_spacing=(2., 2., 2.), scale=(0.8, 1.2)):
        self.norm_spacing = norm_spacing
        self.scale = scale
    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        data['resample_spacing'] = np.array(self.norm_spacing) * scale
        return data


@PIPELINES.register_module()
class SeperateCTMRIImage:
    def __call__(self, data):
        mask_path = data['mask_path']
        if data['tag'] == 'im1':
            data = data['data1']
        elif data['tag'] == 'im2':
            data = data['data2']
        else:
            return NotImplementedError
        data['mask_path'] = mask_path
        return data

@PIPELINES.register_module()
class AddSuffixByProb:
    def __init__(self, suffix: [str], p: [float]):
        self.suffix = suffix
        self.p = np.array(p)
        self.cum_p = np.cumsum(p)

    def __call__(self, data):
        porb = np.random.random()
        ind = np.searchsorted(self.cum_p, porb)
        data['image_fn'] = data['image_fn'] + self.suffix[ind]
        if 'mod' not in data.keys():
            if '0000' in self.suffix[ind]:
                data['mod'] = 'ct'
            elif '0001' in  self.suffix[ind] or '0002' in  self.suffix[ind]:
                data['mod'] = 'mri'
        return data


@PIPELINES.register_module()
class stoppoint:
    def __call__(self, data):
        print('stoppoint')
        return data
