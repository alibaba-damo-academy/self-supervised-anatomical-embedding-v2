# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Written by Xiaoyu Bai

import numpy as np
import logging
import csv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import pickle


@DATASETS.register_module()
class Dataset3dsam(object):
    CLASSES = None

    def __init__(
            self, data_dir, index_file, annot=None, pipeline=None, test_mode=False, multisets=False, set_length=1000,
            mask_dir=None):

        self.data_path = data_dir
        self.with_mask = False
        if not mask_dir is None:
            self.with_mask = True
            self.mask_path = mask_dir
        self.loaddatalist(index_file)
        self.classes = ['lesion']
        self.num_classes = len(self.classes)
        self.num_niis = len(self.filename)
        self.multisets = multisets
        self.sample = False
        if self.multisets:
            self.set_length = set_length
        if not annot == None:
            self.loadannot(annot)
        self.pipline = pipeline
        self.test_mode = test_mode
        self.pipline = Compose(pipeline)
        self._set_group_flag()
        self.logger = logging.getLogger(__name__)

    def __len__(self):
        if self.multisets:
            return self.set_length
        else:
            return self.num_niis

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def pre_pipeline(self, results):
        return results

    def __getitem__(self, index):
        if self.multisets:
            loc = np.random.randint(0, self.num_niis)
        else:
            loc = index
        image_fn = self.filename[loc]

        if self.test_mode:
            data = self.prepare_test_img(self.data_path, image_fn)
            return data
        while True:
            if self.sample:
                anno = self.annot[image_fn.split('.', 1)[0]]
                data = self.prepare_train_img(self.data_path, image_fn, anno)
            if self.with_mask:
                data = self.prepare_train_img(self.data_path, image_fn, mask_path = self.mask_path)
            else:
                data = self.prepare_train_img(self.data_path, image_fn)
            return data

    def prepare_train_img(self, data_path, image_fn, anno=None,mask_path=None):
        if anno is not None:
            data = {'data_path': data_path,
                    'image_fn': image_fn,
                    'anno': anno}
        if mask_path is not None:
            data = {'data_path': data_path,
                    'mask_path': mask_path,
                    'image_fn': image_fn}
        else:
            data = {'data_path': data_path,
                    'image_fn': image_fn}
        data = self.pipline(data)
        return data

    def prepare_test_img(self, data_path, image_fn):
        data = {'data_path': data_path,
                'image_fn': image_fn}
        data = self.pipline(data)
        return data

    def loaddatalist(self, path):
        info = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                info.append(row)
        info = info[1:]
        self.filename = np.array([row[0] for row in info])

    def loadannot(self, annot):
        f = open(annot, 'rb')
        self.annot = pickle.load(f)
        self.sample = True
