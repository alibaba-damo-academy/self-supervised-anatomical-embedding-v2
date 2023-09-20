# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import numpy as np
import os
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from interfaces import init, get_embedding, get_sim_embed_semantic_loc, normalize
from utils import read_image, visualize
from demo import get_random_query_point


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
config_file = 'configs/samv2/samv2_NIHLN.py'
checkpoint_file = 'checkpoints/SAMv2_iter_20000.pth'

# assume all input image are in torchio "LPS+" direction which equal to "RAI" orientation in ITK-Snap.
im1_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz'
im2_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz'


def main():
    time1 = time.time()
    model = init(config_file, checkpoint_file)
    time2 = time.time()
    print('model loading time:', time2 - time1)
    im1, normed_im1, norm_info_1 = read_image(im1_file, is_MRI=False)
    im2, normed_im2, norm_info_2 = read_image(im2_file, is_MRI=False)
    time3 = time.time()
    print('image loading time:', time3 - time2)

    pt1 = get_random_query_point(im1)

    emb1 = get_embedding(normed_im1, model)
    emb2 = get_embedding(normed_im2, model)
    time4 = time.time()
    print('embeddings computing time:', time4 - time3)

    pt1_normed = np.array(pt1) * norm_info_1
    pt2_normed, score = get_sim_embed_semantic_loc(emb1, emb2, pt1_normed,
                                                   (im2['shape'][3], im2['shape'][1], im2['shape'][2]),
                                                   norm_info=norm_info_2,
                                                   write_sim=False, use_sim_coarse=True)
    pt2 = np.array(pt2_normed).astype(int)
    print(pt2, score)
    time5 = time.time()
    print('matching point computing time:', time5 - time4)
    visualize(im1['img'], im2['img'], norm_info_1, norm_info_2, pt1, pt2, score)


if __name__ == '__main__':
    main()
