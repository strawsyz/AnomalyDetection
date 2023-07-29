# -*- coding: utf-8 -*-
# File  : generate_gt.py
# Author: strawsyz
# Date  : 2023/7/12


from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt
import glob
from sklearn.metrics import roc_auc_score, roc_curve
import time
from tqdm import tqdm


# ver = 'wsal/'
# modality = "rgb"
# gpu_id = 0
#
# model_paths = "./weights/" + ver
# output_folder = "./results/" + ver
# videos_pkl_train = "/test/UCF-Crime/UCF/Anomaly_Detection_splits/variables.txt"
# hdf5_path = "/test/UCF-Crime/UCF/gcn_test.hdf5"
# data_root = '/pcalab/tmp/UCF-Crime/UCF_Crimes/Anomaly_train_test_imgs/test/'
#


def get_gts(LABEL_PATH):
    video_path_list = []
    videos = {}
    video_gts = np.array([])
    for line in open(LABEL_PATH):
        video_path_list.append(line)

    for video in video_path_list:
        root_path = r"/workspace/MGFN./UCF_Test_ten_i3d/"
        filepath = os.path.join(root_path, video.split(' ')[0][:-4] + "_i3d.npy")

        features = np.load(filepath, allow_pickle=True)
        print(features.shape)
        num_frames = features.shape[0]
        # data_root = r"/workspace/datasets/ucf-crime/frames/test"
        # if video.split(' ')[0][:-4].startswith("Normal"):
        #     data_root = os.path.join(data_root, "normal")
        # else:
        #     data_root = os.path.join(data_root, "anomaly")
        # data_dir = os.path.join(data_root, video.split(' ')[0][:-4])
        # img_list = glob.glob(os.path.join(data_dir, '*.jpg'))
        start_1 = int(video.split(' ')[4])
        end_1 = int(video.split(' ')[6])
        start_2 = int(video.split(' ')[8])
        end_2 = int(video.split(' ')[10])
        sub_video_gt = np.zeros((num_frames,), dtype=np.int8)  # 每个视频的gt
        sub_video_gt = np.repeat(np.array(sub_video_gt), 16)
        if start_1 >= 0 and end_1 >= 0:
            sub_video_gt[start_1 - 1:end_1] = 1
        if start_2 >= 0 and end_2 >= 0:
            sub_video_gt[start_2 - 1:end_2] = 1
        videos[video.split(' ')[0][:-4]] = sub_video_gt
        video_gts = np.concatenate((video_gts, sub_video_gt))
    return video_gts


if __name__ == '__main__':
    UCFdata_LABEL_PATH = '/workspace/datasets/ucf-crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
    ucf_gt_save_path = '/workspace/datasets/ucf-crime/test_gt'
    # data_root = '/pcalab/tmp/UCF-Crime/UCF_Crimes/Anomaly_train_test_imgs/test/'
    results = get_gts(UCFdata_LABEL_PATH)
    np.save(ucf_gt_save_path, results)CUDA_VISIBLE_DEVICES=5
    # print(results)
    print(results.shape)
