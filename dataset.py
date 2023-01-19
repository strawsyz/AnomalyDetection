import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', feature_dim=512):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.feature_dim = feature_dim
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        normal_mode = True
        test_mode = not self.is_train
        if self.is_train == 1:
            name = self.data_list[idx][:-1]
            if self.feature_dim == 512:
                feature = get_clip_feature(name, test_mode, normal_mode)
                return feature
            elif self.feature_dim == 2048:
                rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                if self.modality == 'RGB':
                    return rgb_npy
                elif self.modality == 'FLOW':
                    return flow_npy
                else:
                    return concat_npy
            elif self.feature_dim == 2560:
                feature = get_clip_feature(name, test_mode, normal_mode)
                rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(
                self.data_list[idx].split(' ')[2][:-1])

            if self.feature_dim == 512:
                feature = get_clip_feature(name, test_mode, normal_mode, mix=True)
                return feature, gts, frames, name
            elif self.feature_dim == 2048:
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                if self.modality == 'RGB':
                    return rgb_npy, gts, frames, name
                elif self.modality == 'FLOW':
                    return flow_npy, gts, frames, name
                else:
                    return concat_npy, gts, frames, name
            elif self.feature_dim == 2560:
                feature = get_clip_feature(name, test_mode, normal_mode, mix=True)
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames, name


class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', feature_dim=2048):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.feature_dim = feature_dim
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        normal_mode = False
        test_mode = not self.is_train
        if self.is_train == 1:
            name = self.data_list[idx][:-1]
            if self.feature_dim == 512:
                feature = get_clip_feature(name, test_mode, normal_mode)
                return feature
            elif self.feature_dim == 2048:
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                if self.modality == 'RGB':
                    return rgb_npy
                elif self.modality == 'FLOW':
                    return flow_npy
                else:
                    return concat_npy
            elif self.feature_dim == 2560:
                feature = get_clip_feature(name, test_mode, normal_mode)
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), \
                self.data_list[idx].split('|')[2][1:-2].split(',')
            if "Explosion021_x264" in name:
                print(1)
                print(gts)
            gts = [int(i) for i in gts]
            if self.feature_dim == 512:
                feature = get_clip_feature(name, test_mode, normal_mode, mix=True)
                return feature, gts, frames, name
            elif self.feature_dim == 2048:
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                if self.modality == 'RGB':
                    return rgb_npy, gts, frames, name
                elif self.modality == 'FLOW':
                    return flow_npy, gts, frames, name
                else:
                    return concat_npy, gts, frames, name
            elif self.feature_dim == 2560:
                feature = get_clip_feature(name, test_mode, normal_mode, mix=True)
                rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
                return concat_npy, gts, frames, name


def get_feature_filepath(video_id, test_mode, normal_mode, mix=False):
    root_path = "/workspace/datasets/ucf-crime/CLIP/image"
    if test_mode:
        # split = "test"
        if mix:
            split = "test"
        else:
            split = "test_32s"
    else:
        split = "train"
    root_path = os.path.join(root_path, split,
                             "normal" if normal_mode else "anomaly")
    feature_path = os.path.join(root_path, video_id + ".npy")
    return feature_path


def get_clip_feature(video_id, test_mode, normal_mode, mix=False):
    video_id = video_id.split("/")[-1].replace(".mp4", "")
    feature_path = get_feature_filepath(video_id, test_mode, normal_mode, mix)
    # print(os.path.exists(feature_path))
    feature = np.load(feature_path, allow_pickle=True)
    if test_mode:
        if mix:
            feature = np.squeeze(np.array([feature[i].cpu().detach().numpy() for i in range(len(feature))]))
        else:
            feature = np.squeeze(np.array([feature[i] for i in range(len(feature))]))
    else:
        pass
        # feature = np.squeeze(np.array([feature[i].cpu().detach().numpy() for i in range(len(feature))]))
    return feature


if __name__ == '__main__':
    # loader2 = Normal_Loader(is_train=0)
    loader2 = Anomaly_Loader(is_train=0)
    for i in range(10):
        print(loader2.__getitem__(i)[1])
    # print(len(loader2))
    # print(loader[1], loader2[1])
