import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', feature_dim=512, args=None):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.feature_name = args.feature_name

        split_tmp = "train" if is_train else "test"
        mode_tmp = "normal"

        self.caption_root_path = rf"/workspace/datasets/ucf-crime/uio/captions/train/normal"
        if self.feature_name == "i3d":
            self.path = path
        elif self.feature_name == "uio":
            # root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
            root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
            self.path = os.path.join(root_path, split_tmp, mode_tmp)
        else:
            if feature_dim in [512, 2048, 2560]:
                self.path = path
            elif feature_dim == 2049:
                if is_train:
                    root_path = "/workspace/datasets/ucf-crime/uio/32clip"
                else:
                    root_path = "/workspace/datasets/ucf-crime/uio/24b"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted"
                # root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
                self.path = os.path.join(root_path, split_tmp, mode_tmp)
            elif feature_dim == 1280:
                root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
                self.path = os.path.join(root_path, split_tmp, mode_tmp)

        self.feature_dim = feature_dim
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        # random.shuffle(self.data_list)
        # self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        normal_mode = True
        test_mode = not self.is_train
        if self.is_train == 1:
            name = self.data_list[idx][:-1]
            # if self.feature_name == "uio":
            #     pass
            # if self.feature_name =="i3d"：
            if self.feature_dim == 512:  # 使用clip的特征量
                feature = get_clip_feature(name, test_mode, normal_mode)
                return feature
            elif self.feature_dim == 2048:  # 使用I3D的特征量
                rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                if self.modality == 'RGB':
                    return rgb_npy
                elif self.modality == 'FLOW':
                    return flow_npy
                else:
                    return concat_npy
            elif self.feature_dim == 2560:  # 使用clip + I3D的特征量
                feature = get_clip_feature(name, test_mode, normal_mode)
                rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
                flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
                concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
                return concat_npy
            else:  # 其他的特征量，比如从uio中提取出来的特征量
                video_id = name.split("/")[-1].split(".")[0]
                data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
                if self.feature_name == "uio-caption":
                    data = data[:, :1024]
                elif self.feature_name == "uio-vqa1":
                    data = data[:, 1024:]
                return data, idx
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
            else:
                video_id = name.split("/")[-1].split(".")[0]
                data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
                if self.feature_name == "uio-caption":
                    data = data[:, :1024]
                elif self.feature_name == "uio-vqa1":
                    data = data[:, 1024:]
                return data, gts, frames, name

    def show_caption(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_name = self.data_list[int(video_id)][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")
        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        return data["orginal"][int(idx)]

    def get_snippet_feature(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_id, idx = int(video_id), int(idx)
        video_name = self.data_list[video_id][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")

        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        caption = data["orginal"][idx]

        features = self.__getitem__(video_id)
        if self.feature_name == "uio":
            features = features[0]
        feature = features[idx]
        return feature, caption

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', feature_dim=2048, args=None):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.feature_name = args.feature_name
        self.feature_dim = feature_dim

        split_tmp = "train" if is_train else "test"
        mode_tmp = "anomaly"

        self.caption_root_path = rf"/workspace/datasets/ucf-crime/uio/captions/train/anomaly"

        if self.feature_name == "uio":
            # root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
            root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
            self.path = os.path.join(root_path, split_tmp, mode_tmp)
        else:
            if feature_dim in [512, 2048, 2560]:
                self.path = path
            elif feature_dim == 1280:
                root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
                self.path = os.path.join(root_path, split_tmp, mode_tmp)
            else:
                if is_train:
                    root_path = "/workspace/datasets/ucf-crime/uio/so"
                else:
                    root_path = "/workspace/datasets/ucf-crime/uio/24b"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
                root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
                self.path = os.path.join(root_path, split_tmp, mode_tmp)

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
                video_id = name.split("/")[-1].split(".")[0]
                data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
                if self.feature_name == "uio-caption":
                    data = data[:, :1024]
                elif self.feature_name == "uio-vqa1":
                    data = data[:, 1024:]
                return data, idx
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), \
                self.data_list[idx].split('|')[2][1:-2].split(',')
            # if "Explosion021_x264" in name:
            #     print(1)
            #     print(gts)

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
            else:
                video_id = name.split("/")[-1].split(".")[0]
                data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
                if self.feature_name == "uio-caption":
                    data = data[:, :1024]
                elif self.feature_name == "uio-vqa1":
                    data = data[:, 1024:]
                return data, gts, frames, name

    def show_caption(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_name = self.data_list[int(video_id)][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")
        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        return data["orginal"][int(idx)]

    def get_snippet_feature(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_id, idx = int(video_id), int(idx)
        video_name = self.data_list[video_id][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")

        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        caption = data["orginal"][idx]

        features = self.__getitem__(video_id)
        if self.feature_name == "uio":
            features = features[0]
        feature = features[idx]
        return feature, caption
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

# 85.306
