import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import numpy as np
import os
import random
import torch.utils.data as data
import numpy as np
# from utils import process_feat
import torch
from torch.utils.data import DataLoader
# torch.set_default_tensor_type('torch.FloatTensor')


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', feature_dim=512, args=None):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.feature_name = args.feature_name
        self.feature_dim = feature_dim

        split_tmp = "train" if is_train else "test"
        mode_tmp = "normal"

        self.caption_root_path = rf"/workspace/datasets/ucf-crime/uio/captions/train/normal"
        self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/uio/caption_embeddings/{split_tmp}/{mode_tmp}"
        self.path = self.get_dataset_root_path(path, split_tmp, mode_tmp)

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

    def get_dataset_root_path(self, path, split_, mode_):
        if self.feature_name in ["i3d", "clip", "i3d_clip"]:
            return path
        elif self.feature_name == "uio_opt_region":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
        elif self.feature_name == "uio_fixed_region":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
        # elif self.feature_name == "uio_caption_vqa1":
        #     if self.is_train:  # 可能需要重建确认一下数据的形式  input_dim:2048
        #         root_path = "/workspace/datasets/ucf-crime/uio/32clip"
        #     else:
        #         root_path = "/workspace/datasets/ucf-crime/uio/24b"
        elif self.feature_name in ["uio_caption_vqa1_68", "uio_caption_34", "uio_vqa1_34"]:
            root_path = "/workspace/datasets/ucf-crime/uio/sorted"
        elif self.feature_name == "vqas_170":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
        elif self.feature_name in ["uio_caption", "uio_vqa1", "uio_caption_vqa1"]:  # 2048
            root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
        elif self.feature_name == "uio_caption_vqa1_1280":  # 640
            root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
        path = os.path.join(root_path, split_, mode_)

        return path

    def __len__(self):
        return len(self.data_list)

    def get_semantic_embedding(self, name):
        name = name.split("/")[-1].replace(".mp4", "")
        embedding = np.load(os.path.join(self.caption_embedding_root_path, f"{name}.npy"), allow_pickle=True)
        return embedding

    def get_feature(self, name, normal_mode=False):
        if self.feature_name == "i3d":
            rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path, 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        elif self.feature_name == "clip":
            feature = get_clip_feature(name, not self.is_train, normal_mode, mix=not self.is_train)
            return feature
        elif self.feature_name == "i3d_clip":
            feature = get_clip_feature(name, not self.is_train, normal_mode, mix=not self.is_train)
            rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path, 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
            return concat_npy
        else:
            video_id = name.split("/")[-1].split(".")[0]
            data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
            if self.feature_name == "uio_caption":
                data = data[:, :1024]
            elif self.feature_name == "uio_vqa1":
                data = data[:, 1024:]
            elif self.feature_name == "uio_caption_34":
                data = data[:, :34]
            elif self.feature_name == "uio_vqa1_34":
                data = data[:, 34:]
            return data

    def __getitem__(self, idx):
        if self.is_train == 1:
            name = self.data_list[idx][:-1]
            feature = self.get_feature(name, normal_mode=True)
            embedding = self.get_semantic_embedding(name)
            if "uio" in self.feature_name:
                return feature, idx, embedding
            else:
                return feature, embedding
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(
                self.data_list[idx].split(' ')[2][:-1])
            feature = self.get_feature(name, normal_mode=True)
            embedding = self.get_semantic_embedding(name)
            # feature = np.array([i for i in feature if (i[-1]!=0).sum() != 0])
            feature = feature[:len(embedding)]
            return feature, gts, frames, name, embedding

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
        if "uio" in self.feature_name:
            features = features[0]

        embedding = self.get_semantic_embedding(video_name)[idx]
        feature = features[idx]
        return feature, caption,embedding

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
        self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/uio/caption_embeddings/{split_tmp}/{mode_tmp}"
        self.path = self.get_dataset_root_path(path, split_tmp, mode_tmp)

        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def get_semantic_embedding(self, name):
        name = name.split("/")[-1].replace(".mp4", "")
        embedding = np.load(os.path.join(self.caption_embedding_root_path, f"{name}.npy"),allow_pickle=True)
        return embedding

    def get_dataset_root_path(self, path, split_, mode_):
        if self.feature_name in ["i3d", "clip", "i3d_clip"]:
            return path
        elif self.feature_name == "uio_opt_region":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
        elif self.feature_name == "uio_fixed_region":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
        # elif self.feature_name == "uio_caption_vqa1":
        #     if self.is_train:  # 可能需要重建确认一下数据的形式  input_dim:2048
        #         root_path = "/workspace/datasets/ucf-crime/uio/32clip"
        #     else:
        #         root_path = "/workspace/datasets/ucf-crime/uio/24b"
        elif self.feature_name in ["uio_caption_vqa1_68", "uio_caption_34", "uio_vqa1_34"]:
            root_path = "/workspace/datasets/ucf-crime/uio/sorted"
        elif self.feature_name == "vqas_170":
            root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
        elif self.feature_name in ["uio_caption", "uio_vqa1", "uio_caption_vqa1"]:  # 2048
            root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
        elif self.feature_name == "uio_caption_vqa1_1280":  # 640
            root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
        path = os.path.join(root_path, split_, mode_)
        return path

    def __len__(self):
        return len(self.data_list)

    def get_feature(self, name, normal_mode=False):
        if self.feature_name == "i3d":
            rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path, 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        elif self.feature_name == "clip":
            feature = get_clip_feature(name, not self.is_train, normal_mode, mix=not self.is_train)
            return feature
        elif self.feature_name == "i3d_clip":
            feature = get_clip_feature(name, not self.is_train, normal_mode, mix=not self.is_train)
            rgb_npy = np.load(os.path.join(self.path, 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path, 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([feature, rgb_npy, flow_npy], axis=1)
            return concat_npy
        else:
            video_id = name.split("/")[-1].split(".")[0]
            data = np.load(os.path.join(self.path, f'{video_id}.npy'), allow_pickle=True)
            if self.feature_name == "uio_caption":
                data = data[:, :1024]
            elif self.feature_name == "uio_vqa1":
                data = data[:, 1024:]
            elif self.feature_name == "uio_caption_34":
                data = data[:, :34]
            elif self.feature_name == "uio_vqa1_34":
                data = data[:, 34:]
            return data

    def __getitem__(self, idx):
        normal_mode = False
        test_mode = not self.is_train
        if self.is_train == 1:
            name = self.data_list[idx][:-1]

            feature = self.get_feature(name)
            embedding = self.get_semantic_embedding(name)

            if "uio" in self.feature_name:
                return feature, idx, embedding
            else:
                return feature, embedding
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), \
                self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            embedding = self.get_semantic_embedding(name)
            # name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(
            #     self.data_list[idx].split(' ')[2][:-1])
            feature = self.get_feature(name)
            # feature = np.array([i for i in feature if (i[-1]!=0).sum() != 0])
            feature = feature[:len(embedding)]
            return feature, gts, frames, name, embedding

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
        if "uio" in self.feature_name:
            features = features[0]
        feature = features[idx]
        embedding = self.get_semantic_embedding(video_name)[idx]

        return feature, caption, embedding


    def check_captions_from_snippet_idxs(self, snippet_idxs):
        idxs = [int(item.split('-')[0]) for item in snippet_idxs]
        idxs = np.argsort(idxs)
        for idx in idxs:
            snippet_idx = snippet_idxs[idx]
            feature, caption, embedding = self.get_snippet_feature(snippet_idx)
            print(f"{snippet_idx} : \t{caption}")

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
    """由于preprocess的方式不同，一部的数据需要将数据转成numpy才能使用"""
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

class ShanghaiTechDataset():
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


if __name__ == '__main__':
    # loader2 = Normal_Loader(is_train=0)
    loader2 = Anomaly_Loader(is_train=0)
    for i in range(10):
        print(loader2.__getitem__(i)[1])
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # print(len(loader2))
    # print(loader[1], loader2[1])

# 85.306
