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

        self.caption_type = "swinbert"
        if self.caption_type == "uio":
            self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/uio/caption_embeddings/{split_tmp}/{mode_tmp}"
            self.caption_root_path = rf"/workspace/datasets/ucf-crime/uio/captions/train/normal"
        elif self.caption_type == "swinbert":
            self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/swinbert/caption_embeddings/{split_tmp}/{mode_tmp}"
            self.caption_root_path = rf"/workspace/datasets/ucf-crime/swinbert/captions/train/normal"

        self.path = get_dataset_root_path(self.feature_name, path, split_tmp, mode_tmp)

        if self.is_train == 1:
            if self.feature_name == "new_i3d":
                pass
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
        elif self.feature_name == "new_i3d":
            name = name.replace(".mp4","")
            name = name.replace("Normal_Videos_event/","")
            name = name.replace("Normal_Videos","Normal_Videos_")
            features_filepath = os.path.join(self.path, name + '_i3d.npy')
            features = np.load(features_filepath, allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            return features
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

    def load_i3d_vf(self, npy_filepath):
        features = np.load(npy_filepath)
        features = features.transpose(1, 0, 2)  # [10, T, F]
        divided_features = []
        divided_mag = []
        for feature in features:
            feature = process_feat(feature, 32)  # ucf(32,2048)
            divided_features.append(feature)
            divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
        divided_features = np.array(divided_features, dtype=np.float32)
        divided_mag = np.array(divided_mag, dtype=np.float32)
        divided_features = np.concatenate((divided_features, divided_mag), axis=2)
        return divided_features

    def __getitem__(self, idx):
        if self.is_train == 1:
            name = self.data_list[idx][:-1]
            feature = self.get_feature(name, normal_mode=True)
            if self.feature_name == "new_i3d":
                feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                feature = np.concatenate((feature, feature_mag), axis=2)
            embedding = self.get_semantic_embedding(name)
            if len(embedding) > len(feature):
                embedding = embedding[:len(feature)]
            if "uio" in self.feature_name:
                return feature, idx, embedding
            else:
                return feature, idx, embedding
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(
                self.data_list[idx].split(' ')[2][:-1])
            feature = self.get_feature(name, normal_mode=True)
            embedding = self.get_semantic_embedding(name)
            # feature = np.array([i for i in feature if (i[-1]!=0).sum() != 0])
            if self.caption_type == "uio" and self.feature_name == "i3d":
                num_captions = embedding.shape[0]
                gap_between_captions = int(num_captions / len(feature))
                start_caption = int(gap_between_captions / 2)
                idx = [int(i) for i in
                       np.linspace(start_caption, len(embedding) - 1 - start_caption, num=len(feature), endpoint=True,
                                   retstep=False, dtype=None)]
                embedding = embedding[idx]
            elif self.caption_type == "swinbert":
                if len(embedding) > len(feature):
                    embedding = embedding[:len(feature)]

            feature = feature[:len(embedding)]
            return feature, gts, frames, name, embedding

    def show_caption(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_name = self.data_list[int(video_id)][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")
        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        if self.caption_type == "uio":
            return data["orginal"][int(idx)]
        elif self.caption_type == "swinbert":
            video_name = f"{video_name}-{idx}.avi"
            return data[video_name][0][0]

    def compare_captions(self, snippet_ids):
        embeddings = []
        for snippet_id in snippet_ids:
            video_id, idx = snippet_id.split("-")
            video_name = self.data_list[int(video_id)][:-1]
            video_name = video_name.split("/")[-1].replace(".mp4", "")
            # caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
            # data = np.load(caption_filepath, allow_pickle=True).item()
            embedding = self.get_semantic_embedding(video_name)[idx]
            embeddings.append(embedding)
        sims = []
        for embedding in embeddings:
            sim = torch.cosine_similarity(embedding, embeddings)
            sims.append(sim)
        print(sims)

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
        return feature, caption, embedding

def get_dataset_root_path(feature_name, path, split_, mode_):
    if feature_name == "new_i3d":
        if split_ =="test":
            path =r"/workspace/MGFN./UCF_Train_ten_i3d"
        elif split_ =="train":
            path = r"/workspace/MGFN./UCF_Test_ten_i3d"
        return path

    if feature_name in ["i3d", "clip", "i3d_clip"]:
        return path
    elif feature_name == "uio_opt_region":
        root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
    elif feature_name == "uio_fixed_region":
        root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
    # elif self.feature_name == "uio_caption_vqa1":
    #     if self.is_train:  # 可能需要重建确认一下数据的形式  input_dim:2048
    #         root_path = "/workspace/datasets/ucf-crime/uio/32clip"
    #     else:
    #         root_path = "/workspace/datasets/ucf-crime/uio/24b"
    elif feature_name in ["uio_caption_vqa1_68", "uio_caption_34", "uio_vqa1_34"]:
        root_path = "/workspace/datasets/ucf-crime/uio/sorted"
    elif feature_name == "vqas_170":
        root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
    elif feature_name in ["uio_caption", "uio_vqa1", "uio_caption_vqa1"]:  # 2048
        root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
    elif feature_name == "uio_caption_vqa1_1280":  # 640
        root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
    path = os.path.join(root_path, split_, mode_)

    return path


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

        self.caption_type = "swinbert"
        if self.caption_type == "uio":
            self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/uio/caption_embeddings/{split_tmp}/{mode_tmp}"
            self.caption_root_path = rf"/workspace/datasets/ucf-crime/uio/captions/train/anomaly"
        elif self.caption_type == "swinbert":
            self.caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/swinbert/caption_embeddings/{split_tmp}/{mode_tmp}"
            self.caption_root_path = rf"/workspace/datasets/ucf-crime/swinbert/captions/train/anomaly"

        self.path = get_dataset_root_path(self.feature_name, path, split_tmp, mode_tmp)

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
        embedding = np.load(os.path.join(self.caption_embedding_root_path, f"{name}.npy"), allow_pickle=True)
        return embedding

    # def get_dataset_root_path(self, path, split_, mode_):
    #     if self.feature_name in ["i3d", "clip", "i3d_clip"]:
    #         return path
    #     elif self.feature_name == "uio_opt_region":
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted5"
    #     elif self.feature_name == "uio_fixed_region":
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted6"
    #     # elif self.feature_name == "uio_caption_vqa1":
    #     #     if self.is_train:  # 可能需要重建确认一下数据的形式  input_dim:2048
    #     #         root_path = "/workspace/datasets/ucf-crime/uio/32clip"
    #     #     else:
    #     #         root_path = "/workspace/datasets/ucf-crime/uio/24b"
    #     elif self.feature_name in ["uio_caption_vqa1_68", "uio_caption_34", "uio_vqa1_34"]:
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted"
    #     elif self.feature_name == "vqas_170":
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted2"
    #     elif self.feature_name in ["uio_caption", "uio_vqa1", "uio_caption_vqa1"]:  # 2048
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted3"
    #     elif self.feature_name == "uio_caption_vqa1_1280":  # 640
    #         root_path = "/workspace/datasets/ucf-crime/uio/sorted4"
    #     path = os.path.join(root_path, split_, mode_)
    #     return path

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
        elif self.feature_name == "new_i3d":
            name = name.replace(".mp4", "")
            name = name.replace("Normal_Videos_event/","")
            features_filepath = os.path.join(self.path, name + '.npy')
            features = np.load(features_filepath, allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            return features
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
            if self.feature_name == "new_i3d":
                feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                feature = np.concatenate((feature, feature_mag), axis=2)
            embedding = self.get_semantic_embedding(name)
            if len(embedding) > len(feature):
                embedding = embedding[:len(feature)]
            if "uio" in self.feature_name:
                return feature, idx, embedding
            else:
                return feature, idx, embedding
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), \
                self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            embedding = self.get_semantic_embedding(name)
            # name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(
            #     self.data_list[idx].split(' ')[2][:-1])
            feature = self.get_feature(name)
            # feature = np.array([i for i in feature if (i[-1]!=0).sum() != 0])

            # num_captions = embedding.shape[0]
            # gap_between_captions = int(num_captions / len(feature))
            # start_caption = int(gap_between_captions / 2)
            # idx = [int(i) for i in
            #        np.linspace(start_caption, len(embedding) - 1 - start_caption, num=len(feature), endpoint=True,
            #                    retstep=False, dtype=None)]
            # embedding = embedding[idx]

            if len(embedding) > len(feature):
                embedding = embedding[:len(feature)]

            feature = feature[:len(embedding)]
            return feature, gts, frames, name, embedding

    def show_caption(self, snippet_id):
        video_id, idx = snippet_id.split("-")
        video_name = self.data_list[int(video_id)][:-1]
        video_name = video_name.split("/")[-1].replace(".mp4", "")
        caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        data = np.load(caption_filepath, allow_pickle=True).item()
        if self.caption_type == "uio":
            return video_name, data["orginal"][int(idx)]
        elif self.caption_type == "swinbert":
            video_name = f"{video_name}-{idx}.avi"
            return video_name, data[video_name][0][0]

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

        self.transform = transform
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

        if self.transform is not None:
            features = self.transform(features)
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


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)  # UCF(32,2048)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int)  # (33,)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


def show_video_captions():
    """显示一个视频内所有的caption"""
    video_name = "Arrest049_x264"
    video_name = "Burglary043_x264"
    video_name = "Burglary078_x264"
    video_name = "Shoplifting040_x264"
    video_name = "RoadAccidents047_x264"
    caption_root_path = r"/workspace/datasets/ucf-crime/swinbert/captions/train/anomaly"
    caption_filepath = os.path.join(caption_root_path, video_name + ".npy")
    data = np.load(caption_filepath, allow_pickle=True).item()
    # print(data)
    for idx in range(32):
        snippet_name = f"{video_name}-{idx}.avi"
        print(f"{data[snippet_name][0][0]}\t{data[snippet_name][0][1]}")


def show_caption_confidence():
    caption_root_path = r"/workspace/datasets/ucf-crime/swinbert/captions/train/anomaly"
    # 检查记忆的可行度，来筛选应该保留的记忆

    # video_names = ["Arrest047_x264-1.avi","RoadAccidents051_x264-14.avi","RoadAccidents041_x264-9.avi","Abuse041_x264-24.avi","Vandalism008_x264-8.avi","Robbery104_x264-2.avi","RoadAccidents115_x264-26.avi"]
    video_names = ["RoadAccidents047_x264-15.avi", "Robbery128_x264-4.avi", "Shoplifting040_x264-13.avi",
                   "Burglary078_x264-14.avi", "Stealing071_x264-22.avi", "Burglary083_x264-4.avi",
                   "Robbery144_x264-29.avi", "Arrest049_x264-18.avi", "Burglary043_x264-19.avi", "Arson052_x264-21.avi"]

    ids = [int(video_name.replace(".avi", "").split("-")[1]) for video_name in video_names]
    video_names = [video_name.replace(".avi", "").split("-")[0] for video_name in video_names]

    for video_name, idx in zip(video_names, ids):
        video_name_raw = f"{video_name}-{idx}.avi"
        confidence = \
        np.load(os.path.join(caption_root_path, f"{video_name}.npy"), allow_pickle=True).item()[video_name_raw][0][1]
        print(f"{video_name}: \t{confidence}")


def show_caption_simlarity():
    caption_embedding_root_path = rf"/workspace/datasets/ucf-crime/swinbert/caption_embeddings/train/anomaly"

    # 分析记忆和被使用次数之间的关系
    video_names = ["RoadAccidents047_x264-15.avi", "Robbery128_x264-4.avi", "Shoplifting040_x264-13.avi",
                   "Burglary078_x264-14.avi", "Stealing071_x264-22.avi", "Burglary083_x264-4.avi",
                   "Robbery144_x264-29.avi", "Arrest049_x264-18.avi", "Burglary043_x264-19.avi", "Arson052_x264-21.avi"]

    ids = [int(video_name.replace(".avi", "").split("-")[1]) for video_name in video_names]
    video_names = [video_name.replace(".avi", "").split("-")[0] for video_name in video_names]

    # 检测记忆之间相似度的关系，筛选比较合适的保留
    embeddings = []
    # video_names = ["RoadAccidents100_x264", "Fighting032_x264", "Robbery129_x264", "RoadAccidents073_x264",
    #                "Stealing075_x264", "Arrest049_x264", "Arrest002_x264", "Robbery135_x264"]
    # ids = [11, 13, 27, 13, 20, 12, 24, 24]
    #
    # video_names = ["Assault036_x264", "Burglary050_x264", "RoadAccidents140_x264", "Shoplifting053_x264",
    #                "Stealing093_x264", "RoadAccidents120_x264", "RoadAccidents051_x264"]
    # ids = [29, 6, 25, 9, 27, 0, 28]
    for video_name, idx in zip(video_names, ids):
        # video_name = video_name.split("/")[-1].replace(".mp4", "")
        # caption_filepath = os.path.join(self.caption_root_path, video_name + ".npy")
        # data = np.load(caption_filepath, allow_pickle=True).item()
        # embedding = self.get_semantic_embedding(video_name)[idx]
        # name = name.split("/")[-1].replace(".mp4", "")
        embedding = np.load(os.path.join(caption_embedding_root_path, f"{video_name}.npy"), allow_pickle=True)[idx]
        embeddings.append(embedding)
    sims = []
    from sklearn.metrics.pairwise import cosine_similarity

    for idx, embedding in enumerate(embeddings):
        # sim = cosine_similarity(torch.Tensor(embedding), torch.stack(torch.Tensor(embeddings), dim=0))
        if idx == 0:
            sim = cosine_similarity([embedding], embeddings[1:])
        elif idx == len(embeddings) - 1:
            sim = cosine_similarity([embedding], embeddings[:-1])
        else:
            sim1 = cosine_similarity([embedding], embeddings[:idx])
            sim2 = cosine_similarity([embedding], embeddings[idx + 1:])
            sim = np.concatenate((sim1, sim2), axis=1)
        print(sim)
        sims.append(sim.mean())
    print("sims:")
    print(sims)


if __name__ == '__main__':
    # loader2 = Normal_Loader(is_train=0)
    # loader2 = Anomaly_Loader(is_train=0)
    # video_name, result = loader2.show_caption("378-15")
    # print(result)
    import sys

    # caption_idx = ["88-1","378-14","369-9","38-24","771-8","569-2","442-26"]

    # show_video_captions()
    show_caption_simlarity()

    # for i in range(10):
    #     print(loader2.__getitem__(i)[1])
    #     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # print(len(loader2))
    # print(loader[1], loader2[1])

# 85.306
