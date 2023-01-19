import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from kmeans_pytorch import kmeans
import random


class NK(nn.Module):
    def __init__(self, memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                 threshold_caption_score=0.1):
        super(NK, self).__init__()

        # self.weight_init()
        # self.vars = nn.ParameterList()

        self.n_memory = []
        self.a_memory = []

        self.n_memory_0 = []
        self.a_memory_0 = []

        self.memory_rate = memory_rate  # 范围0-1， 按照一定概率随机记忆， 等于1的时候会记忆所有数据

        self.rates = [0.4, 0.6, 0.8, 0.9]
        # 多级memory，根据不同的layer层次存储不同的memory

        self.threshold_a_caption_score = threshold_caption_score
        self.threshold_n_caption_score = threshold_caption_score  # 越大，需要记忆的memory就越多，loss会有一点点减少，auc能有一点的提升
        # self.threshold_memory_size = 3
        self.threshold_a_memory_size = max_memory_size
        self.threshold_n_memory_size = max_memory_size
        self.min_a_memory_size = num_key_memory
        self.min_n_memory_size = num_key_memory
        self.topk_socre = 3
        self.optimize_topk = 1

    def cluster_memory(self, memory, num_clusters):
        cluster_ids_x, cluster_centers = kmeans(
            X=memory, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0')
        )
        memory = []
        for cluster_center in cluster_centers:
            memory.append(cluster_center)
        return memory
        # return cluster_centers.cuda()

    def _calcu_saliency_score_in_memory(self, memory):

        # cosine abs
        # saliency_scores = []
        # for _memory in memory:
        #     saliency_score = torch.cosine_similarity(_memory, memory, dim=1).abs().mean()
        #     saliency_scores.append(saliency_score)
        # saliency_scores = torch.stack(saliency_scores)

        # cosine no abs
        # saliency_scores = []
        # for _memory in memory:
        #     saliency_score = torch.cosine_similarity(_memory, memory, dim=1)
        #     saliency_score = (saliency_score + 1) / 2
        #     saliency_score = saliency_score.mean()
        #     saliency_scores.append(saliency_score)
        # saliency_scores = torch.stack(saliency_scores)

        # multiply
        saliency_scores = memory.mm(torch.transpose(memory, 0, 1)).mean(dim=1)

        return saliency_scores

    def optimize_memory(self):
        """optimize_memory"""
        a_memory = []
        print(self.threshold_a_caption_score)
        print(self.threshold_n_caption_score)

        if len(self.a_memory) > self.threshold_a_memory_size:
            memory = torch.stack(self.a_memory)
            # feat_magnitudes = torch.norm(video_embeds, p=2, dim=2)

            # self.a_memory = self.cluster_memory(memory, self.min_a_memory_size)

            saliency_scores = self._calcu_saliency_score_in_memory(memory)

            saliency_indexes = torch.argsort(saliency_scores)
            indexes = saliency_indexes[
                      self.optimize_topk:self.min_a_memory_size + self.optimize_topk]  # 挑选相似度最小的几个，让记忆的内容尽可能不相同
            for index in indexes:
                a_memory.append(self.a_memory[index])

            self.threshold_a_caption_score = min(self.threshold_a_caption_score, saliency_scores[index])
            # self.cluster_memory(torch.stack(a_memory), 3)
            print(f" {len(self.a_memory)} -> {len(indexes)}")
            self.a_memory = a_memory
        n_memory = []
        if len(self.n_memory) > self.threshold_n_memory_size:
            memory = torch.stack(self.n_memory)
            # self.n_memory = self.cluster_memory(memory, self.min_n_memory_size)

            # indexes = torch.argmax(memory * torch.transpose(memory, 0, 1))

            saliency_scores = self._calcu_saliency_score_in_memory(memory)

            saliency_indexes = torch.argsort(saliency_scores)
            indexes = saliency_indexes[self.optimize_topk:self.min_n_memory_size + self.optimize_topk]
            for index in indexes:
                n_memory.append(self.n_memory[index])

            self.threshold_n_caption_score = min(self.threshold_n_caption_score, saliency_scores[index])

            # n_memory = self.cluster_memory(torch.stack(n_memory), 3)
            print(f" {len(self.n_memory)} -> {len(indexes)}")
            self.n_memory = n_memory

    def clear_memory(self, rate=None, epoch=None):
        """按照一定的概率删除掉一部分数据"""
        print("=======================starting clear memory=======================")

        if rate is None:
            pass
            # self.a_memory = []
            # self.n_memory = []
        else:
            indexes = [i for i in range(len(self.a_memory))]
            old_length = len(indexes)
            new_length = max(1, int(len(indexes) * rate))
            if new_length > self.threshold_a_memory_size:
                np.random.shuffle(indexes)
                print(f"{len(indexes)}->{max(1, int(len(indexes) * rate))}")
                indexes = indexes[:max(1, int(len(indexes) * rate))]
                self.a_memory = [self.a_memory[i] for i in indexes]
            print(f"clear a memory {old_length} -> {new_length}")

            indexes = [i for i in range(len(self.n_memory))]
            old_length = len(indexes)
            new_length = max(1, int(len(indexes) * rate))
            if new_length > self.threshold_n_memory_size:
                np.random.shuffle(indexes)
                print(f"{len(indexes)}->{max(1, int(len(indexes) * rate))}")
                indexes = indexes[:max(1, int(len(indexes) * rate))]
                self.n_memory = [self.n_memory[i] for i in indexes]
            print(f"clear n memory {old_length} -> {new_length}")

        print("=======================ending clear memory=======================")

    def calculate_feature_score(self, memory, feature):
        """caption score越高，说明和其他的memory比较接近，没有保存的必要
        越低，说明是一个比较新的样本，可以用来扩张memory的空间"""

        if len(memory) == 0:
            return Variable(torch.tensor(0.5), requires_grad=True).cuda()

        feature = torch.unsqueeze(feature, dim=0)

        caption_scores = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1)

        if self.topk_socre == 1:
            caption_score = caption_scores.mean()
        else:
            k = min(self.topk_socre, len(caption_scores))
            topk_caption_scores = caption_scores.topk(k=k, largest=True).values
            caption_score = topk_caption_scores.mean()

        return caption_score

    def calculate_anomaly_score(self, caption, gt, update=True):
        if gt == -1:
            memory = self.n_memory
            threshold = self.threshold_n_caption_score
            # memory_threshold = self.threshold_n_memory_size
        elif gt == 1:
            memory = self.a_memory
            threshold = self.threshold_a_caption_score
            # memory_threshold = self.threshold_a_memory_size
        else:
            raise RuntimeError("No such memory")

        a_caption_score = self.calculate_feature_score(self.a_memory, caption)
        n_caption_score = self.calculate_feature_score(self.n_memory, caption)
        # print("a_caption_score:", a_caption_score, "n_caption_score:", n_caption_score, )
        if self.training and update:
            if len(memory) > 1:
                if gt == 1:
                    feature_score = a_caption_score
                elif gt == -1:
                    feature_score = n_caption_score
                # feature_score = self.calculate_feature_score(memory, caption)
                # feature score越大，说明和其他的特征量越相似
                # print("feature_score:", feature_score)
                if threshold > feature_score:
                    if random.random() < self.memory_rate:
                        memory.append(caption)
                # if len(memory) > self.threshold_n_memory_size
            else:
                if random.random() < self.memory_rate:
                    memory.append(caption)
        # print("a_score", a_caption_score)
        # print("n_score", n_caption_score)
        return a_caption_score - n_caption_score

    def add_anomaly_memory(self, x, output1):
        batch_size = int(len(x) / 64)
        assert len(x) % 64 == 0
        if len(x) % 64 == 0:
            for i in range(batch_size):
                # 添加异常视频中可能为异常片段的记忆
                # index_a = torch.argmax(output1[i * 2 * 32: 2 * i * 32 + 32])
                # index_a = torch.argmax(x[i * 2 * 32: 2 * i * 32 + 32].max(dim=1)[0])
                index_a = torch.argmax(x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1))
                # x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1)
                index = 2 * i * 32 + index_a
                caption = x[index]
                gt = 1
                anomaly_score = self.calculate_anomaly_score(caption, gt, update=True)
                # 添加正常的记忆
                # index_n = torch.argmax(output1[2 * i * 32 + 32: 2 * i * 32 + 64])
                # index = 2 * i * 32 + 32 + index_n
                # caption = x[index]
                # gt = -1
                # anomaly_score = self.calculate_anomaly_score(caption, gt)

    def forward(self, x, vars=None, gt=None):

        if self.training:
            self.add_anomaly_memory(x, None)
        outputs = [0 for i in range(len(x))]

        indexes = np.arange(len(x))
        # np.random.shuffle(indexes)  # 防止一个视频一个视频的放入数据，导致初始空间偏向第一个视频，感觉没什么用
        for index in indexes:
            caption = x[index]
            # if index < int(len(x) // 2):
            if (index // 32) % 2 == 0:  # 如果是0~31，之类的情况
                gt = 1  # 异常的数据
                anomaly_score = self.calculate_anomaly_score(caption, gt, update=False)
            else:
                gt = -1  # 正常的数据
                if self.training:
                    anomaly_score = self.calculate_anomaly_score(caption, gt, update=True)
                else:
                    anomaly_score = self.calculate_anomaly_score(caption, gt, update=False)
            outputs[index] = anomaly_score
        outputs = torch.stack(outputs, dim=0)
        outputs = torch.sigmoid(outputs)
        outputs = torch.unsqueeze(outputs, dim=1)

        return outputs

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:me
        """
        return self.vars
