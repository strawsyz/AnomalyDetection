import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from kmeans_pytorch import kmeans
import random
from transformer import Transformer


class Learner(nn.Module):
    def __init__(self, input_dim=512, drop_p=0.6, memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                 threshold_caption_score=0.1, nk=False, args=None):
        super(Learner, self).__init__()
        if input_dim == 512:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif input_dim == 2048:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif input_dim == 2560:
            self.classifier = nn.Sequential(
                nn.Linear(512, 32),
                nn.Linear(2048, 512),
                # nn.ReLU(),
                # nn.Dropout(drop_p),
                nn.Linear(512, 32),
                # nn.ReLU(),
                # nn.Dropout(drop_p),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        self.input_dim = args.input_dim
        self.drop_p = args.drop
        self.weight_init()
        self.vars = nn.ParameterList()
        self.nk = args.nk

        self.n_memory = []
        self.a_memory = []

        self.n_memory_0 = []
        self.a_memory_0 = []

        self.memory_rate = args.memory_rate  # 范围0-1， 按照一定概率随机记忆， 等于1的时候会记忆所有数据
        self.a_topk = args.a_topk

        self.rates = [0.4, 0.6, 0.8, 0.9]
        # 多级memory，根据不同的layer层次存储不同的memory

        self.threshold_a_caption_score = args.threshold_caption_score
        self.threshold_n_caption_score = args.threshold_caption_score  # 越大，需要记忆的memory就越多，loss会有一点点减少，auc能有一点的提升
        # self.threshold_memory_size = 3
        self.threshold_a_memory_size = args.max_memory_size
        self.threshold_n_memory_size = args.max_memory_size
        self.min_a_memory_size = args.num_key_memory
        self.min_n_memory_size = args.num_key_memory
        self.topk_score = args.topk_score
        self.optimize_topk = 0
        self.update_threshold = args.update_threshold
        self.tf = Transformer(args.input_dim, args.embedding_dim, args.n_layer, args.n_head, 0)
        self.reducer_4_tf = nn.Linear(args.embedding_dim, 32)
        self.mse = torch.nn.MSELoss(reduce=False)

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def similarity_a_n_memory_space(self):
        # cosine
        # n_memory = torch.stack(self.n_memory)
        # similarity_scores = []
        # for a_memory in self.a_memory:
        #     saliency_score = a_memory.mm(torch.transpose(n_memory, 0, 1)).mean(dim=1)
        #     saliency_score = torch.cosine_similarity(a_memory, n_memory, dim=1).abs().mean()
        #     similarity_scores.append(saliency_score)
        # similarity_score = torch.stack(similarity_scores).mean()

        # mm
        similarity_score = torch.stack(self.n_memory).mm(torch.transpose(torch.stack(self.a_memory),0,1)).mean()

        return similarity_score

    def memory_stability(self):
        memory = torch.stack(self.a_memory)
        saliency_scores = self._calcu_saliency_score_in_memory(memory)
        a_stability = torch.mean(saliency_scores)
        memory = torch.stack(self.n_memory)
        saliency_scores = self._calcu_saliency_score_in_memory(memory)
        n_stability = torch.mean(saliency_scores)
        return a_stability + n_stability

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
        #     saliency_score = torch.cosine_similarity(_memory, memory, dim=1).abs() #.median()
        #     saliency_score = torch.nn.functional.softmax(saliency_score)  # softmax
        #     saliency_score = saliency_score.median()  # median
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
        # saliency_scores = memory.mm(torch.transpose(memory, 0, 1)).mean(dim=1)

        # mse two
        # self.mse(memory, torch.transpose(memory, 0, 1))
        saliency_scores = []
        for memory_  in memory:
            saliency_scores.append(self.mse(memory_, memory).mean())
        saliency_scores = torch.stack(saliency_scores)
        # print(saliency_scores)

        return saliency_scores


    def clear_memory(self, rate=None, epoch=None):
        """按照一定的概率删除掉一部分数据"""
        print("=======================starting clear memory=======================")

        # if self.rates is not None:
        #     if len(self.rates) <= epoch:
        #         rate = self.rates[-1]
        #     else:
        #         rate = self.rates[epoch]

        if rate is None:
            raise RuntimeError("Do not have a clear rate")
            # self.a_memory = []
            # self.n_memory = []
        else:
            indexes = [i for i in range(len(self.a_memory))]
            old_length = len(indexes)
            new_length = max(1, int(len(indexes) * rate))
            if new_length > self.threshold_a_memory_size:
                np.random.shuffle(indexes)
                indexes = indexes[:new_length]
                self.a_memory = [self.a_memory[i] for i in indexes]
                print(f"clear a memory {old_length} -> {new_length}")

            indexes = [i for i in range(len(self.n_memory))]
            old_length = len(indexes)
            new_length = max(1, int(len(indexes) * rate))
            if new_length > self.threshold_n_memory_size:
                np.random.shuffle(indexes)
                indexes = indexes[:new_length]
                self.n_memory = [self.n_memory[i] for i in indexes]
                print(f"clear n memory {old_length} -> {new_length}")

        print("=======================ending clear memory=======================")

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

            saliency_indexes = torch.argsort(saliency_scores)  #  选择相似度最小，特异度最大的记忆保存
            # saliency_indexes = torch.argsort(saliency_scores, descending=True)  # 选择相似度最大，特异度最小的模型保存
            indexes = saliency_indexes[
                      self.optimize_topk:self.min_a_memory_size + self.optimize_topk]  # 挑选相似度最小的几个，让记忆的内容尽可能不相同
            for index in indexes:
                a_memory.append(self.a_memory[index])
            if self.update_threshold:
                print(saliency_scores[index])
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

            saliency_indexes = torch.argsort(saliency_scores)  # 选择相抵最小，特异度最大的记忆保存
            # saliency_indexes = torch.argsort(saliency_scores, descending=True)
            indexes = saliency_indexes[self.optimize_topk:self.min_n_memory_size + self.optimize_topk]
            for index in indexes:
                n_memory.append(self.n_memory[index])
            if self.update_threshold:
                print(saliency_scores[index])
                self.threshold_n_caption_score = min(self.threshold_n_caption_score, saliency_scores[index])

            # n_memory = self.cluster_memory(torch.stack(n_memory), 3)
            print(f" {len(self.n_memory)} -> {len(indexes)}")
            self.n_memory = n_memory
    def calculate_feature_score(self, memory, feature):
        """caption score越高，说明和其他的memory比较接近，没有保存的必要
        越低，说明是一个比较新的样本，可以用来扩张memory的空间"""
        cal_method = ["mul", "mul-avg", "ss", "cos", "cos-abs"]
        if len(memory) == 0:
            return Variable(torch.tensor(0.5), requires_grad=True).cuda()
        # cap 特征量不是小数无法直接用于计算相似度
        # cap_feature = cap.cpu().numpy()
        # 可以生成caption，但是必须转化为list

        # embeddings = self.ss.encode(memory + [feature])
        # from sklearn.metrics.pairwise import cosine_similarity
        feature = torch.unsqueeze(feature, dim=0)

        # 计算cos相似度作为距离
        # if cal_method == "cos":
        #     caption_scores = []
        #     for in_feature in memory:
        #         in_feature = torch.unsqueeze(in_feature, dim=0)
        #         caption_scores.append(Variable(torch.cosine_similarity(in_feature, feature), requires_grad=True).abs())
        #     caption_score = torch.cat(caption_scores, dim=0).max()
        # elif cal_method =="cos-abs":
        #     # abs
        #     caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
        #                              requires_grad=True).abs()
        #     caption_score = torch.nn.functional.softmax(caption_score)
        #     caption_score = caption_score.median()

        # no abs
        # caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
        #                          requires_grad=True).max()
        # caption_score = (caption_score + 1) / 2

        # if type(memory) is list:
        # print("list")
        # else:
        #     raise RuntimeError("not list")
        #     caption_score = (Variable(memory) * feature).mean(dim=1).max()  # 使用乘号来表示相似度
        # print("cosine similarity:", caption_score.data, "multiple similarity:", caption_score1.data)
#  相似度的计算方式， 优化记忆保存的方式，清理记忆的方式
        # 使用乘号来表示相似度
        caption_scores = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1)
        # length_feature = feature.shape[1]
        # caption_scores = Variable(self.mse(feature, torch.stack(memory, dim=0)))/length_feature

        if self.topk_score == 1:
            caption_score = caption_scores.max()
        else:
            k = min(self.topk_score, len(caption_scores))
            topk_caption_scores = caption_scores.topk(k=k, largest=True).values
            caption_score = topk_caption_scores.mean()

        # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        return caption_score

        # if len(caption_scores) == 1:
        #     return caption_scores[0]
        # else:
        #     # caption_score = torch.max(torch.tensor(caption_scores), dim=0)[1]
        #     caption_score = torch.cat(caption_scores, dim=0).max()
        #     # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        #     return caption_score

    def calculate_anomaly_score(self, caption, gt, update=True):
        if gt == -1:
            memory = self.n_memory
            threshold = self.threshold_n_caption_score
        elif gt == 1:
            memory = self.a_memory
            threshold = self.threshold_a_caption_score
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
        return a_caption_score - n_caption_score
        # return a_caption_score

    def add_anomaly_memory(self, x):
        batch_size = int(len(x) / 64)
        assert len(x) % 64 == 0
        k = self.a_topk
        if len(x) % 64 == 0:
            for i in range(batch_size):
                gt = 1
                # 添加异常视频中可能为异常片段的记忆
                # index_a = torch.argmax(output1[i * 2 * 32: 2 * i * 32 + 32])
                # index_a = torch.argmax(x[i * 2 * 32: 2 * i * 32 + 32].max(dim=1)[0])

                # index_a = torch.argmax(x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1))  # 默认将第一个最强的目标当作异常，
                # index = 2 * i * 32 + index_a
                # caption = x[index]
                # anomaly_score = self.calculate_anomaly_score(caption, gt, update=True)

                #  添加前k个重要的记忆
                index_a = torch.argsort(x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1))[-k:]
                # x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1)
                index = 2 * i * 32 + index_a
                captions = x[index]
                for caption in captions:
                    self.calculate_anomaly_score(caption, gt, update=True)

                # 添加正常的记忆
                # index_n = torch.argmax(output1[2 * i * 32 + 32: 2 * i * 32 + 64])
                # index = 2 * i * 32 + 32 + index_n
                # caption = x[index]
                # gt = -1
                # anomaly_score = self.calculate_anomaly_score(caption, gt)

    def forward(self, x, vars=None, gt=None):
        if vars is None:
            vars = self.vars

        # feat_magnitudes = torch.norm(x, p=2, dim=2)  # 可以用于CLIP提取的参数
        # print(feat_magnitudes.shape)  # 10, 28  # 640, 28

        batch_size = int(x.shape[0] / 64)
        feature_dim = x.shape[1]
        # if self.training:  # batch_size > 0:
        #     x = torch.reshape(x, (batch_size, 64, feature_dim))
        #     a_x = x[:, :32, :]
        #     n_x = x[:, 32:, :]
        #     a_attn, a_output = self.tf(a_x, None)
        #     n_attn, n_output = self.tf(n_x, None)
        #     a_output = a_attn.mean(dim=2) + a_output.mean(dim=2)
        #     n_output = n_attn.mean(dim=2) + n_output.mean(dim=2)
        #     # a_output = self.reducer_4_tf(a_output)
        #     # n_output = self.reducer_4_tf(n_output)
        #     # a_output = a_attn + a_output
        #     # n_output = n_attn + n_output
        #     x = torch.cat([a_output, n_output], dim=1)
        #     # x = x.reshape(batch_size * 64, 32)
        # else:
        #     attn, output = self.tf(x)
        #     output = attn.mean(dim=2) + output.mean(dim=2)
        #
        #     # output = self.reducer_4_tf(output)
        #     # output = attn + output
        #     x = torch.squeeze(output)

        # 不通过模型，直接使用特征量计算，loss一直在2以上 # AUC: 0.34005
        if self.input_dim == 512:
            x = x.float()
            x = F.linear(x, vars[0], vars[1])
            # 增加一层
            x = F.relu(x)
            # x = F.softmax(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
        elif self.input_dim == 2048:
            # x = x.float()
            x = F.linear(x, vars[0], vars[1])
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
        elif self.input_dim == 2560:
            x = x.float()
            x1 = x[:, :512]
            x2 = x[:, 512:]
            x1 = F.linear(x1, vars[0], vars[1])
            x2 = F.linear(x2, vars[2], vars[3])
            x2 = F.relu(x2)
            x2 = F.dropout(x2, self.drop_p, training=self.training)
            x2 = F.linear(x2, vars[4], vars[5])
            x = x1 + x2
        else:
            x = x.float()
            x = F.linear(x, vars[0], vars[1])
            # 增加一层
            x = F.relu(x)
            # x = F.softmax(x)
            x = F.dropout(x, self.drop_p, training=self.training)

        if self.nk:
            if self.training:
                self.add_anomaly_memory(x)
            outputs = [0 for i in range(len(x))]

            indexes = np.arange(len(x))
            np.random.shuffle(indexes)  # 防止一个视频一个视频的放入数据，导致初始空间偏向第一个视频，感觉没什么用
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
            outputs = torch.stack(outputs, dim=0) * x.mean(dim=1) # 0.8339315091549631
            # outputs = torch.stack(outputs, dim=0) + x.mean(dim=1) # 0.8173835234501754
            outputs = torch.sigmoid(outputs)
            outputs = torch.unsqueeze(outputs, dim=1)

            return outputs
        else:
            # x = F.relu(x)
            # x = F.dropout(x, self.drop_p, training=self.training)
            # # if self.input_dim == 512:
            # #     x = F.linear(x, vars[2], vars[3])
            # # else:
            # x = F.linear(x, vars[4], vars[5])
            # x = torch.sigmoid(x)
            # a_attn.mean(dim=2) + a_output.mean(dim=2)

            x = x.mean(dim=1)
            x = F.sigmoid(x)
            return x
        # inputs = torch.clone(x)
        # x = F.relu(x)  # 改善了0.02
        # x = F.dropout(x, self.drop_p, training=self.training)  # 改善了不少  #0.5191871161161531
        # x = F.linear(x, vars[2], vars[3])  # AUC 是0.468左右
        # x = F.dropout(x, self.drop_p, training=self.training)  # 0.44033766356249904
        # 层数越深，需要记忆的模式就越多，AUC就越高，直接使用memory反而会对结果产生不好的影响
        # loss 也有微妙的下降的趋势，但是auc只在0.5的附近徘徊
        # 层数越深，需要记忆的模式越多，但是每个记忆的数量非常的少
        # 记忆的长度有个合适的值，太短会导致视频的大部分信息的丢失，太长会导致特征基本都比较相似，需要记忆的内容就比较小
        # 层数越深，特征之间的相似度就越小，特征量越短，
        # 模型太多比较的次数可能会太多

        # 由于没有使用预训练的模型，导致初期生成的特征量没有参考性，结果就不好
        # x_1 = F.linear(x, vars[4], vars[5])
        # output1 = torch.sigmoid(x_1)
        # return output1

        # self.add_anomaly_memory(inputs, output1)

        # return  output1
        # return (outputs + output1)/2  # 加了sigmoid的之后的loss不再下降, 精度会有一点的下降

        # --input_dim 2048 --lr 0.01 --tf0.8383009988878762

        # batch_size = int(x.shape[0] / 64)
        # if self.training: #batch_size > 0:
        #     x = torch.reshape(x, (batch_size, 64, 32))
        #     a_x = torch.reshape(x, (batch_size, 64, 32))[:, :32, :]
        #     n_x = torch.reshape(x, (batch_size, 64, 32))[:, 32:, :]
        #     a_output = self.tf(a_x)
        #     n_output = self.tf(n_x)
        #     output = torch.cat([a_output, n_output], dim=1)
        #     output = torch.reshape(output, (int(batch_size*64), 1))
        #     output = F.sigmoid(output)
        #     # output = F.relu(output)
        #     return output
        # else:
        #     output = self.tf(x)
        #     output = output.squeeze()
        #     output = F.sigmoid(output)
        #     # output = F.relu(output)
        #     return output

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:me
        """
        return self.vars


class NK(nn.Module):
    def __init__(self, input_dim=512, drop_p=0.6, memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                 threshold_caption_score=0.1, nk=False):
        super(NK, self).__init__()
        # INIT PARAMETER
        self.input_dim = input_dim
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        self.nk = nk

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
        self.topk_score = 3
        self.optimize_topk = 1

        self.reducer = None
        if input_dim == 512:
            self.reducer = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        elif input_dim == 1024:
            self.reducer = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        elif input_dim == 2048:
            self.reducer = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        elif input_dim == 2560:
            self.reducer = nn.Sequential(
                nn.Linear(512, 32),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
        else:
            self.reducer = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )
            # raise RuntimeError("Can supprt such input dim")

        # self.tf = Transformer(32, 32, 6, 8, 0)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

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

        # multiply two memory
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

            # self.threshold_a_caption_score = min(self.threshold_a_caption_score, saliency_scores[index])
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

            # self.threshold_n_caption_score = min(self.threshold_n_caption_score, saliency_scores[index])

            # n_memory = self.cluster_memory(torch.stack(n_memory), 3)
            print(f" {len(self.n_memory)} -> {len(indexes)}")
            self.n_memory = n_memory

    def clear_memory(self, rate=None, epoch=None):
        """按照一定的概率删除掉一部分数据"""
        print("=======================starting clear memory=======================")
        # if self.rates is not None:
        #     if len(self.rates) <= epoch:
        #         rate = self.rates[-1]
        #     else:
        #         rate = self.rates[epoch]

        if rate is None:
            raise RuntimeError("Do not have a clear rate")
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
        # cap 特征量不是小数无法直接用于计算相似度
        # cap_feature = cap.cpu().numpy()
        # 可以生成caption，但是必须转化为list

        # embeddings = self.ss.encode(memory + [feature])
        # from sklearn.metrics.pairwise import cosine_similarity
        feature = torch.unsqueeze(feature, dim=0)
        # caption_scores = []
        # for in_feature in memory:
        #     in_feature = torch.unsqueeze(in_feature, dim=0)
        #     caption_scores.append(Variable(torch.cosine_similarity(in_feature, feature), requires_grad=True).abs)
        #
        # caption_score = torch.cat(caption_scores, dim=0).max()

        # abs
        # caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
        #                          requires_grad=True).abs().max()

        # no abs
        # caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
        #                          requires_grad=True).max()
        # caption_score = (caption_score + 1) / 2

        # multiple
        # caption_score = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1).max()  # 使用乘号来表示相似度
        # print((Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1))
        # print(torch.argmax((Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1)))

        caption_scores = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1)

        if self.topk_score == 1:
            caption_score = caption_scores.max()
        else:
            k = min(self.topk_score, len(caption_scores))
            topk_caption_scores = caption_scores.topk(k=k, largest=True).values
            caption_score = topk_caption_scores.mean()

        # if type(memory) is list:
        # print("list")
        # else:
        #     raise RuntimeError("not list")
        #     caption_score = (Variable(memory) * feature).mean(dim=1).max()  # 使用乘号来表示相似度
        # print("cosine similarity:", caption_score.data, "multiple similarity:", caption_score1.data)

        # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        return caption_score

        # if len(caption_scores) == 1:
        #     return caption_scores[0]
        # else:
        #     # caption_score = torch.max(torch.tensor(caption_scores), dim=0)[1]
        #     caption_score = torch.cat(caption_scores, dim=0).max()
        #     # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        #     return caption_score

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
        if vars is None:
            vars = self.vars

        # feat_magnitudes = torch.norm(x, p=2, dim=2)
        # print(feat_magnitudes.shape)  # 10, 28  # 640, 28
        # print(vars[0][0][-10:])
        # print(vars[1][0])
        # 不通过模型，直接使用特征量计算，loss一直在2以上 # AUC: 0.34005
        x = x.float()
        x = self.reducer(x)

        # --input_dim 2048 --lr 0.01 --tf0.8383009988878762
        # batch_size = int(x.shape[0] / 64)
        # if self.training: #batch_size > 0:
        #     x = torch.reshape(x, (batch_size, 64, 32))
        #     a_x = torch.reshape(x, (batch_size, 64, 32))[:, :32, :]
        #     n_x = torch.reshape(x, (batch_size, 64, 32))[:, 32:, :]
        #     a_output = self.tf(a_x)
        #     n_output = self.tf(n_x)
        #     output = torch.cat([a_output, n_output], dim=1)
        #     output = torch.reshape(output, (int(batch_size*64), 1))
        #     output = F.sigmoid(output)
        #     # output = F.relu(output)
        #     return output
        # else:
        #     output = self.tf(x)
        #     output = output.squeeze()
        #     output = F.sigmoid(output)
        #     # output = F.relu(output)
        #     return output

        if self.nk:
            if self.training:
                self.add_anomaly_memory(x, None)
            outputs = [0 for i in range(len(x))]

            indexes = np.arange(len(x))
            # np.random.shuffle(indexes)  # 防止按顺序放入所有视频时，导致初始空间偏向第一个视频，感觉没什么用
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
        else:
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            if self.input_dim == 512:
                x = F.linear(x, vars[2], vars[3])
            else:
                x = F.linear(x, vars[4], vars[5])
            x = torch.sigmoid(x)
            return x
        # inputs = torch.clone(x)
        # x = F.relu(x)  # 改善了0.02
        # x = F.dropout(x, self.drop_p, training=self.training)  # 改善了不少  #0.5191871161161531
        # x = F.linear(x, vars[2], vars[3])  # AUC 是0.468左右
        # x = F.dropout(x, self.drop_p, training=self.training)  # 0.44033766356249904
        # 层数越深，需要记忆的模式就越多，AUC就越高，直接使用memory反而会对结果产生不好的影响
        # loss 也有微妙的下降的趋势，但是auc只在0.5的附近徘徊
        # 层数越深，需要记忆的模式越多，但是每个记忆的数量非常的少
        # 记忆的长度有个合适的值，太短会导致视频的大部分信息的丢失，太长会导致特征基本都比较相似，需要记忆的内容就比较小
        # 层数越深，特征之间的相似度就越小，特征量越短，
        # 模型太多比较的次数可能会太多

        # 由于没有使用预训练的模型，导致初期生成的特征量没有参考性，结果就不好
        # x_1 = F.linear(x, vars[4], vars[5])
        # output1 = torch.sigmoid(x_1)
        # return output1

        # self.add_anomaly_memory(inputs, output1)

        # return  output1
        # return (outputs + output1)/2  # 加了sigmoid的之后的loss不再下降, 精度会有一点的下降

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:me
        """
        return self.vars
