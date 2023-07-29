import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from kmeans_pytorch import kmeans
import random
from transformer import Transformer, Encoder
from modules import ScaledDotProductAttention
from sentence_transformers import SentenceTransformer


class Learner(nn.Module):
    def __init__(self, input_dim=512, drop_p=0.6, memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                 threshold_caption_score=0.1, nk=False, args=None):
        super(Learner, self).__init__()
        self.feature_name = args.feature_name

        self.n_memory = []
        self.a_memory = []
        self.n_caption_memory = []
        self.a_caption_memory = []
        self.n_caption_embedding = []
        self.a_caption_embedding = []

        self.n_memory_0 = []
        self.a_memory_0 = []

        self.memory_rate = args.memory_rate  # 范围0-1， 按照一定概率随机记忆， 等于1的时候会记忆所有数据
        self.a_topk = args.a_topk

        self.rates = [0.4, 0.6, 0.8, 0.9]
        # 多级memory，根据不同的layer层次存储不同的memory
        self.encoder = Encoder(768, 768, 6, 8)

        self.threshold_a_caption_score = args.threshold_caption_score
        self.threshold_n_caption_score = args.threshold_caption_score  # 越大，需要记忆的memory就越多，loss会有一点点减少，auc能有一点的提升
        # self.threshold_memory_size = 3
        self.threshold_a_memory_size = args.max_memory_size
        self.threshold_n_memory_size = args.max_memory_size
        self.min_a_memory_size = args.num_key_memory
        self.min_n_memory_size = args.num_key_memory
        self.topk_score = args.topk_score  # 选择计算top-k的发呢书
        self.optimize_topk = 0  # 优化的使用，不保存最前面的几个值，来避免掉
        self.update_threshold = args.update_threshold

        self.caption_temp = args.caption_temp
        self.used_caption_in_inference = []
        # self.tf = Transformer(args.input_dim, args.embedding_dim, args.n_layer, args.n_head, 0)

        if input_dim == 512:
            self.classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                # nn.Sigmoid()
            )
        elif input_dim in [2048, 2049]:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                # nn.Sigmoid()
            )
        elif input_dim == 2560:
            self.classifier = nn.Sequential(
                nn.Linear(512, 32),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                # nn.Sigmoid()
            )
        elif input_dim == 1024:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
            )

        # caption embedding 用的线性层
        self.mlp_4_caption_embedding = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        # 将特征量的压缩为32位，来让nk模型使用
        self.mlp_4_caption_embedding_32 = nn.Sequential(
            # nn.Linear(768, 768),
            nn.Linear(768, 128),
            nn.Sigmoid(),
            # nn.Dropout(drop_p),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            # nn.Dropout(drop_p),
            # nn.Linear(32, 1)
        )

        # self.scaled_dot_product_attention_4_locate_anomaly = ScaledDotProductAttention(math.sqrt(32), 0)
        self.scaled_dot_product_attention_4_optimize_memory = ScaledDotProductAttention(math.sqrt(32), 0)
        self.scaled_dot_product_attention_4_calculate_scores = ScaledDotProductAttention(math.sqrt(32), 0)
        self.input_dim = args.input_dim
        self.drop_p = args.drop
        self.weight_init()  # 初始化模型权重
        self.vars = nn.ParameterList()
        self.nk = args.nk

        self.reducer_4_tf = nn.Linear(args.embedding_dim, 32)
        self.mse = torch.nn.MSELoss(reduce=False)
        cal_method = ["mul", "mul-avg", "ss", "cos", "cos-abs", "mse", "mul-abs", "dot"]
        self.distance = args.distance  # 计算距离的方式
        assert self.distance in cal_method, self.distance
        self.score_cal_method = self.distance  # "mul"
        self.saliency_cal_method = self.distance  # "cos" # self.distance

        self.sentence_similarity = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # embeddings = model.encode([s1, s2], convert_to_tensor=True)
        # print(embeddings)
        # res = cosine_similarity(torch.unsqueeze(embeddings[0], dim=0), torch.unsqueeze(embeddings[1], dim=0))

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        # 初始化全线性层的参数
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def embed_attn_output(self, attn, output):
        attn = torch.stack(attn)
        attn = attn[-1, :, :, :]
        attn = attn.mean(dim=1)

        return attn, output

    def tf(self, src, src_mask=None):
        output, attn = self.encoder(src, src_mask)
        attn, output = self.embed_attn_output(attn, output)
        output = output.mean(dim=2)  # 0.8175993762866813
        return output

    def init_memory_space(self, anomaly_dataset, normal_dataset, args):
        # 读取提前保存好的
        # ，读取之后直接使用
        # 模型会参数会改变保留memory space可能没什么用
        # 更具保存好的snippetid。找打对应的特征量，每个epoch读取特征量，然后生成与之对应的memory，并保存到memory space中
        #     必须使用训练数据集中的snippet， 最后两个空间都可以初始化
        # self.inference_4_init_memory_space(anomaly_dataset, a_snippet_ids)
        print("==============init memory space==================")
        a_snippet_ids = ['203-9', '688-12']
        n_snippet_ids = ['11-4', '22-28', "22-28"]
        # todo 防止重复放入相同的记忆，判断是否已经存在该记忆
        # 在目前的记忆的基础上进行修改，还是清空已经存在的记忆？
        self.a_memory, self.a_caption_memory = self.inference_4_init_memory_space(anomaly_dataset, a_snippet_ids)
        self.n_memory, self.n_caption_memory = self.inference_4_init_memory_space(normal_dataset, n_snippet_ids)
        self.threshold_a_caption_score = args.threshold_caption_score
        self.threshold_n_caption_score = args.threshold_caption_score

    def inference_4_init_memory_space(self, dataset, snippet_ids):
        self.eval()
        memorys = []
        captions = []
        with torch.no_grad():
            for snippet_id in snippet_ids:
                # video_id, idx = snippet_id.split("-")
                # video_id, idx = int(video_id), int(idx)
                feature, caption, embedding = dataset.get_snippet_feature(snippet_id)
                feature = torch.from_numpy(feature).cuda()
                embedding = torch.from_numpy(embedding).cuda()

                # memory = self.base_model(feature, self.vars)
                memory = self.mlp_4_caption_embedding_32(embedding)

                memorys.append(memory)
                captions.append(snippet_id)
        self.train()
        return memorys, captions

    # def similarity_a_n_memory_space(self):
    #     # cosine
    #     # n_memory = torch.stack(self.n_memory)
    #     # similarity_scores = []
    #     # for a_memory in self.a_memory:
    #     #     saliency_score = a_memory.mm(torch.transpose(n_memory, 0, 1)).mean(dim=1)
    #     #     saliency_score = torch.cosine_similarity(a_memory, n_memory, dim=1).abs().mean()
    #     #     similarity_scores.append(saliency_score)
    #     # similarity_score = torch.stack(similarity_scores).mean()
    #
    #     # mm
    #     similarity_score = torch.stack(self.n_memory).mm(torch.transpose(torch.stack(self.a_memory), 0, 1)).mean()
    #
    #     return similarity_score

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

    def _calcu_saliency_score_in_memory(self, memory, cal_method=None):
        if cal_method is None:
            cal_method = self.saliency_cal_method

        if cal_method == "cos-abs":
            saliency_scores = []
            for _memory in memory:
                saliency_score = torch.cosine_similarity(_memory, memory, dim=1).abs()  # .median()
                saliency_score = torch.nn.functional.softmax(saliency_score)  # softmax
                saliency_score = saliency_score.mean()  # 也可以换成median玩玩，但结果不好
                saliency_scores.append(saliency_score)
            saliency_scores = torch.stack(saliency_scores)
        elif cal_method == "cos":
            saliency_scores = []
            for _memory in memory:
                saliency_score = torch.cosine_similarity(_memory, memory, dim=1)
                # saliency_score = (saliency_score + 1) / 2  # 帮助呢个所有saliency score只有正数, 如果本身就只有正数的话，会使得正数之间距离变为1/2
                # saliency_score = saliency_score.max()
                saliency_score = saliency_score.topk(self.topk_score).values.mean()
                saliency_scores.append(saliency_score)
            saliency_scores = torch.stack(saliency_scores)
        elif cal_method == "mul":
            saliency_scores = memory.mm(torch.transpose(memory, 0, 1)).mean(dim=1)
        elif cal_method == "mul-abs":
            saliency_scores = memory.mm(torch.transpose(memory, 0, 1)).abs().mean(dim=1)
        elif cal_method == "mse":
            # mse two
            # self.mse(memory, torch.transpose(memory, 0, 1))
            saliency_scores = []
            for memory_ in memory:
                saliency_scores.append(self.mse(memory_, memory).mean())
            saliency_scores = torch.stack(saliency_scores)
        elif cal_method == "dot":
            # todo 有问题
            _, saliency_scores = self.scaled_dot_product_attention_4_optimize_memory(memory, memory, memory)
            saliency_scores = saliency_scores.mean(dim=0)
            # saliency_scores = saliency_scores.topk(7, dim=0).values.mean(dim=0)
        else:
            raise NotImplementedError("No such saliency calculation method")

        return saliency_scores

    def reset_memory(self, args):
        self.n_memory = []
        self.n_caption_memory = []
        self.threshold_n_caption_score = args.threshold_caption_score
        self.a_memory = []
        self.a_caption_memory = []
        self.threshold_a_caption_score = args.threshold_caption_score

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
            if new_length > self.threshold_a_memory_size:  # 设置这个判断之后几乎不会被更新
                np.random.shuffle(indexes)
                indexes = indexes[:new_length]
                # todo 更新记忆空间的阈值
                self.a_memory = [self.a_memory[i] for i in indexes]
                self.a_caption_memory = [self.a_caption_memory[i] for i in indexes]
                print(f"clear a memory {old_length} -> {new_length}")

            indexes = [i for i in range(len(self.n_memory))]
            old_length = len(indexes)
            new_length = max(1, int(len(indexes) * rate))
            if new_length > self.threshold_n_memory_size:
                # todo 更新记忆空间的阈值
                np.random.shuffle(indexes)
                indexes = indexes[:new_length]
                self.n_memory = [self.n_memory[i] for i in indexes]
                self.n_caption_memory = [self.n_caption_memory[i] for i in indexes]
                print(f"clear n memory {old_length} -> {new_length}")

        print("=======================ending clear memory=======================")

    def optimize_memory(self):
        """optimize_memory"""
        a_memory = []
        a_caption_memory = []
        a_caption_embedding_memory = []
        ce_temp_4_optimize = 0.1

        if len(self.a_memory) > self.threshold_a_memory_size:
            memory = torch.stack(self.a_memory)
            caption_embedding = torch.stack(self.a_caption_embedding)
            # feat_magnitudes = torch.norm(video_embeds, p=2, dim=2)
            # self.a_memory = self.cluster_memory(memory, self.min_a_memory_size)
            # saliency_scores = self._calcu_saliency_score_in_memory(memory, cal_method="mul")  # 和所有参数的相关性的平均值，
            # saliency_scores = -saliency_scores_tmp  # 将显著度逆转
            saliency_scores_caption = self._calcu_saliency_score_in_memory(caption_embedding, cal_method="cos")
            saliency_scores = saliency_scores_caption
            #
            # if ce_temp_4_optimize == -1:
            #     saliency_scores = saliency_scores_caption
            # elif ce_temp_4_optimize == 0:
            #     pass
            # else:
            #     saliency_scores = (saliency_scores_caption * ce_temp_4_optimize + saliency_scores) / (
            #             1 + ce_temp_4_optimize)
            # 选择需要保存的memory，将memory保存然后观察结果
            saliency_indexes = torch.argsort(saliency_scores)  # 选择相似度最小，特异度最大的记忆保存
            # saliency_indexes = torch.argsort(saliency_scores, descending=True)  # 选择相似度最大，特异度最小的模型保存
            indexes = saliency_indexes[
                      self.optimize_topk:self.min_a_memory_size + self.optimize_topk]  # 挑选相似度最小的几个，让记忆的内容尽可能不相同
            print("saliency_scores", saliency_scores)
            for index in indexes:
                a_memory.append(self.a_memory[index])
                a_caption_memory.append(self.a_caption_memory[index])
                a_caption_embedding_memory.append(self.a_caption_embedding[index])

            # if ce_temp_4_optimize == -1:
            #     saliency_scores = saliency_scores_caption
            # elif ce_temp_4_optimize == 0:
            #     pass
            # else:
            #     saliency_scores = (saliency_scores_caption * ce_temp_4_optimize + saliency_scores) / (
            #             1 + ce_temp_4_optimize)

            if self.update_threshold:
                # print(saliency_scores[index])
                # saliency_scores.topk(4, largest=False).values.mean()
                # saliency_scores = self._calcu_saliency_score_in_memory(torch.stack(a_memory))
                # 基于更新后的空间，计算一个相似度
                saliency_scores = self._calcu_saliency_score_in_memory(torch.stack(a_caption_embedding_memory),
                                                                       cal_method="cos")
                threshold_ = saliency_scores.max()
                # threshold_ = saliency_scores.mean()
                # threshold_ = saliency_scores[index]
                # threshold_ = saliency_scores.topk(max(1, int(saliency_scores.shape[-1] * 0.5)), largest=True).values.mean()
                # self.threshold_a_caption_score = min(self.threshold_a_caption_score, threshold_)
                # 获得top-k小的相似度作为指标
                # self.threshold_a_caption_score = min(self.threshold_a_caption_score, saliency_scores[index])
                self.threshold_a_caption_score = threshold_

            # clustering memory, bad result
            # self.cluster_memory(torch.stack(a_memory), 3)

            print(f" {len(self.a_memory)} -> {len(indexes)}")
            self.a_memory = a_memory
            self.a_caption_memory = a_caption_memory
            self.a_caption_embedding = a_caption_embedding_memory

            # 进一步筛选掉。和所有基于正向相似的caption
            # self.optimize_ms_based_ce()
        print("threshold for a: ", self.threshold_a_caption_score)
        # print("threshold for n: ", self.threshold_n_caption_score)

        # optimize normal memory space
        # n_memory = []
        # n_caption_memory = []
        # if len(self.n_memory) > self.threshold_n_memory_size:
        #     memory = torch.stack(self.n_memory)
        #     # self.n_memory = self.cluster_memory(memory, self.min_n_memory_size)
        #
        #     # indexes = torch.argmax(memory * torch.transpose(memory, 0, 1))
        #
        #     saliency_scores = self._calcu_saliency_score_in_memory(memory)
        #
        #     saliency_indexes = torch.argsort(saliency_scores)  # 选择相抵最小，特异度最大的记忆保存
        #     # saliency_indexes = torch.argsort(saliency_scores, descending=True)
        #     indexes = saliency_indexes[self.optimize_topk:self.min_n_memory_size + self.optimize_topk]
        #     for index in indexes:
        #         n_memory.append(self.n_memory[index])
        #         n_caption_memory.append(self.n_caption_memory[index])
        #
        #     if self.update_threshold:
        #         # print(saliency_scores[index])
        #         self.threshold_n_caption_score = min(self.threshold_n_caption_score, saliency_scores[index])
        #         # saliency_scores = self._calcu_saliency_score_in_memory(torch.stack(n_memory))
        #         # threshold_ = saliency_scores.mean()
        #         # threshold_ = saliency_scores.topk(max(1, int(saliency_scores.shape[-1] * 0.5))).values.mean()
        #         # self.threshold_n_caption_score = min(self.threshold_n_caption_score, threshold_)
        #         # self.threshold_n_caption_score = threshold_
        #
        #     # n_memory = self.cluster_memory(torch.stack(n_memory), 3)
        #     print(f" {len(self.n_memory)} -> {len(indexes)}")
        #     self.n_memory = n_memory
        #     self.n_caption_memory = n_caption_memory
    def optimize_ms_based_ce(self):
        # 基于ce，再进一步改善ms的内容

        memory = torch.stack(self.a_memory)
        a_caption_memory = torch.stack(self.a_caption_embedding)
        indexes = []
        for idx, ce in enumerate(a_caption_memory):
            saliency_score = torch.cosine_similarity(ce, a_caption_memory, dim=1)
            # saliency_score = (saliency_score + 1) / 2  # 帮助呢个所有saliency score只有正数, 如果本身就只有正数的话，会使得正数之间距离变为1/2
            # saliency_score = saliency_score.max()
            # 如果和所有的记忆都是正数，就删除, 否则就保存记忆
            if (saliency_score<0).sum() > 0:
                indexes.append(idx)
        a_memory = []
        a_caption_memory = []
        a_caption_embedding_memory = []
        for index in indexes:
            a_memory.append(self.a_memory[index])
            a_caption_memory.append(self.a_caption_memory[index])
            a_caption_embedding_memory.append(self.a_caption_embedding[index])

        self.a_memory = a_memory
        self.a_caption_memory = a_caption_memory
        self.a_caption_embedding = a_caption_embedding_memory


    def show_stored_snippet_ids(self):
        return self.a_caption_memory, self.n_caption_memory

    def max_score(self, similarity_scores, top_k):
        """计算top-k的参数"""
        if top_k == 1:
            caption_score = similarity_scores.max()
        else:
            k = min(top_k, similarity_scores.shape[-1])
            topk_caption_scores = similarity_scores.topk(k=k, largest=True).values
            caption_score = topk_caption_scores.mean()
        return caption_score

    def calculate_memory_similarity(self, memory, feature, caption_embedding_memory=None, caption_embedding=None):
        """caption score越高，说明和其他的memory比较接近，没有保存的必要
        越低，说明是一个比较新的样本，可以用来扩张memory的空间"""
        cal_method = self.score_cal_method
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
        memory_ = torch.stack(memory, dim=0)
        caption_embedding_memory_ = torch.stack(caption_embedding_memory, dim=0)
        if cal_method == "cos-abs":
            # abs
            similarity_score = Variable(torch.cosine_similarity(memory_, feature), requires_grad=True).abs()
        elif cal_method == "cos":
            # no abs
            similarity_score = Variable(torch.cosine_similarity(memory_, feature), requires_grad=True)
        elif cal_method == "mul":
            similarity_score = (Variable(memory_) * feature).mean(dim=1)
        elif cal_method == "mul-abs":
            similarity_score = (Variable(memory_) * feature).abs().mean(dim=1)
        elif cal_method == "mse":  # clip的特征量会比较大，可能再加个平均会比较好
            # length_feature = feature.shape[1]
            similarity_score = Variable(self.mse(feature, memory_))
        elif cal_method == "dot":
            _, similarity_score = self.scaled_dot_product_attention_4_calculate_scores(feature, memory_, memory_)
        else:
            raise NotImplementedError("No such cal method for score calculation")

        feature_similarity_score = self.max_score(similarity_score, top_k=self.topk_score)

        # 计算caption的相似度
        caption_similarity_score = Variable(torch.cosine_similarity(caption_embedding_memory_, caption_embedding),
                                            requires_grad=True)
        #  计算各个参数被使用频率，根据使用频率决定使用的caption
        if not self.training:
            idx = torch.argmax(caption_similarity_score)
            self.used_caption_in_inference.append(self.a_caption_memory[idx])
            # print("used caption: ", self.a_caption_memory[idx])

        # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        caption_score_2 = self.max_score(caption_similarity_score, top_k=self.topk_score)
        # print(f"caption_similarity_score: {caption_score_2}, feature_similarity_score: {feature_similarity_score}")
        if self.caption_temp == -1:
            return caption_score_2
        elif self.caption_temp == 0:
            return feature_similarity_score
        else:
            # 基于视频特征量的相似度和caption的相似度，来进行计算
            return (feature_similarity_score + caption_score_2 * self.caption_temp) / (1 + self.caption_temp)

        # if len(caption_scores) == 1:
        #     return caption_scores[0]
        # else:
        #     # caption_score = torch.max(torch.tensor(caption_scores), dim=0)[1]
        #     caption_score = torch.cat(caption_scores, dim=0).max()
        #     # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        #     return caption_score

    def _add_memory(self, feature_memory, caption_id_memory, feature, idx, caption_embedding_memory=None,
                    caption_embedding=None):
        if random.random() < self.memory_rate and not idx in caption_id_memory:
            #  还没有保存过对应的记忆，并且能够随机保存数据
            feature_memory.append(feature.clone().detach())
            caption_id_memory.append(idx)
            if caption_embedding is not None:
                caption_embedding_memory.append(caption_embedding)

    def calculate_anomaly_score(self, feature, gt, update=True, idx=None, caption_embedding=None):
        if gt == -1:
            memory = self.n_memory
            caption_memory = self.n_caption_memory
            caption_embedding_memory = self.n_caption_embedding
            threshold = self.threshold_n_caption_score
        elif gt == 1:
            memory = self.a_memory
            caption_memory = self.a_caption_memory
            caption_embedding_memory = self.a_caption_embedding
            threshold = self.threshold_a_caption_score
        else:
            raise RuntimeError("No such memory")

        a_caption_score = self.calculate_memory_similarity(self.a_memory, feature, self.a_caption_embedding,
                                                           caption_embedding)
        # n_caption_score = self.calculate_feature_score(self.n_memory, caption)

        if self.training and update:
            if len(memory) > 0:
                if gt == 1:
                    feature_score = a_caption_score
                    # elif gt == -1:
                    #     feature_score = n_caption_score
                    # feature_score = self.calculate_feature_score(memory, caption)
                    # feature score越大，说明和其他的特征量越相似
                    # print("feature_score:", feature_score)
                    if threshold > feature_score:  # 大于阈值，将特征量添加到记忆空间
                        self._add_memory(memory, caption_memory, feature, idx, caption_embedding_memory,
                                         caption_embedding)
                # if len(memory) > self.threshold_n_memory_size
            else:
                self._add_memory(memory, caption_memory, feature, idx, caption_embedding_memory, caption_embedding)

        # return a_caption_score - n_caption_score
        # return (a_caption_score - n_caption_score) / (a_caption_score + n_caption_score)
        return a_caption_score
        # return a_caption_score / n_caption_score  # 效果可能不错，但是可能变成nan
        # return a_caption_score

    def add_anomaly_memory(self, x, video_ids=None, caption_embeddings=None):
        batch_size = int(len(x) / 64)
        assert len(x) % 64 == 0
        # k = self.a_topk
        snippet_idxs = []

        key_video_ids = [37, 622, 68, 588, 372, 228, 334, 262, 391, 62, 709, 344, 280, 549, 38, 369, 682, 533]

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

                #  选择异常的视频特征量，添加前k个重要的记忆
                #  选择和比较大的作为可能是异常的候选者候选者，这可能不太好
                # index_a = torch.argsort(x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1))[-self.a_topk:]
                # x[i * 2 * 32: 2 * i * 32 + 32].sum(dim=1)
                # print(index_a)

                # 获得异常的特征量, 一个视频中的异常的特征量
                batch_anomaly_feature = x[i * 2 * 32: i * 2 * 32 + 32]
                batch_normal_feature = x[i * 2 * 32 + 32: 2 * i * 32 + 64]
                # 使用world embedding作为视频的特征量
                batch_anomaly_caption = caption_embeddings[i * 2 * 32: i * 2 * 32 + 32]
                batch_normal_caption = caption_embeddings[i * 2 * 32 + 32: 2 * i * 32 + 64]
                # saliency_scores = torch.cosine_similarity(batch_anomaly_feature, batch_anomaly_feature, dim=1).abs()
                # todo 这种相似度计算方式是否合适
                # saliency_scores = torch.cosine_similarity(batch_anomaly_feature, batch_anomaly_feature.T, dim=1)

                # 计算相邻特征量之间的相似度，异常发生前后的动作往往变化比较大

                normal_scores = []
                length = 5
                num_snippets = len(batch_anomaly_feature)
                for idx, (anomaly_vf, anomaly_ce) in enumerate(zip(batch_anomaly_feature, batch_anomaly_caption)):
                    if idx in [0, num_snippets - 1]:
                        # 跳过开头和结尾的snippet
                        normal_score = torch.from_numpy(np.array(1000)).cuda()
                    else:
                        start = max(idx - length // 2, 0)
                        end = min(start + length, num_snippets)
                        # 与正常的片段越是相似，就越是normal
                        normal_score = torch.cosine_similarity(anomaly_ce, batch_normal_caption, dim=1).mean()
                        normal_score = normal_score + torch.cosine_similarity(anomaly_vf, batch_normal_feature,
                                                                              dim=1).mean()
                        # 在异常视频中，一个片段和它附近的片段的变化越小。相似度就越大，是异常的可能性就越小
                        # normal_score = normal_score + torch.cosine_similarity(anomaly_vf, batch_anomaly_feature[start:end],
                        #                                                       dim=1).mean()
                    normal_scores.append(normal_score)
                # print(normal_scores)
                normal_scores = torch.stack(normal_scores)

                # saliency_scores = torch.nn.functional.softmax(saliency_scores)  # softmax
                # index_a = torch.argsort(normal_scores)[-self.a_topk:]   # 默认从小到达排序
                # 挑选normal score最小的视频作为最有可能的异常候补
                index_a = torch.argsort(normal_scores)[:self.a_topk]  # 跳过第一个特异值，也可以考虑做一个平滑
                # saliency_score = saliency_score.mean()  # 也可以换成median玩玩，但结果不好
                # saliency_scores.append(saliency_score)

                # for idx in index_a:
                #     if idx in [0, 31]:  # 很可能出现一些log，这些log很容易被误认为异常
                #         continue
                #     else:

                indexes = i * 2 * 32 + index_a
                # features = x[indexes]
                num_snippet_per_video = 32
                for index in indexes:
                    if video_ids is None:
                        snippet_idx = None
                    else:
                        snippet_idx = f"{video_ids[i]}-{index.item() % num_snippet_per_video}"
                    if video_ids[i] in key_video_ids:
                        snippet_idxs.append(snippet_idx)
                    feature = x[index]
                    caption_embedding = caption_embeddings[index]
                    self.calculate_anomaly_score(feature, gt, update=True, idx=snippet_idx,
                                                 caption_embedding=caption_embedding)
                # 添加正常的记忆
                # index_n = torch.argmax(output1[2 * i * 32 + 32: 2 * i * 32 + 64])
                # index = 2 * i * 32 + 32 + index_n
                # caption = x[index]
                # gt = -1
                # anomaly_score = self.calculate_anomaly_score(caption, gt)
        return snippet_idxs
        # if snippet_idxs != []:
        #     print("snippet_idxs", snippet_idxs)

    def linear_video_feature(self, x, vars, for_nk=False):
        """使用线性层处理视频特征量"""

        if self.input_dim == 512:
            x = x.float()
            x = F.linear(x, vars[0], vars[1])
            # 增加一层
            x = F.relu(x)
            # x = F.softmax(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
        elif self.input_dim == 2048:
            x = x.float()
            x = F.linear(x, vars[0], vars[1])
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
        elif self.input_dim == 2560:
            x = x.float()
            x1 = x[:, :512]  # 将两种特征量分别进行处理
            x2 = x[:, 512:]
            x1 = F.linear(x1, vars[0], vars[1])
            x2 = F.linear(x2, vars[2], vars[3])
            x2 = F.relu(x2)
            x2 = F.dropout(x2, self.drop_p, training=self.training)
            x2 = F.linear(x2, vars[4], vars[5])
            x = x1 + x2
        elif self.input_dim == 1024:
            x = x.float()
            x = F.linear(x, vars[0], vars[1])
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
            if not for_nk:
                # 将32 维的向量转化为1维的
                x = F.relu(x)
                x = F.dropout(x, self.drop_p, training=self.training)
                x = F.linear(x, vars[4], vars[5])

            # x = self.classifier(x)
            # x = F.linear(x, vars[0], vars[1])
            # 增加一层
            # x = F.relu(x)
            # x = F.softmax(x)
            # x = F.dropout(x, self.drop_p, training=self.training)
        else:
            raise NotImplementedError("No such input dim")
        return x

    def linear_caption_embedding(self, caption_embedding):
        """使用线性层处理视频特征量"""
        result = self.mlp_4_caption_embedding(caption_embedding).squeeze()
        return result

    def tf_caption_embedding(self, caption_embedding):
        # 使用transformer处理caption feature
        # 总是保持一个循环，分数在0.4左右徘徊
        caption_embedding = caption_embedding.float()
        batch_size = int(caption_embedding.shape[0] / 64)
        feature_dim = caption_embedding.shape[1]
        if self.training:  # batch_size > 0:
            caption_embedding = torch.reshape(caption_embedding, (batch_size, 64, feature_dim))
            # tmp_x = x[0, 0, :]
            # plt.plot((tmp_x))
            # plt.show()
            a_x = caption_embedding[:, :32, :]  # 异常caption的embedding
            n_x = caption_embedding[:, 32:, :]  # 正常caption的embedding
            a_output = self.tf(a_x, None)  # 使用tf处理异常视频
            n_output = self.tf(n_x, None)  # 使用tf处理正常视频
            output = torch.cat([a_output, n_output], dim=1)  # 拼接输出的结果
            output_from_caption_embedding = torch.reshape(output, (int(batch_size * 64), 1))
            output_from_caption_embedding = output_from_caption_embedding.squeeze()
        else:
            # print("x.shape", x.shape)
            num_clips = caption_embedding.size(-2)
            caption_embedding = caption_embedding.squeeze()
            output = []
            # 一次获得所有snippet的异常分数, 如果snippet太多会导致内存不够用
            # output = self.tf(torch.unsqueeze(caption_embedding, dim=0))
            # 分别计算snippet的异常分数，32个32个的计算
            for i in range(0, num_clips, 32):
                output_tmp = self.tf(torch.unsqueeze(caption_embedding[i:i + 32], dim=0))
                output.append(output_tmp)
            output = torch.cat(output, dim=1)
            output_from_caption_embedding = output.squeeze()
        return output_from_caption_embedding
        from matplotlib import pyplot as plt
        plt.plot(output_from_caption_embedding.cpu().detach().numpy())
        plt.show()

    def tf_video_features(self, video_features):
        """暂时没有必要实现"""
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
        pass

    def forward(self, x, caption_embeddings=None, video_ids=None, vars=None, gt=None):
        if vars is None:
            vars = self.vars

        # feat_magnitudes = torch.norm(x, p=2, dim=2)  # 可以用于CLIP提取的参数
        # print(feat_magnitudes.shape)  # 10, 28  # 640, 28
        # x = self.to_tokens(x)
        batch_size = int(x.shape[0] / 64)
        feature_dim = x.shape[1]

        # 将caption embedding作为特征量，似乎用tf进行处理
        # output_from_caption_embedding = self.tf_caption_embedding(caption_embedding)
        # output_from_caption_embedding = self.linear_caption_embedding(caption_embedding)

        # 添加最后的激活曾，然后输出，sigmoid没有都没有什么区别，因为异常分数本身差异就很小
        # return torch.sigmoid(output_from_caption_embedding)
        # return torch.softmax(output_from_caption_embedding, dim=0)

        # 使用全连接层，将数据压缩为32位，让nk使用
        # x = self.mlp_4_caption_embedding_32(caption_embedding)

        # 将视频的embedding作为特征量来使用'
        caption_embeddings = Variable(caption_embeddings, requires_grad=True)

        x = self.linear_video_feature(x, vars, for_nk=True)

        # # return x
        # if self.training:
        #     t = x.reshape(30, 64)
        #     a = t[:, :32]
        #     n = t[:, :32]
        #     a = torch.sigmoid(a)
        #     n = torch.sigmoid(n)
        #     x = torch.cat((a,n),dim=1).reshape(1920)
        #     return x
        # else:
        #     return torch.sigmoid(x)

        if self.nk:
            if self.training:
                a_snippet_ids = self.add_anomaly_memory(x, video_ids, caption_embeddings)
            anomaly_scores = [0 for i in range(len(x))]

            indexes = np.arange(len(x))
            # np.random.shuffle(indexes)  # 防止一个视频一个视频的放入数据，导致初始空间偏向第一个视频，感觉没什么用
            for index in indexes:
                feature = x[index]
                caption_embedding = caption_embeddings[index]
                # if index < int(len(x) // 2):
                num_snippet_per_video = 32
                if self.training and video_ids is not None:
                    snippet_idx = f"{video_ids[batch_size + index // (num_snippet_per_video * 2)]}-{index % num_snippet_per_video}"
                else:
                    snippet_idx = None

                if (index // 32) % 2 == 0:  # 如果是0~31，之类的情况
                    gt = 1  # 异常的数据
                    # find corresponding snippet idx
                    anomaly_score = self.calculate_anomaly_score(feature, gt, update=False, idx=snippet_idx,
                                                                 caption_embedding=caption_embedding)
                else:
                    gt = -1  # 正常的数据
                    if self.training:
                        anomaly_score = self.calculate_anomaly_score(feature, gt, update=False, idx=snippet_idx,
                                                                     caption_embedding=caption_embedding)
                    else:
                        anomaly_score = self.calculate_anomaly_score(feature, gt, update=False,
                                                                     caption_embedding=caption_embedding)
                anomaly_scores[index] = anomaly_score
            # outputs = torch.stack(outputs, dim=0) * x.mean(dim=1)  # 0.8339315091549631
            # outputs = torch.stack(outputs, dim=0) + x.mean(dim=1) # 0.8173835234501754
            anomaly_scores = torch.stack(anomaly_scores, dim=0)
            anomaly_scores = torch.sigmoid(anomaly_scores)

            # calculate anomaly scores based on results from the linear layers
            anomaly_scores = anomaly_scores + torch.sigmoid(x.mean(dim=1))
            # x = F.linear(x, vars[4], vars[5])
            # x = torch.squeeze(x)
            # anomaly_scores = anomaly_scores + torch.sigmoid(x.min(dim=1).values)
            # anomaly_scores = anomaly_scores + torch.sigmoid(x)

            if self.training:
                return anomaly_scores, a_snippet_ids
            else:
                return anomaly_scores
            # return torch.softmax(outputs, dim=0)  # 不应该使用softmax.因为本来就有1920个snippet，这会导致分数都变得很低，除非给每一个视频进行softmax
            return torch.sigmoid(anomaly_scores)
            return torch.sigmoid(anomaly_scores + output_from_caption_embedding)
            anomaly_scores = torch.sigmoid(anomaly_scores)
            anomaly_scores = (anomaly_scores + torch.sigmoid(output_from_caption_embedding)) / 2
            return anomaly_scores

            # outputs = torch.unsqueeze(outputs, dim=1)
            # x = x.mean(dim=1)
            # x = F.sigmoid(x)
            # outputs = (outputs + x) / 2
            # return outputs
        else:
            # x = F.relu(x)
            # x = F.dropout(x, self.drop_p, training=self.training)
            # # if self.input_dim == 512:
            # #     x = F.linear(x, vars[2], vars[3])
            # # else:
            # x = F.linear(x, vars[4], vars[5])
            # x = torch.sigmoid(x)
            # a_attn.mean(dim=2) + a_output.mean(dim=2)

            anomaly_scores = x.mean(dim=1)
            # outputs += output_from_caption_embedding
            anomaly_scores = F.sigmoid(anomaly_scores)
            anomaly_scores = anomaly_scores  # + torch.sigmoid(output_from_caption_embedding)
            if self.training:
                return anomaly_scores, []
            else:
                return anomaly_scores
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
