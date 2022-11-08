import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()

        self.n_memory = []
        self.a_memory = []
        self.n_captions = []
        self.a_captions = []

        # 多级memory，根据不同的layer层次存储不同的memory

        self.threshold_caption_score = 0.6 # 越大，需要记忆的memory就越多，loss会有一点点减少，auc能有一点的提升

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def calculate_caption_score(self, memory, caption):
        """caption score越高，说明和其他的memory比较接近，没有保存的必要
        越低，说明是一个比较新的样本，可以用来扩张memory的空间"""

        # cap 特征量不是小数无法直接用于计算相似度
        # cap_feature = cap.cpu().numpy()
        # 可以生成caption，但是必须转化为list
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = self.ss.encode(memory + [caption])
        caption_scores = []
        for i in range(len(memory)):
            caption_scores.append(cosine_similarity([embeddings[i]], [embeddings[-1]]))

        caption_score = torch.mean(torch.tensor(caption_scores))
        # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        return caption_score

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

        caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
                                 requires_grad=True).abs().max()
        # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        return caption_score

        # if len(caption_scores) == 1:
        #     return caption_scores[0]
        # else:
        #     # caption_score = torch.max(torch.tensor(caption_scores), dim=0)[1]
        #     caption_score = torch.cat(caption_scores, dim=0).max()
        #     # 找到最相似的memory的距离，作为这个caption的特异性,这个数字越小，说明和这个caption相似的memory越少，也说明越有保存的价值
        #     return caption_score

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def calculate_anomaly_score(self, caption, gt):
        if gt == -1:
            memory = self.n_memory
        else:
            memory = self.a_memory

        if self.training:
            if len(memory) > 1:
                caption_score = self.calculate_feature_score(memory, caption)
                if self.threshold_caption_score > caption_score:
                    memory.append(caption)
            else:
                # caption_score = self.calculate_caption_score(memory, caption)
                memory.append(caption)
                # captions.append()

        a_caption_score = self.calculate_feature_score(self.a_memory, caption)
        n_caption_score = self.calculate_feature_score(self.n_memory, caption)
        return a_caption_score - n_caption_score

    def forward(self, x, vars=None, gt=None):
        if vars is None:
            vars = self.vars
        # 不通过模型，直接使用特征量计算，loss一直在2以上 # AUC: 0.34005
        x = F.linear(x, vars[0], vars[1])  # 好像改善了一点
        x = F.relu(x)  # 改善了0.02
        x = F.dropout(x, self.drop_p, training=self.training)  # 改善了不少  #0.5191871161161531
        x = F.linear(x, vars[2], vars[3])  # AUC 是0.468左右
        x = F.dropout(x, self.drop_p, training=self.training)  # 0.44033766356249904
        # 层数越深，需要记忆的模式就越多，AUC就越高，直接使用memory反而会对结果产生不好的影响
        # loss 也有微妙的下降的趋势，但是auc只在0.5的附近徘徊
        # 层数越深，需要记忆的模式越多，但是每个记忆的数量非常的少
        # 记忆的长度有个合适的值，太短会导致视频的大部分信息的丢失，太长会导致特征基本都比较相似，需要记忆的内容就比较小
        # 层数越深，特征之间的相似度就越小，特征量越短，
        # 模型太多比较的次数可能会太多

        # 由于没有使用预训练的模型，导致初期生成的特征量没有参考性，结果就不好

        outputs = [0 for i in range(len(x))]

        indexes = np.arange(len(x))
        np.random.shuffle(indexes)
        for index in indexes:
            caption = x[index]
            if index < int(len(x) // 2):
                gt = 1
            else:
                gt = -1
            anomaly_score = self.calculate_anomaly_score(caption, gt)
            # outputs.append(anomaly_score)
            outputs[index] = anomaly_score

        return torch.stack(outputs, dim=0)


        x = F.linear(x, vars[4], vars[5])

        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
