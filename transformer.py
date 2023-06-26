import copy
import math
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


def attention(q, k, v, d_k, mask=None, dec_mask=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if dec_mask:
            mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))
        else:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output, scores


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()

        self.size = embedding_dim

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // heads
        self.h = heads

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, mask=None, dec_mask=False):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, attn = attention(q, k, v, self.d_k, mask, dec_mask)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_dim)
        output = self.out(concat)

        return output, attn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff=2048):
        super().__init__()

        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.attn = MultiHeadAttention(heads, embedding_dim)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        x = x + attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x, enc_attn


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=200):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(-2)
        # pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        pe = Variable(self.pe[:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        # print(x.shape)
        # print(pe.shape)
        # if seq_len != 32:
        #     for i in range(0, seq_len - 32, 32):
        #         x[i:i + 32] += pe
        #     x[i + 32:] = pe[:seq_len - i - 32]  # x[seq_len % 32]
        # else:
        x = x + pe
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim, max_seq_len=10000)
        if vocab_size != embedding_dim:
            self.ln = nn.Linear(vocab_size, embedding_dim)
        else:
            self.ln = None
        self.layers = get_clones(EncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)

    def forward(self, src, mask):
        if self.ln is not None:
            src = self.ln(src)
        x = self.pe(src)
        # x = src
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)

        return x, Attn
        # return self.norm(x), Attn


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class):
        super().__init__()
        # self.feature_extractor = nn.Linear(2048, 512)
        self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid1 = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.1)
        # self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.fc1 = nn.Linear(embedding_dim, 108)
        # self.sigmoid2 = nn.Sigmoid()
        # self.fc2 = nn.Linear(108, num_class + 1)
        # self.sig = nn.Sigmoid()

    def forward(self, src, src_mask=None):
        # batch_size = int(src.shape[0] / 64)
        # if self.training:  # batch_size > 0:
        #     x = torch.reshape(src, (batch_size, 64, 32))
        #     a_x = torch.reshape(x, (batch_size, 64, 32))[:, :32, :]
        #     n_x = torch.reshape(x, (batch_size, 64, 32))[:, 32:, :]
        # else:
        #     x = src
        # src = self.feature_extractor(src)
        # shape of e_outputs : batchsize, chunksize, size of feature
        output, enc_attn = self.encoder(src, src_mask)
        # print(f"`e_outputs : {e_outputs.shape}")
        # print(f"`enc_attn : {len(enc_attn)}")
        # batch, seq, feat = e_outputs.shape
        # print(f"`e_outputs : {e_outputs.shape}")
        # print(output.shape)
        attn = torch.stack(enc_attn)
        attn = attn[-1, :, :, :]
        attn = attn.mean(dim=1)
        # attn = attn.mean(dim=2)
        # output = attn # 无法学习
        # output = attn # + output.mean(dim=2)

        # print(output)
        # print("max,", output.max())
        # print("min", output.min())
        # print("mean", output.mean())
        return attn, output
        return output.mean(dim=2), tuple(enc_attn)

        output = output.mean(dim=1)  # mean on frames in one chunk

        output = self.sigmoid(output)
        # output = self.dropout(output)
        # output_mean = self.out(output)
        # output_mean = output_mean.mean(dim=1)  # mean on frames in one chunk
        # output = self.sigmoid1(output)
        # output = self.dropout(output)
        output = self.fc1(output)
        print(output.shape)
        output_4_ml = self.sigmoid2(output)
        output = self.fc2(output_4_ml)
        print(output.shape)
        # output = self.sig(output)
        # batch, 4
        return output, output_4_ml, tuple(enc_attn)
        # return output_mean, output, tuple(enc_attn)


class TFLeaner(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, args):
        super().__init__()
        if vocab_size == 2560:
            self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
            self.encoder2 = Encoder(vocab_size, embedding_dim, N, heads)
        else:
            self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
            self.encoder2 = None

        self.input_dim = args.input_dim
        self.drop_p = args.drop
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
        self.reducer_4_tf = nn.Linear(args.embedding_dim, 32)

        # init param
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def embed_attn_output(self, attn, output):
        attn = torch.stack(attn)
        attn = attn[-1, :, :, :]
        attn = attn.mean(dim=1)

        # output = attn # 无法学习
        # output = attn.mean(dim=2) + output.mean(dim=2)
        return attn, output

        # output = output.mean(dim=2)

    def memory_stability(self):
        memory = torch.stack(self.a_memory)
        saliency_scores = self._calcu_saliency_score_in_memory(memory)
        a_stability = torch.mean(saliency_scores)
        memory = torch.stack(self.n_memory)
        saliency_scores = self._calcu_saliency_score_in_memory(memory)
        n_stability = torch.mean(saliency_scores)
        return a_stability + n_stability

    def tf(self, src, src_mask=None, gt=None):
        # normalize features
        # print(src.shape)
        # src = torch.nn.functional.normalize(src,dim=2)

        if self.encoder2 is None:
            output, attn = self.encoder(src, src_mask)
            attn, output = self.embed_attn_output(attn, output)

        else:
            src1 = src[:, :512]
            src2 = src[:, 512:]
            output, attn = self.encoder(src1, src_mask)
            output1 = self.embed_attn_output(attn, output)

            output, attn = self.encoder2(src2, src_mask)
            output2 = self.embed_attn_output(attn, output)
            raise RuntimeError("Need to be update")
            return output1 + output2

        if self.nk:
            # attn = attn.reshape(-1, 32)
            x = self.reducer_4_tf(output)
            x = x.reshape(-1, 32)
            if self.training and gt == 1:
                self.add_anomaly_memory(x, None)
            outputs = [0 for i in range(len(x))]

            indexes = np.arange(len(x))
            np.random.shuffle(indexes)  # 防止一个视频一个视频的放入数据，导致初始空间偏向第一个视频，感觉没什么用
            for index in indexes:
                caption = x[index]
                # if index < int(len(x) // 2):
                if gt == 1:  # 如果是0~31，之类的情况
                    anomaly_score = self.calculate_anomaly_score(caption, gt, update=False)
                elif gt == -1:
                    if self.training:
                        anomaly_score = self.calculate_anomaly_score(caption, gt, update=True)
                    else:
                        anomaly_score = self.calculate_anomaly_score(caption, gt, update=False)
                else:
                    anomaly_score = self.calculate_anomaly_score(caption, 1, update=False)
                outputs[index] = anomaly_score
            outputs = torch.stack(outputs, dim=0)
            outputs = torch.unsqueeze(outputs, dim=1)
            # output = torch.nn.functional.normalize(output.mean(dim=2), dim=1)
            # print("nk: ", torch.mean(outputs))
            # print("attn: ", torch.mean(attn) )
            # print("output: ",  torch.mean(output))
            # return outputs + (torch.reshape(attn.mean(dim=2), (-1, 1))
            #                   + torch.reshape(output, (-1, 1)))
            return outputs
            # return 0.01*outputs + torch.reshape(output.mean(dim=2), (-1, 1))  #  0.6487404177232166
            # return 0.1 * outputs + torch.reshape(output.mean(dim=2), (-1, 1))  # 0.6326768455348053
            # return 0.01*outputs + (torch.reshape(attn.mean(dim=2), (-1, 1)) + torch.reshape(output.mean(dim=2), (-1, 1)))
            # return outputs + torch.reshape(attn.mean(dim=2), (-1, 1))
        else:
            # output = attn.mean(dim=2)
            output = output.mean(dim=2)  # 0.8175993762866813
            # output = torch.nn.functional.normalize(output.mean(dim=2), dim=0)  # 0.7209975554922788
            # output = torch.nn.functional.normalize(output.mean(dim=2), dim=1)  # 0.7329674390747498
            # output = attn.mean(dim=2) * output.mean(dim=2)  # 0.8054970923107212
            # output = attn.mean(dim=2) + output.mean(dim=2)  #  0.804344499073478
            # output = attn.mean(dim=2) + torch.nn.functional.normalize(output.mean(dim=2), dim=1)  #  0.6958079149865213
            return output

    def forward(self, src, src_mask=None):
        x = src.float()
        batch_size = int(x.shape[0] / 64)
        feature_dim = x.shape[1]
        if self.training:  # batch_size > 0:
            x = torch.reshape(x, (batch_size, 64, feature_dim))
            # tmp_x = x[0, 0, :]
            # plt.plot((tmp_x))
            # plt.show()
            a_x = x[:, :32, :]
            n_x = x[:, 32:, :]
            a_output = self.tf(a_x, src_mask, gt=1)
            n_output = self.tf(n_x, src_mask, gt=-1)
            output = torch.cat([a_output, n_output], dim=1)
            output = torch.reshape(output, (int(batch_size * 64), 1))
            output = F.sigmoid(output)
            return output
        else:
            # print("x.shape", x.shape)
            num_clips = x.size(0)
            output = []
            for i in range(0, num_clips, 32):
                output_tmp = self.tf(torch.unsqueeze(x[i:i + 32], dim=0))
                output.append(output_tmp)
            output = torch.cat(output,dim=1)
            output = output.squeeze()
            output = F.sigmoid(output)
            return output

    def _calcu_saliency_score_in_memory(self, memory):
        # cosine abs
        saliency_scores = []
        for _memory in memory:
            saliency_score = torch.cosine_similarity(_memory, memory, dim=1).abs().mean()
            saliency_scores.append(saliency_score)
        saliency_scores = torch.stack(saliency_scores)

        # cosine no abs
        # saliency_scores = []
        # for _memory in memory:
        #     saliency_score = torch.cosine_similarity(_memory, memory, dim=1)
        #     saliency_score = (saliency_score + 1) / 2
        #     saliency_score = saliency_score.mean()
        #     saliency_scores.append(saliency_score)
        # saliency_scores = torch.stack(saliency_scores)

        # multiply two memory  # 第二版本除以模长
        # saliency_scores = memory.mm(torch.transpose(memory, 0, 1)).mean(dim=1) / (memory[:] * memory[:]).sum(dim=1)
        # saliency_scores = torch.abs(saliency_scores)
        # saliency_scores = torch.abs(memory.mm(torch.transpose(memory, 0, 1))).mean(dim=1) / (memory[:] * memory[:]).sum(
        #     dim=1)
        return saliency_scores

    def similarity_a_n_memory_space(self):
        n_memory = torch.stack(self.n_memory)
        similarity_scores = []
        for a_memory in self.a_memory:
            saliency_score = torch.cosine_similarity(a_memory, n_memory, dim=1).abs().mean()
            similarity_scores.append(saliency_score)
        similarity_score = torch.stack(similarity_scores).mean()
        return similarity_score

    def optimize_memory(self):
        """optimize_memory"""
        a_memory = []
        print(self.threshold_a_caption_score)
        print(self.threshold_n_caption_score)
        # todo
        # 只留下重要的memory
        # 让异常memory和正常memory保持一定距离
        #     计算a memory space和n memory space的相似度
        #     将相似度作为loss的一部分进行计算

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
            if self.update_threshold:
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

            if self.update_threshold:
                self.threshold_n_caption_score = min(self.threshold_n_caption_score, saliency_scores[index])

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

            # reset threhsold
            self.threshold_a_caption_score = 1
            self.threshold_n_caption_score = 1

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

        # 计算cos相似度作为距离
        # caption_scores = []
        # for in_feature in memory:
        #     in_feature = torch.unsqueeze(in_feature, dim=0)
        #     caption_scores.append(Variable(torch.cosine_similarity(in_feature, feature), requires_grad=True).abs())
        # caption_score = torch.cat(caption_scores, dim=0).max()

        # abs
        # caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
        #                          requires_grad=True).abs().max()
        # mean + abs
        caption_score = Variable(torch.cosine_similarity(torch.stack(memory, dim=0), feature),
                                 requires_grad=True).abs().mean()

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

        # 使用乘号来表示相似度
        # caption_scores = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1) / (feature * feature).mean(dim=1)
        # # caption_scores = (Variable(torch.stack(memory, dim=0)) * feature).mean(dim=1)# / (feature * feature).mean(dim=1)
        # caption_scores = torch.abs_(caption_scores)
        # if self.topk_score == 1:
        #     caption_score = caption_scores.max()
        # else:
        #     k = min(self.topk_score, len(caption_scores))
        #     topk_caption_scores = caption_scores.topk(k=k, largest=True).values
        #     caption_score = topk_caption_scores.mean()

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
                        memory.append(caption.detach())
                # if len(memory) > self.threshold_n_memory_size
            else:
                if random.random() < self.memory_rate:
                    memory.append(caption.detach())

        return a_caption_score - n_caption_score
        # return a_caption_score

    def add_anomaly_memory(self, x, output1):
        batch_size = len(x)
        # int(len(x) / 64)
        # assert len(x) % 64 == 0
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


def create_masks(batch_size, src):
    src_mask = torch.from_numpy(np.ones((batch_size, 1, src), dtype=np.float32))
    # batch, 1, sequence
    return src_mask


if __name__ == '__main__':
    model = Transformer(32, 32, 6, 8, 0)
    n_batch = 30
    n_frame = 64
    n_feature = 32
    # np_data = np.random.randn(n_batch, n_frame, n_feature)
    # src_mask = np.random.randn(n_batch, 1, n_frame)

    # src_mask = create_masks(10, 512)
    # data = torch.from_numpy(np_data)
    data = torch.randn(n_batch, n_frame, n_feature)
    src_mask = torch.randn(n_batch, 1, n_frame)
    print(data.shape)
    print(src_mask.shape)
    # out, _, _ = model(data, src_mask)
    out = model(data, None)
    print(out.shape)
