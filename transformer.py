import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        # print(x.shape)
        # print(pe.shape)
        x = x + pe
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim, max_seq_len=32)
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
        attn = attn.mean(dim=1).mean(dim=2)
        # output = attn # 无法学习
        # print(attn.mean())
        # print(output.mean())
        output = attn + output.mean(dim=2)

        # print(output)
        # print("max,", output.max())
        # print("min", output.min())
        # print("mean", output.mean())
        return output
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
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class):
        super().__init__()
        if vocab_size == 2560:
            self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
            self.encoder2 = Encoder(vocab_size, embedding_dim, N, heads)
        else:
            self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
            self.encoder2 = None
        # self.sigmoid = nn.Sigmoid()
        # self.fc1 = nn.Linear(embedding_dim, 108)
        # self.sigmoid2 = nn.Sigmoid()
        # self.fc2 = nn.Linear(108, num_class + 1)

        # init param
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def embed_attn_output(self, attn, output):
        attn = torch.stack(attn)
        attn = attn[-1, :, :, :]
        attn = attn.mean(dim=1).mean(dim=2)

        # output = attn # 无法学习
        output = attn + output.mean(dim=2)
        return output

        # output = output.mean(dim=2)

        # print(output)
        # print("max,", output.max())
        # print("min", output.min())
        # print("mean", output.mean())

    def tf(self, src, src_mask=None):

        if self.encoder2 is None:
            output, attn = self.encoder(src, src_mask)
            output = self.embed_attn_output(attn, output)
            return output
        else:
            src1 = src[:, :512]
            src2 = src[:, 512:]
            output, attn = self.encoder(src1, src_mask)
            output1 = self.embed_attn_output(attn, output)

            output, attn = self.encoder2(src2, src_mask)
            output2 = self.embed_attn_output(attn, output)
            return output1 + output2
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

    def forward(self, src, src_mask=None):
        x = src.float()
        batch_size = int(x.shape[0] / 64)
        feature_dim = x.shape[1]
        if self.training:  # batch_size > 0:
            x = torch.reshape(x, (batch_size, 64, feature_dim))
            a_x = x[:, :32, :]
            n_x = x[:, 32:, :]
            a_output = self.tf(a_x, src_mask)
            n_output = self.tf(n_x, src_mask)
            output = torch.cat([a_output, n_output], dim=1)
            output = torch.reshape(output, (int(batch_size * 64), 1))
            output = F.sigmoid(output)
            # output = F.relu(output)
            return output
        else:
            # print("x.shape", x.shape)
            output = self.tf(x)
            output = output.squeeze()
            output = F.sigmoid(output)
            # output = F.relu(output)
            return output


# class Transformer(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, N, heads, num_class):
#         super().__init__()
#         # self.feature_extractor = nn.Linear(2048, 512)
#         self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
#         self.sigmoid = nn.Sigmoid()
#         self.out = nn.Linear(embedding_dim, num_class + 1)
#         self.dropout = nn.Dropout(p=0.1)
#
#     def forward(self, src, src_mask):
#         # src = self.feature_extractor(src)
#         # shape of e_outputs : batchsize, chunksize, size of feature
#         output, enc_attn = self.encoder(src, src_mask)
#         output = output.mean(dim=1)  # mean on frames in one chunk
#         output = self.sigmoid(output)
#         output = self.dropout(output)
#         output = self.out(output)
#         # batch, 4
#         return output, None, tuple(enc_attn)


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
