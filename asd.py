#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 14:33
# @Author  : strawsyz
# @File    : asd.py
# @desc:


import numpy as np
from matplotlib import pyplot as plt
import torch


# def KMeans(x, K=10, Niter=10, verbose=True):
#     """Implements Lloyd's algorithm for the Euclidean metric."""
#
#     start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space
#
#     c = x[:K, :].clone()  # Simplistic initialization for the centroids
#
#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
#
#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):
#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
#
#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)
#
#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average
#
#     if verbose:  # Fancy display -----------------------------------------------
#         if use_cuda:
#             torch.cuda.synchronize()
#         end = time.time()
#         print(
#             f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
#         )
#         print(
#             "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                 Niter, end - start, Niter, (end - start) / Niter
#             )
#         )
#
#     return cl, c
def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32) #UCF(32,2048)
    r = np.linspace(0, len(feat), length+1, dtype=np.int) #(33,)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


if __name__ == '__main__':
    # t = [1,2,3,5,3]
    # res = np.argsort(t)
    # print(res)
    # print([t[idx] for idx in res])
    # root_path = rf"/workspace/datasets/ucf-crime/uio/caption_embeddings/test/normal"
    # root_path = rf"/workspace/datasets/ucf-crime/swinbert/caption_embeddings/test/normal"
    # import os
    # for filename in os.listdir(root_path):
    #     data = np.load(os.path.join(root_path, filename) , allow_pickle=True)
    #     print(data)
    # te=[1,4,5]
    # te.remove(1)
    # te.remove(0)
    # print(te)

    # npy_filepath = r"/workspace/MGFN./UCF_Train_ten_i3d/Arson053_x264_i3d.npy"
    npy_filepath = r"/workspace/datasets/XD-Violence/i3d-features/RGB/Spectre.2015__#02-11-01_02-11-29_label_B2-0-0__4.npy"

    features = np.load(npy_filepath)
    print(features.shape)

    features = features.transpose(1, 0, 2)  # [10, T, F]
    divided_features = []

    #
    # divided_mag = []
    # for feature in features:
    #     feature = process_feat(feature, 32)  # ucf(32,2048)
    #     divided_features.append(feature)
    #     divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
    # divided_features = np.array(divided_features, dtype=np.float32)
    # divided_mag = np.array(divided_mag, dtype=np.float32)
    # divided_features = np.concatenate((divided_features, divided_mag), axis=2)
    #
    # value = np.linalg.norm(feature, axis=1)
    # print(value)

    # assert  2< 1, print("asser error")
    # t = 0.235235435
    # print(f"{t:.2}")
    # test
    # data = np.load(r"/workspace/AnomalyDetection/tmp/01_001_i3d.npy",allow_pickle=True)
    # data = np.load(r"/workspace/AnomalyDetection/tmp/01_0015_i3d.npy",allow_pickle=True)
    # train
    # data = np.load(r"/workspace/AnomalyDetection/tmp/01_0014_i3d.npy", allow_pickle=True)
    # data = np.load(r"/workspace/datasets/ucf-crime/custom_dataset/train.npy", allow_pickle=True)
    # data = np.load(r"/workspace/datasets/ucf-crime/uio/32b/test/anomaly/Arrest001_x264-96.npy", allow_pickle=True)
    # data = np.load(r"/workspace/datasets/ucf-crime/swinbert/captions/test/normal/Normal_Videos_063_x264.npy", allow_pickle=True)
    # print(data)
    # print(data.shape)
    # print(data[0].shape)

    # for filename in os.listdir("/workspace/datasets/ucf-crime/custom_anno_2"):
    #     data = np.load(os.path.join(r"/workspace/datasets/ucf-crime/custom_anno_2", filename), allow_pickle=True)
    #     print(filename)
    #     # print(data)
    #     print(data.shape)
    #     print("=====================")

    #####     重要      ############
    # temp_data = [0.6796410083501576, 0.5, 0.5, 0.4989147261468929, 0.5, 0.5]
    # # temp_data = [0.3099559765322104, 0.7043553725272184, 0.7008323699944315, 0.7355491793524616, 0.7344436992589234, 0.7372540557233029, 0.725538716584849, 0.7312445723783755, 0.7343317904654451, 0.7248407706815164, 0.7288006708813064]
    # temp_data = [i*100 for i in temp_data]
    # plt.plot(temp_data)
    # plt.xlabel("Epoch")
    # plt.ylabel("AUC (%)")
    # plt.show()
    ########          重要          ###################

    # 使用mul的时候，有些分数是固定，989多个分数没有变化
    # 输出的异常文书几乎都保持着0
    # a_caption_score: tensor(11.3593, device='cuda:0') n_caption_score: tensor(24.8932, device='cuda:0')

    # t = torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1])
    # loss = torch.nn.CrossEntropyLoss()
    # ce_loss = loss(t, t)
    # print(ce_loss)
    # print(len(t))
    # print(ce_loss/len(t))
    # from torch import nn
    #
    # transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    # src = torch.rand((10, 32, 512))
    # tgt = torch.rand((20, 32, 512))
    # out = transformer_model(src, None)
    # print(out)

    # import random
    # a = 0
    # for i in range(100):
    #     if random.random() < 0.5:
    #         a+=1
    # print(a/100)

    # score_list = np.zeros(32)
    # # step = np.round(np.linspace(0, 32 // 16, 33))
    # num_frames = 32
    # # print(num_frames)
    # score_list = [i for i in range(num_frames)]
    # for idx, i in enumerate(range(0, num_frames - 15, 16)):
    #     # assert (len(score_list) >= i + 16), "len of list:" + str(len(score_list)) + ", index: " + str(i + 16)
    #     if i + 16 == len(score_list):
    #         print(score_list[i:])
    #     else:
    #         print(score_list[i:min((i + 16), len(score_list) - 1)])

    # import torch
    # import clip
    # from PIL import Image
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    #
    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    #
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
    #
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #
    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    # import time
    # import torch
    # from matplotlib import pyplot as plt
    # from pykeops.torch import LazyTensor
    #
    # use_cuda = torch.cuda.is_available()
    # dtype = torch.float32 if use_cuda else torch.float64
    # device_id = "cuda:0" if use_cuda else "cpu"
    # N, D, K = 10000, 2, 50
    # x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3
    # cl, c = KMeans(x, K)
    # # gt = np.load("gt-1120.npy")
    # # score = np.load("predict-1120.npy")
    # # plt.plot(gt)
    # # plt.plot(score)
    # # plt.show()
    #
    # # # data = np.load("soft-scores.npy")
    # # # print(data)
    # # # plt.plot(data)
    # # # plt.show()
    # # # print(torch.randperm(30).max())
    # # x =95
    # # print((x//32)%2)
    # #
    # # # 一般的梯度更新
    # # #     optimizer.zero_grad()  ## 梯度清零
    # # #     optimizer.step()  更新权重参数
    # # #     loss.backward()  # 更新权重参数
    # #
    # # # 凭什么认定输出的结果是中间帧的内容
    # #
    # # # 梯度累加， 学习率也要适当的放大
    # # #     # 2.1 loss regularization
    # # #     loss = loss/accumulation_steps
    # # #     # 2.2 back propagation
    # # #     loss.backward()
    # # #     # 3. update parameters of net
    # # #     if((i+1)%accumulation_steps)==0:
    # # #         # optimizer the net
    # # #         optimizer.step()        # update parameters of net
    # # #
    # #
    # # import torch
    # #
    # # # import vision transformer
    # #
    # # from vit_pytorch import ViT
    # # from vit_pytorch.extractor import Extractor
    # # from coca_pytorch.coca_pytorch import CoCa
    # #
    # # # init vit
    # # vit = ViT(
    # #     image_size=256,
    # #     patch_size=32,
    # #     num_classes=1000,
    # #     dim=1024,
    # #     depth=6,
    # #     heads=16,
    # #     mlp_dim=2048
    # # )
    # #
    # # # init extractor based on vit
    # # vit = Extractor(vit, return_embeddings_only=True, detach=False)
    # #
    # # # extractor will enable it so the vision transformer returns its embeddings
    # #
    # # # import CoCa and instantiate it
    # #
    # #
    # # # init coca model
    # # coca = CoCa(
    # #     dim=512,  # model dimension
    # #     img_encoder=vit,  # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    # #     image_dim=1024,  # image embedding dimension, if not the same as model dimensions
    # #     num_tokens=20000,  # number of text tokens
    # #     unimodal_depth=6,  # depth of the unimodal transformer
    # #     multimodal_depth=6,  # depth of the multimodal transformer
    # #     dim_head=64,  # dimension per attention head
    # #     heads=8,  # number of attention heads
    # #     caption_loss_weight=1.,  # weight on the autoregressive caption loss
    # #     contrastive_loss_weight=1.,  # weight on the contrastive loss between image and text CLS embeddings
    # # ).cuda()
    # #
    # # # mock text and images
    # #
    # # text = torch.randint(0, 20000, (4, 512)).cuda()
    # # images = torch.randn(4, 3, 256, 256).cuda()
    # #
    # # # train by giving CoCa your text and images with `return_loss = True`
    # #
    # # loss = coca(
    # #     text=text,
    # #     images=images,
    # #     return_loss=True  # set this to True to get the full caption + contrastive loss
    # # )
    # #
    # # loss.backward()
    # # # gsutil cp -r gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012
    # # # do the above for as much text and images...
    # # # then you can get the caption logits as so
    # #
    # # logits = coca(
    # #     text=text,
    # #     images=images
    # # )  # (4, 512, 20000)
    # #
    # # # and the CLIP-like text and image embeddings as
    # #
    # # text_embeds, image_embeds = coca(
    # #     text=text,
    # #     images=images,
    # #     return_embeddings=True
    # # )  # (4, 512), (4, 512)

    # 6105y4c5u2
    # 3D secur ： straw63831209
