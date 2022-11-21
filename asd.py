#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 14:33
# @Author  : strawsyz
# @File    : asd.py
# @desc:


import numpy as np
from matplotlib import pyplot as plt
import torch
def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

if __name__ == '__main__':
    import time
    import torch
    from matplotlib import pyplot as plt
    from pykeops.torch import LazyTensor

    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64
    device_id = "cuda:0" if use_cuda else "cpu"
    N, D, K = 10000, 2, 50
    x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3
    cl, c = KMeans(x, K)
    # gt = np.load("gt-1120.npy")
    # score = np.load("predict-1120.npy")
    # plt.plot(gt)
    # plt.plot(score)
    # plt.show()

    # # data = np.load("soft-scores.npy")
    # # print(data)
    # # plt.plot(data)
    # # plt.show()
    # # print(torch.randperm(30).max())
    # x =95
    # print((x//32)%2)
    #
    # # 一般的梯度更新
    # #     optimizer.zero_grad()  ## 梯度清零
    # #     optimizer.step()  更新权重参数
    # #     loss.backward()  # 更新权重参数
    #
    # # 凭什么认定输出的结果是中间帧的内容
    #
    # # 梯度累加， 学习率也要适当的放大
    # #     # 2.1 loss regularization
    # #     loss = loss/accumulation_steps
    # #     # 2.2 back propagation
    # #     loss.backward()
    # #     # 3. update parameters of net
    # #     if((i+1)%accumulation_steps)==0:
    # #         # optimizer the net
    # #         optimizer.step()        # update parameters of net
    # #
    #
    # import torch
    #
    # # import vision transformer
    #
    # from vit_pytorch import ViT
    # from vit_pytorch.extractor import Extractor
    # from coca_pytorch.coca_pytorch import CoCa
    #
    # # init vit
    # vit = ViT(
    #     image_size=256,
    #     patch_size=32,
    #     num_classes=1000,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048
    # )
    #
    # # init extractor based on vit
    # vit = Extractor(vit, return_embeddings_only=True, detach=False)
    #
    # # extractor will enable it so the vision transformer returns its embeddings
    #
    # # import CoCa and instantiate it
    #
    #
    # # init coca model
    # coca = CoCa(
    #     dim=512,  # model dimension
    #     img_encoder=vit,  # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    #     image_dim=1024,  # image embedding dimension, if not the same as model dimensions
    #     num_tokens=20000,  # number of text tokens
    #     unimodal_depth=6,  # depth of the unimodal transformer
    #     multimodal_depth=6,  # depth of the multimodal transformer
    #     dim_head=64,  # dimension per attention head
    #     heads=8,  # number of attention heads
    #     caption_loss_weight=1.,  # weight on the autoregressive caption loss
    #     contrastive_loss_weight=1.,  # weight on the contrastive loss between image and text CLS embeddings
    # ).cuda()
    #
    # # mock text and images
    #
    # text = torch.randint(0, 20000, (4, 512)).cuda()
    # images = torch.randn(4, 3, 256, 256).cuda()
    #
    # # train by giving CoCa your text and images with `return_loss = True`
    #
    # loss = coca(
    #     text=text,
    #     images=images,
    #     return_loss=True  # set this to True to get the full caption + contrastive loss
    # )
    #
    # loss.backward()
    # # gsutil cp -r gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012
    # # do the above for as much text and images...
    # # then you can get the caption logits as so
    #
    # logits = coca(
    #     text=text,
    #     images=images
    # )  # (4, 512, 20000)
    #
    # # and the CLIP-like text and image embeddings as
    #
    # text_embeds, image_embeds = coca(
    #     text=text,
    #     images=images,
    #     return_embeddings=True
    # )  # (4, 512), (4, 512)