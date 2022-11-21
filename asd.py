#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 14:33
# @Author  : strawsyz
# @File    : asd.py
# @desc:


import numpy as np
from matplotlib import pyplot as plt
import torch

if __name__ == '__main__':
    gt = np.load("gt-1120.npy")
    score = np.load("predict-1120.npy")
    plt.plot(gt)
    plt.plot(score)
    plt.show()

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