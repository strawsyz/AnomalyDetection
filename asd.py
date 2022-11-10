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
    # data = np.load("soft-scores.npy")
    # print(data)
    # plt.plot(data)
    # plt.show()
    # print(torch.randperm(30).max())
    x =95
    print((x//32)%2)

    # 一般的梯度更新
    #     optimizer.zero_grad()  ## 梯度清零
    #     optimizer.step()  更新权重参数
    #     loss.backward()  # 更新权重参数

    # 凭什么认定输出的结果是中间帧的内容

    # 梯度累加， 学习率也要适当的放大
    #     # 2.1 loss regularization
    #     loss = loss/accumulation_steps
    #     # 2.2 back propagation
    #     loss.backward()
    #     # 3. update parameters of net
    #     if((i+1)%accumulation_steps)==0:
    #         # optimizer the net
    #         optimizer.step()        # update parameters of net
    #
