# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:39
# @Author  : Ravilace
# @File    : LayerNorm.py
# @Software: PyCharm
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """用于处理残差进行归一化"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b