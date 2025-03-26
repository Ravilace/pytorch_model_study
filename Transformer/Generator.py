# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:24
# @Author  : Ravilace
# @File    : Generator.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """定义生成器，由Linear和Softmax组成"""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)