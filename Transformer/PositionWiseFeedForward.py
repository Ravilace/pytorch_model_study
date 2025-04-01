# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 10:50
# @Author  : Ravilace
# @File    : PositionWiseFeedForward.py
# @Software: PyCharm
from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
