# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 11:07
# @Author  : Ravilace
# @File    : EncoderLayer.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.LayerNorm import LayerNorm
from pytorch_model_study.Transformer.MultiHeadAttention import MultiHeadAttention
from pytorch_model_study.Transformer.PositionWiseFeedForward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. 计算自注意力
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. 残差连接 + 归一化
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. 前馈神经网络
        _x = x
        x = self.ffn(x)

        # 4. 残差连接 + 归一化
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x