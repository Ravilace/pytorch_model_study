# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:35
# @Author  : Ravilace
# @File    : Encoder.py
# @Software: PyCharm
import torch.nn as nn


from pytorch_model_study.Transformer import clones
from pytorch_model_study.Transformer.LayerNorm import LayerNorm


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """残差连接"""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """编码器由自注意力和前馈网络组成"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self.self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """架构图左侧的连接"""
        # 自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))