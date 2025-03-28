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