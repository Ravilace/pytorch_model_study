# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 15:39
# @Author  : Ravilace
# @File    : MultiHeadAttention.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.layers.ScaleDotProductAttention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):
        """
        依照n_head的数量来分割张量
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

    def concat(self, tensor):
        """
        分割的逆操作
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def forward(self, q, k, v, mask=None):
        # 1.和权重矩阵点乘
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2.依照n_head的数量来分割张量
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3.通过缩放点乘计算相似度
        out, attention = self.attention(q, k, v, mask=mask)

        # 4.连接并传递给线性层
        out = self.concat(out)
        out = self.w_concat(out)

        return out
