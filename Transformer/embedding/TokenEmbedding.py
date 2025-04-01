# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 16:11
# @Author  : Ravilace
# @File    : TokenEmbedding.py
# @Software: PyCharm
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    使用加权矩阵对单词进行密集表示
    """
    def __init__(self, vocab_size, d_model):
        """
        包含位置信息的token嵌入
        :param vocab_size: 词汇量size of vocabulary
        :param d_model: 词向量维度
        """
        super().__init__(vocab_size, d_model, padding_idx=1)