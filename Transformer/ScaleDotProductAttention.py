# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 15:41
# @Author  : Ravilace
# @File    : ScaleDotProductAttention.py
# @Software: PyCharm
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    计算缩放点乘注意力
    Query: 用户输入的句子 (decoder)
    Key: 需要和Query进行查询关系的每一条句子 (encoder)
    Value: 上面句子代表的实际信息 (encoder)
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 输入是一个四维的张量
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. 点乘Query和Key^T来计算相似度
        k_t = k.transpose(2, 3) # 转置
        score = (q @ k_t) / math.sqrt(d_tensor)  # 缩放点乘

        # 2. 应用掩码 (可选地）
        if mask is not None:
            score = score.masked_fill(mask== 0, -10000)

        # 3. 将score传给softmax使得范围在[0, 1]
        score = self.softmax(score)

        # 4. 再和Value点乘
        v = score @ v
        return v, score