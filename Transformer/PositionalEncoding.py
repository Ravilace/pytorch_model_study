# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 10:11
# @Author  : Ravilace
# @File    : PositionalEncoding.py
# @Software: PyCharm
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """计算位置编码"""
    def __init__(self, d_model, max_len, device):
        """
        编码类构造函数
        :param d_model: 隐藏层的维度，将每个单词映射到一个d_model维的向量空间中
        :param max_len: 文本的最大长度
        :param device: 硬件，如CPU或GPU
        """
        super().__init__()
        # self.encoding会是一个max_len * d_model的二维矩阵，max_len代表词的数量，d_model代表每个单词被映射成的向量的维度
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 无需计算这里的梯度

        pos = torch.arange(0, max_len, device=device)
        # 一维展开成二维表示词的位置
        pos = pos.float().unsqueeze(dim=1)

        # i意味着词嵌入向量的内部元素编号，step代表步长
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 计算考虑词信息的位置编码
        # 实现两个需求
        # 1.同序列不同元素的位置编码不同冲突
        # 2.元素之间应该存在相对距离或顺序关系
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # 自编码
        seq_len = x.size(-1)

        # 它将添加token embedding
        return self.encoding[:seq_len, :]


def main():
    # 参数设置
    d_model = 16           # 隐藏层维度
    max_len = 10          # 最大序列长度
    seq_len = 5           # 实际序列长度
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 实例化位置编码
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)

    # 创建输入张量 [seq_len]
    # 这里我们使用随机初始化的嵌入向量，实际应用中通常是通过词嵌入层获得
    x = torch.randn(seq_len, device=device)
    # 添加位置编码
    x = pos_encoder(x)
    print(x)

if __name__ == "__main__":
    main()