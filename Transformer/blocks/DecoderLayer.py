# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 15:07
# @Author  : Ravilace
# @File    : DecoderLayer.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.layers.LayerNorm import LayerNorm
from pytorch_model_study.Transformer.layers.MultiHeadAttention import MultiHeadAttention
from pytorch_model_study.Transformer.layers.PositionWiseFeedForward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1.计算自注意力
        _x = dec
        x = self.masked_self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2.残差与归一化
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3.计算编码器-解码器注意力
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4.add and norm
            x = self.dropout2(x)
            x = self.norm2(x +_x)

        # 5. 逐位置的前馈网络
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x