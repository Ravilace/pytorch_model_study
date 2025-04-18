# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 11:22
# @Author  : Ravilace
# @File    : Encoder.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.blocks.EncoderLayer import EncoderLayer
from pytorch_model_study.Transformer.embedding.TransformerEmbedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model,
            max_len,
            enc_vocab_size,
            drop_prob,
            device
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
             for _ in range(n_layers)]
        )

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
