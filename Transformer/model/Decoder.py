# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 16:55
# @Author  : Ravilace
# @File    : Decoder.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.blocks.DecoderLayer import DecoderLayer
from pytorch_model_study.Transformer.embedding.TransformerEmbedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model,
            drop_prob,
            max_len,
            dec_vocab_size,
            device
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
             for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output = self.self.linear(trg)
        return output
