# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 16:19
# @Author  : Ravilace
# @File    : TransformerEmbedding.py
# @Software: PyCharm
from torch import nn

from pytorch_model_study.Transformer.embedding.PositionalEncoding import PositionalEncoding
from pytorch_model_study.Transformer.embedding.TokenEmbedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)