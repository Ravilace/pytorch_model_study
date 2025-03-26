# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:12
# @Author  : Ravilace
# @File    : EncoderDecoder.py
# @Software: PyCharm
import torch.nn as nn

class EncoderDecoder(nn.Module):
    """
    基本的编码器-解码器架构
    """
    def __init__(self, encoder, decoder, source_embedding, target_embedding, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encode(source, source_mask)
        return self.decode(memory, target, source_mask, target_mask)

    def encode(self, source, source_mask):
        embedding_source = self.source_embedding(source)
        return self.encoder(embedding_source, source_mask)

    def decode(self, memory, target, source_mask, target_mask):
        embedding_target = self.target_embedding(target)
        return self.decoder(memory, embedding_target, source_mask, target_mask)
