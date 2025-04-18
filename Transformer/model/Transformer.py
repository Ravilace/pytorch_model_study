# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 17:09
# @Author  : Ravilace
# @File    : Transformer.py
# @Software: PyCharm
import torch
from torch import nn

from pytorch_model_study.Transformer.model.Decoder import Decoder
from pytorch_model_study.Transformer.model.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_vocab_size, dec_vocab_size, d_model,
                 n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(
            d_model,
            n_head,
            max_len,
            ffn_hidden,
            enc_vocab_size,
            drop_prob,
            n_layers,
            device
        )

        self.decoder = Decoder(
            d_model,
            n_head,
            max_len,
            ffn_hidden,
            dec_vocab_size,
            drop_prob,
            n_layers,
            device
        )

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output