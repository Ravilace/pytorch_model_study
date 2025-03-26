# -*- coding: utf-8 -*-
# @Time    : 2025/3/25 15:46
# @Author  : Ravilace
# @File    : __init__.py.py
# @Software: PyCharm
import copy

import torch.nn as nn

def clones(module, N):
    """产生N个相同的module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])