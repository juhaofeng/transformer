import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable  # 这个东西已经被淘汰，其实就是torch.Tersor
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        # 定义Embedding层
        self.luf = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.luf(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])

emb = Embeddings(d_model , vocab) #d_model : 词表的大小 vocab： 词嵌入空间向量的大小
embr = emb(x)
print(embr.shape)
