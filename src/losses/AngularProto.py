#! /usr/bin/python
# -*- encoding: utf-8 -*-

import pdb
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


class AngularProto(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(AngularProto, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        device = x.get_device()

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(
            out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda(device=device)
        nloss = self.criterion(cos_sim_matrix, label)
        prec1 = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1