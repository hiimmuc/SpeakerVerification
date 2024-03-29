#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.AngularProto as angleproto
import losses.Softmax as softmax


class SoftmaxAngularProto(nn.Module):

    def __init__(self, **kwargs):
        super(SoftmaxAngularProto, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.Softmax(**kwargs)
        self.angleproto = angleproto.AngularProto(**kwargs)

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.angleproto(x, None)

        return nlossS+nlossP, prec1
