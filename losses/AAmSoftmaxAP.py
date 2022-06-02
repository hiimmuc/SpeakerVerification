#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.AAmSoftmax as aamsoftmax
import losses.AngularProto as angleproto
import losses.Softmax as softmax


class AAmSoftmaxAP(nn.Module):

    def __init__(self, **kwargs):
        super(AAmSoftmaxAP, self).__init__()

        self.test_normalize = True

        self.aamsoftmax = aamsoftmax.AAmSoftmax(**kwargs)
        self.angleproto = angleproto.AngularProto(**kwargs)
        self.softmax = softmax.Softmax(**kwargs)

    def forward(self, x, label=None):

        assert x.size()[1] == 2 # 2 sub senters

        nlossAAm, prec1 = self.aamsoftmax(x, label)

        nlossAP, _ = self.angleproto(x, label)

        nlossSm, prec2 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        return sum([nlossAAm, nlossAP, nlossSm])/3 , prec2