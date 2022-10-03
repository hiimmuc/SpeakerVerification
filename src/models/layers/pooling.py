import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ECAPA_TDNN import TDNNBlock
from models.layers.utils import length_to_mask
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def new_parameter(*size):
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):

    def __init__(self, embedding_size):

        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.att = new_parameter(self.embedding_size, 1)

    def forward(self, ht):
        attention_score = torch.matmul(ht, self.att).squeeze()
        attention_score = F.softmax(
            attention_score, dim=-1).view(ht.size(0), ht.size(1), 1)
        ct = torch.sum(ht * attention_score, dim=1)

        return ct, attention_score


class HeadAttention(nn.Module):

    def __init__(self, encoder_size, heads_number, mask_prob=0.25, attentionSmoothing=False):

        super(HeadAttention, self).__init__()
        self.embedding_size = encoder_size//heads_number
        self.att = new_parameter(self.embedding_size, 1)
        self.mask_prob = int(1/mask_prob)
        self.attentionSmoothing = attentionSmoothing

    def __maskAttention(self, attention_score, mask_value=-float('inf')):

        mask = torch.cuda.FloatTensor(
            attention_score.size()).random_(self.mask_prob) > 0
        attention_score[~mask] = mask_value
        return attention_score

    def __narrowAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score)
        attention_score = F.softmax(
            attention_score, dim=-1).view(new_ht.size(0), new_ht.size(1), 1)
        return attention_score

    def __wideAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(
                attention_score, mask_value=-1)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1)
        return attention_score.view(new_ht.size(0), new_ht.size(1), 1)

    def forward(self, ht):

        if self.attentionSmoothing:
            attention_score = self.__wideAttention(ht)
        else:
            attention_score = self.__narrowAttention(ht)

        weighted_ht = ht * attention_score
        ct = torch.sum(weighted_ht, dim=1)

        return ct, attention_score


def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k),
                            dim1=-2, dim2=-1).view(value.size(0), value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim=-2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_size, heads_number):
        super(MultiHeadAttention, self).__init__()
        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0  # d_model
        self.head_size = self.encoder_size // heads_number
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None

    def getAlignments(self, ht):
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size, -1, self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(
            self.query, key, value)
        return self.alignment

    def getHeadsContextVectors(self, ht):
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size, -1, self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(
            self.query, key, value)
        return headsContextVectors

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)
        return headsContextVectors.view(headsContextVectors.size(0), -1), copy.copy(self.alignment)


class DoubleMHA(nn.Module):
    def __init__(self, encoder_size, heads_number, mask_prob=0.2):
        super(DoubleMHA, self).__init__()
        self.heads_number = heads_number
        self.utteranceAttention = MultiHeadAttention(
            encoder_size, heads_number)
        self.heads_size = encoder_size // heads_number
        self.headsAttention = HeadAttention(
            encoder_size, heads_number, mask_prob=mask_prob, attentionSmoothing=False)

    def getAlignments(self, x):

        utteranceRepresentation, alignment = self.utteranceAttention(x)
        headAlignments = self.headsAttention(utteranceRepresentation.view(
            utteranceRepresentation.size(0), self.heads_number, self.heads_size))[1]
        return alignment, headAlignments

    def forward(self, x):
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        compressedRepresentation = self.headsAttention(utteranceRepresentation.view(
            utteranceRepresentation.size(0), self.heads_number, self.heads_size))[0]
        return compressedRepresentation, alignment


class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [
            int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor(
                (w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(
                math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor(
                (h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(
                math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                h_pad1 + h_pad2 == (h_kernel *
                                    levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError(
                    "Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """
        Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [
            int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            #
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(math.floor(
                (w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(
                math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + \
                w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError(
                    "Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class TemporalPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
        Temporal Pyramid Pooling Module, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

        :returns (forward) a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        super(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
        Calculates the output shape given a filter_amount: sum(filter_amount*level) for each level in levels
        Can be used to x.view(-1, tpp.get_output_size(filter_amount)) for the fully-connected layers
        :param filters: the amount of filter of output fed into the temporal pyramid pooling
        :return: sum(filter_amount*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level
        return out


# ==============================================pooling==================================================


class StatisticsPooling(torch.nn.Module):
    """
    Mean and Standard deviation pooling
    """

    def __init__(self):
        """

        """
        super(StatisticsPooling, self).__init__()
        pass

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)


class GlobalAveragePooling(torch.nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)

    # ==========> code <===========
    def forward(self, x):
        '''
        input: (64, 1500, 86)
        output: (64, 1500)
        '''
        pass


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context

        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)

        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)
        return pooled_stats
