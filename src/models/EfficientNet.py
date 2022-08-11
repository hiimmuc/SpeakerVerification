import math

import torch
import torch.nn as nn

from models.Efficient_utils import (DropConnect, Flatten, SamePadConv2d,
                                    SEModule, conv_bn_act)
from models.OnStreamAugment.specaugment import SpecAugment


class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(
            in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)
                           ) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        # self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride,
                         skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1,
                          skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 in_channels=1, aggregate='flatten',
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 nOut=1000, **kwargs):
        super().__init__()

        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_options']['augment_chain']
        self.n_mels = kwargs['n_mels']
        self.kwargs = kwargs
        self.aggregate = aggregate

        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) //
                        depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = conv_bn_act(in_channels, renew_ch(
            32), kernel_size=3, stride=2, bias=False)

        self.blocks = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1,
                    renew_repeat(1), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2,
                    renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2,
                    renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2,
                    renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1,
                    renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2,
                    renew_repeat(4), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1,
                    renew_repeat(1), True, 0.25, drop_connect_rate)
        )
        # attention
        self.specaug = SpecAugment()
        self.instancenorm = nn.InstanceNorm1d(self.n_mels)
        outmap_size = int(self.n_mels / 8)
        att_dim = 128
        self.attention = nn.Sequential(
            nn.Conv1d(renew_ch(1280), att_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(att_dim),
            nn.Conv1d(
                att_dim, renew_ch(1280), kernel_size=1),
            nn.Softmax(dim=2),
        )

        ####
        self.head = nn.Sequential(
            *conv_bn_act(renew_ch(320), renew_ch(1280),
                         kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(
                dropout_rate, True) if dropout_rate > 0 else nn.Identity()
        )
        self.fc = nn.Linear(renew_ch(1280), nOut)
        self.fc_attn = nn.Linear(renew_ch(1280) * 2, nOut)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, x):
        with torch.no_grad():
            if self.kwargs['features'] == 'melspectrogram':
                x = x + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instancenorm(x).unsqueeze(1)

        # batch x channel x n_mels x n_frames
        assert len(x.size()) == 4, f"got {x.size()}"

        stem = self.stem(x)
        x = self.blocks(stem)
        x = self.head(x)

        ##
        if self.aggregate == 'flatten':
            x = Flatten()(x)
            out = self.fc(x)
            return out

        elif self.aggregate == "SAP":
            x = torch.mean(x, dim=2, keepdim=True)
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
            x = x.view(x.size()[0], -1)
        elif self.aggregate == "ASP":
            x = x.reshape(x.size()[0], -1, x.size()[-1])
            w = self.attention(x)
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt(
                (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)
            x = x.view(x.size()[0], -1)

        out = self.fc_attn(x)
        return out


efficient_net_version_params = {
    'b0': [1.0, 1.0, 0.2],
    'b1': [1.0, 1.1, 0.2],
    'b2': [1.1, 1.2, 0.3],
    'b3': [1.2, 1.4, 0.3],
    'b4': [1.4, 1.8, 0.4],
    'b5': [1.6, 2.2, 0.4],
    'b6': [1.8, 2.6, 0.5],
    'b7': [2.0, 3.1, 0.5],
}


def MainModel(nOut=512, **kwargs):
    # Number of filters
    version = 'b4'
    aggregate = 'ASP'
    width_coeff, depth_coeff, dropout_rate = efficient_net_version_params[version]
    model = EfficientNet(width_coeff, depth_coeff,
                         aggregate=aggregate,
                         depth_div=8, min_depth=None,
                         dropout_rate=dropout_rate, drop_connect_rate=0.2,
                         nOut=nOut, **kwargs)
    return model


if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = MainModel(nOut=512)
    from torchsummary import summary
    summary(net.cuda(), (3, 224, 224))
