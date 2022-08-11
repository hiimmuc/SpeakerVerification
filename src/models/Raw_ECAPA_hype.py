import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ECAPA_TDNN, RawNet2_custom


class Raw_ECAPA(nn.Module):
    """
    Refactored RawNet2 combined with ECAPA architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143)
    """

    def __init__(self, nOut=512, **kwargs):
        super(Raw_ECAPA, self).__init__()
        self.context_dim = 512 + 192

        self.ECAPA_TDNN = ECAPA_TDNN.MainModel(nOut=192,
                                               channels=[
                                                   512, 512, 512, 512, 1536],
                                               input_norm=True, **kwargs)
        self.rawnet2v2 = RawNet2_custom.MainModel(nOut=512,
                                                  front_proc='sinc',  aggregate='gru',
                                                  att_dim=128, **kwargs)

        features = 'melspectrogram'
        Features_extractor = importlib.import_module(
            'models.FeatureExtraction.feature').__getattribute__(f"{features}")
        self.compute_features = Features_extractor(**kwargs)

        att_size = 128
        self.bn_before_agg = nn.BatchNorm1d(self.context_dim)
        self.attention = nn.Sequential(
            nn.Conv1d(self.context_dim, att_size, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm1d(att_size),
            nn.Conv1d(att_size, self.context_dim, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.bn_final = nn.BatchNorm1d(self.context_dim * 2)
        self.fc = nn.Linear(self.context_dim * 2, nOut)
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        #####

        # #####
        # # forward model 1
        # #####

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x_spec = self.compute_features(x)

        out1 = self.ECAPA_TDNN(x_spec)

        # #####
        # # forward model 2
        # #####
        out2 = self.rawnet2v2(x)
        #
        out = torch.cat([out1, out2], dim=-1)

        out = self.bn_before_agg(out)

        out = self.lrelu(out)

        out = out.unsqueeze(-1)  # bs, nOut -> bs, nOut, 1  to perform conv1

        w = self.attention(out)

        m = torch.sum(out * w, dim=-1)  # mean
        s = torch.sqrt(
            (torch.sum((out ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-9))  # standard
        out = torch.cat([m, s], dim=1)
        out = out.view(out.size(0), -1)

        #####
        # speaker embedding layer
        #####
        out = self.bn_final(out)
        out = self.fc(out)
        out = out.squeeze()
        return out


def MainModel(nOut=512, **kwargs):
    model = Raw_ECAPA(nOut=nOut, **kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
