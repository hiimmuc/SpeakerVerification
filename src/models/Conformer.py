import numpy as np
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
# from models.conformer.conformer.model import Conformer
from models.conformer.conformer.encoder import ConformerEncoder
from models.layers.cnn import Conv1d
from models.layers.normalization import BatchNorm1d
from models.layers.pooling import AttentiveStatisticsPooling
from models.OnStreamAugment.specaugment import SpecAugment


class Conformer_(torch.nn.Module):
    """
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        attention_dim=128,
        **kwargs
    ):

        super().__init__()

        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_options']['augment_chain']
        self.kwargs = kwargs

        self.blocks = nn.ModuleList()

        self.specaug = SpecAugment()  # Spec augmentation

        self.instance_norm = nn.InstanceNorm1d(input_size, affine=True,
                                               eps=1e-05, momentum=0.1,
                                               track_running_stats=False)

        # Conformer
        encoder_dim = 256
        self.conformer_block = ConformerEncoder(
            input_dim=input_size,
            encoder_dim=encoder_dim,
            num_layers=6,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            input_dropout_p=0.1,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=15,
            half_step_residual=True,
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            encoder_dim,
            attention_channels=attention_dim,
            global_context=True,
        )
        self.asp_bn = BatchNorm1d(input_size=(encoder_dim * 2))
        self.attention = nn.Sequential(
            nn.Conv1d(encoder_dim, attention_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Conv1d(attention_dim, encoder_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.attention_norm = nn.BatchNorm1d(encoder_dim * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=encoder_dim * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency

        with torch.no_grad():
            if self.aug and 'spec_domain' in self.aug_chain:
                x = self.specaug(x)
            if self.kwargs['features'] == 'melspectrogram':
                x = x + 1e-6
                x = x.log()  # this will cause nan value for MFCC features
                x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instance_norm(x)

        x = x.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(x.shape[0]).to(x.device)
        lens = torch.round(lens*x.shape[1]).int()

        # conformer
        x, masks = self.conformer_block(x, lens)
        x = x.permute(0, 2, 1)
        # Attentive Statistical Pooling
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )
        x = torch.cat((mu, sg), 1)
        x = self.attention_norm(x)
        x = x.unsqueeze(1).permute(0, 2, 1)

        # Attentive Statistical Pooling
        # x = self.asp(x, lengths=lengths)
        # print(x.shape)
        # x = self.asp_bn(x)
        # print(x.shape)

        # Final linear transformation
        x = self.fc(x)
        x = x.squeeze()
        return x


def MainModel(nOut=512, **kwargs):
    model = Conformer_(lin_neurons=nOut, input_size=kwargs['n_mels'], **kwargs)
    return model
