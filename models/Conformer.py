import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

from models.ECAPA_TDNN import *
from models.ECAPA_utils import Conv1d as _Conv1d
from models.ECAPA_utils import BatchNorm1d as _BatchNorm1d
from models.conformer.conformer.model import Conformer

from utils import PreEmphasis
from nnAudio import features



class Conformer_(torch.nn.Module):
    """
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.GELU,
        attention_channels=128,
        global_context=True,
        **kwargs
    ):

        super().__init__()

        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_options']['augment_chain']        
        self.kwargs = kwargs
        
        self.blocks = nn.ModuleList()
        
        self.specaug = SpecAugment() # Spec augmentation
        
        self.instance_norm = nn.InstanceNorm1d(input_size, affine=True, 
                                               eps=1e-05, momentum=0.1, 
                                               track_running_stats=False)

        # Conformer
        self.conformer_block = Conformer(
            input_dim = input_size,
            encoder_dim = 256,
            num_attention_heads = 4,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            input_dropout_p = 0.1,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size = 15,
            half_step_residual = True,
            num_classes= lin_neurons,   
            num_encoder_layers= 6,
        )
        
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            lin_neurons,
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=(lin_neurons * 2))

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=lin_neurons * 2,
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
                x = x.log() # this will cause nan value for MFCC features
                x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instance_norm(x)

        # conformer
        x = x.transpose(1, -1)
        x = self.conformer_block(x)        
        x = x.transpose(1, -1)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)
        x = x.squeeze()       
        return x
    
def MainModel(nOut=512, **kwargs):
    model = Conformer_(lin_neurons=nOut, **kwargs)
    return model