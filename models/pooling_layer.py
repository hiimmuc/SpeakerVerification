import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1, sym=False):#, n_out:int):
        super().__init__()
           
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)      
       
        self.gamma = nn.Parameter(tensor([0.]))
        
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):
        
        
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)
                
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
          
        o = self.gamma * o + x
        
          
        return o.view(*size).contiguous()
    

class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            # unifrom initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        # NaN grad bug fixed at pytorch 0.3. Release note:
        #     `when torch.norm returned 0.0, the gradient was NaN.
        #     We now use the subgradient at 0.0, so the gradient is 0.0.`
        norm2 = torch.norm(x, 2, 1).view(-1, 1)

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        cos = self.beta * \
            torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        mask = (1. - adj) * -1e9
        masked = cos + mask

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'