import torch
from torch import nn
import torch.nn.functional as F


class CircleLossV2(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLossV2, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        print(feats.shape)
        if len(feats.shape) == 3:
            labels = labels.repeat_interleave(feats.shape[1])
            feats = feats.reshape(-1, feats.shape[-1])
        elif len(feats.shape) == 2:
            pass
        else:
            raise "Invalid shape of input"
        assert feats.size()[0] == labels.size()[0]

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(
            torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(
            torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss


if __name__ == '__main__':
    batch_size = 10
    feats = torch.rand(batch_size, 1028)
    labels = torch.randint(high=10, dtype=torch.long, size=(batch_size,))
    circleloss = CircleLoss(similarity='cos')
    print(circleloss(feats, labels))
