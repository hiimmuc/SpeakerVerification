import torch
from torch import nn
from torch.autograd import Variable

from math import sqrt

class AdaptiveSoftmax(nn.Module):
    def __init__(self, input_size, cutoff):
        super(AdaptiveSoftmax, self).__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        
        for i in range(1, len(cutoff)):
            seq = nn.Sequential(
                nn.Linear(input_size, input_size // 4 ** i, False),
                nn.Linear(input_size // 4 ** i, cutoff[i] - cutoff[i - 1], False)
            )

            self.tail.append(seq)

    def reset(self):
        std = 0.1

        nn.init.xavier_normal(self.head.weight)

        for tail in self.tail:
            nn.init.xavier_normal(tail[0].weight)
            nn.init.xavier_normal(tail[1].weight)

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.sum() > 0:
                self.id.append(Variable(mask.float().nonzero().squeeze(1)))

            else:
                self.id.append(None)

    def forward(self, input):
        output = [self.head(input)]

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](input.index_select(0, self.id[i])))

            else:
                output.append(None)

        return output

    def log_prob(self, input):
        lsm = nn.LogSoftmax()#.cuda()

        head_out = self.head(input)

        batch_size = head_out.size(0)
        prob = torch.zeros(batch_size, self.cutoff[-1])#.cuda()

        lsm_head = lsm(head_out)
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)

        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = lsm(self.tail[i](input))
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

class AdaptiveLoss(nn.Module):
    def __init__(self, cutoff):
        super(AdaptiveLoss, self).__init__()

        self.cutoff = cutoff
        self.criterions = nn.ModuleList()

        for i in self.cutoff:
            self.criterions.append(nn.CrossEntropyLoss(size_average=False))

    def remap_target(self, target):
        new_target = [target.clone()]

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.sum() > 0:
                new_target.append(target[mask].add(-self.cutoff[i]))

            else:
                new_target.append(None)

        return new_target

    def forward(self, input, target):
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)

        output = 0.0

        for i in range(len(input)):
            if i is not None:
                assert(target[i].min() >= 0 and target[i].max() <= input[i].size(1))
                criterion = self.criterions[i]
                output += criterion(input[i], Variable(target[i]))

        output /= batch_size

        return output