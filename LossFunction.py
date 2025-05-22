import torch.nn as nn
import torch
import numpy as np


class Loss(nn.Module):
    def __init__(self, converter):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.converter = converter

    def forward(self, output, targets):
        losses = []
        for i in range(targets.shape[1]):
            if targets[:, i].float().sum().item() > 0:
                # mask = targets[:, i] == 0
                # flat_mask = torch.flatten(mask)
                loss = self.criterion(output, targets[:, i].contiguous().view(-1))
                losses.append(loss)
        return min(losses)
