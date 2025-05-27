import torch.nn as nn
import torch
import numpy as np


class Loss(nn.Module):
    def __init__(self, converter):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=converter.word2index["<PAD>"])
        self.converter = converter

    def forward(self, output, targets):
        # mask = targets[:, 0] != padding
        loss = self.criterion(output, targets.contiguous().view(-1))

        # return loss[mask].sum() / mask.sum()
        return loss

    # def arg_forward(self, output, targets):
    #     losses = []
    #     for i in range(targets.shape[1]):
    #         if targets[:, i].float().sum().item() > 0:
    #             # mask = targets[:, i] == 0
    #             # flat_mask = torch.flatten(mask)
    #             loss = self.criterion(output, targets[:, i].contiguous().view(-1))
    #             losses.append(loss)
    #     vals = torch.tensor(losses)
    #     _, idxs = torch.sort(vals)
    #     print(vals)
    #     return losses[idxs[0]], idxs[0]
