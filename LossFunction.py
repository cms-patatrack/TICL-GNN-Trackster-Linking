import torch.nn as nn
import torch
import numpy as np


class Loss(nn.Module):
    def __init__(self, converter, vocab_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Loss, self).__init__()
        
        weights = torch.ones(vocab_size)
        weights[converter.word2index[";"]] = 0.5
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(device), ignore_index=converter.word2index["<PAD>"])

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
