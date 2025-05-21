import torch.nn as nn
import torch
import numpy as np


class Loss(nn.Module):
    def __init__(self, converter):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.converter = converter

    def min_sequence_loss(self, output, target, prediction, input, group):
        # pred_logits: (seq_len, vocab_size)
        # valid_targets: list of sequences [ [y1_1, y1_2,...], [y2_1,...], ... ]
        losses = []
        for target_seq in group:
            if (target_seq != 0):
                loss = self.criterion(output, self.converter.word2index[target_seq])
                losses.append(loss)
        return min(losses)

    def forward(self, output, targets):
        losses = []
        for i in range(targets.shape[1]):
            if targets[:, i].float().sum().item() > 0:
                # mask = targets[:, i] == 0
                # flat_mask = torch.flatten(mask)
                loss = self.criterion(output, targets[:, i].contiguous().view(-1))
                losses.append(loss)
        return min(losses)
        # if (prediction <= 3):
        #     # print("prediction <= 3")
        #     return self.criterion(output, target)

        # # print(f"{prediction}, {input}, {group}")
        # if (3 in input):
        #     # print("two groups")
        #     if (prediction in np.split(input, np.nonzero(input == 3)[0])[-1]):
        #         # print("two groups in input")
        #         return self.criterion(output, target)
        # else:
        #     if (prediction in input):
        #         # print("one group in input")
        #         return self.criterion(output, target)

        # if (int(self.converter.index2word[prediction]) not in group):
        #     # print("not in group")
        #     return self.criterion(output, target)

        # # print("in group")
        # target[-1] = int(self.converter.index2word[prediction])
        # return self.criterion(output, target)
