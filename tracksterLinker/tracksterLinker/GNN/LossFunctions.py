
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets, weights):
        """Binary focal loss, mean.

        Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
        improvements for alpha.
        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        ce_loss = F.binary_cross_entropy(predictions, targets, reduction='none', weight=weights)
        p_t = torch.exp(-ce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = (alpha_tensor * (1 - p_t) ** self.gamma * ce_loss).mean()
        return f_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))
        return loss_contrastive


class CombinedLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4, margin=0.3, weightFocal=0.6, weightContrastive=0.4):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.contrastive = ContrastiveLoss(margin=margin)

        self.weightFocal = weightFocal
        self.weightContrastive = weightContrastive

    def forward(self, predictions, embeddings, emb_dupl, targets, label, weights):
        return self.weightFocal * self.focal(predictions, targets, weights) + self.weightContrastive * self.contrastive(embeddings, emb_dupl, label)
