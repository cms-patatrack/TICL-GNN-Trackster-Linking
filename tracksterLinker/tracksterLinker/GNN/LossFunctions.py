
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossLogits(nn.Module):
    """
    Binary focal loss operating on *logits* for numerical stability.
    Supports per-sample weights (same shape as targets).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.4, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        # Ensure floats
        targets = targets.float()
        logits = logits.float()

        # Base BCE in logit space (stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [N]

        # p_t = sigmoid(logit) if y=1 else 1 - sigmoid(logit)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # [N] in (0,1)

        # alpha_t = alpha if y=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # modulator
        mod = (1.0 - p_t).clamp(min=0.0, max=1.0) ** self.gamma

        loss = alpha_t * mod * bce  # [N]

        if weights is not None:
            # Make sure weights are finite and >=0
            weights = torch.clamp(weights.float(), min=0.0)
            # Detach so no gradients flow through your weight computation
            weights = weights.detach()
            # Weighted mean with safe denominator
            denom = weights.sum().clamp_min(self.eps)
            return (loss * weights).sum() / denom

        return loss.mean()

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
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 0 = similar, 1 = dissimilar (as in your code)
        e1 = F.normalize(output1, dim=-1)
        e2 = F.normalize(output2, dim=-1)
        d = F.pairwise_distance(e1, e2)  # [N], in [0,2]
        loss = ((1 - label) * d.pow(2) + label * (torch.clamp(self.margin - d, min=0.0).pow(2))).mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4, margin=0.3, weightFocal=0.6, weightContrastive=0.4):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLossLogits(gamma=gamma, alpha=alpha)
        self.contrastive = ContrastiveLoss(margin=margin)

        self.weightFocal = weightFocal
        self.weightContrastive = weightContrastive

    def forward(self, predictions, embeddings, emb_dupl, targets, label, weights):
        return self.weightFocal * self.focal(predictions, targets, weights) + self.weightContrastive * self.contrastive(embeddings, emb_dupl, label)
