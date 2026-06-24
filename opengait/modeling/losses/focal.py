import torch
import torch.nn.functional as F

from .base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, scale=2**4, gamma=2.0, label_smooth=False, eps=0.1,
                 loss_term_weight=1.0, log_accuracy=False, weight=None, alpha=None):
        super(FocalLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.gamma = gamma
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

        class_weight = alpha if alpha is not None else weight
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
        else:
            self.class_weight = None

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float() * self.scale
        labels = labels.long().unsqueeze(1)

        flat_logits = logits.permute(0, 2, 1).reshape(-1, c)
        flat_labels = labels.repeat(1, p).reshape(-1)

        class_weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
        ce = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            weight=class_weight,
            label_smoothing=self.eps if self.label_smooth else 0.0,
        )
        focal = ((1.0 - torch.exp(-ce)) ** self.gamma) * ce
        loss = focal.mean()

        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info

