import torch
import torch.nn as nn
import torch.nn.functional as F

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.class_weights.to(logits.device))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha.to(logits.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes=5):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # Implementation for ordinal loss
        # Convert targets to ordinal encoding (e.g. class 2 -> [1, 1, 0, 0])
        batch_size = targets.size(0)
        levels = torch.arange(self.num_classes - 1).to(targets.device).expand(batch_size, self.num_classes - 1)
        target_levels = (targets.view(-1, 1) > levels).float()
        
        # Logits should have dim num_classes - 1 for this specific implementation,
        # but if using standard num_classes logits, we can just use CE with a penalty.
        # A simpler ordinal penalty: MSE on predictions
        preds = torch.argmax(logits, dim=1).float()
        mse_loss = F.mse_loss(preds, targets.float())
        ce_loss = F.cross_entropy(logits, targets)
        return ce_loss + 0.5 * mse_loss

def get_loss_function(name, class_weights, num_classes):
    if name == 'cross_entropy':
        return WeightedCrossEntropyLoss(class_weights)
    elif name == 'focal':
        return FocalLoss(alpha=class_weights)
    elif name == 'ordinal':
        return OrdinalRegressionLoss(num_classes)
    else:
        raise ValueError(f"Unknown loss function {name}")
