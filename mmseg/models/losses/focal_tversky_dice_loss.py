import torch 
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class FocalTverskyDiceLoss(nn.Module):
    def __init__(self,
                 alpha=0.7,
                 beta=0.3,
                 gamma=0.75,
                 smooth=1e-5,
                 loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        pred: (N, C, H, W)
        target: (N, H, W)
        """

        num_classes = pred.shape[1]

        # convert target -> one-hot
        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred = torch.softmax(pred, dim=1)

        dims = (0, 2, 3)

        TP = torch.sum(pred * target_one_hot, dims)
        FP = torch.sum(pred * (1 - target_one_hot), dims)
        FN = torch.sum((1 - pred) * target_one_hot, dims)

        # Tversky Index
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )

        # Focal Tversky Loss
        focal_tversky = torch.pow((1 - tversky), self.gamma).mean()

        # Dice Loss
        dice = (2 * TP + self.smooth) / (
            2 * TP + FP + FN + self.smooth
        )
        dice_loss = (1 - dice).mean()

        total_loss = focal_tversky + 2.0 * dice_loss
        total_loss = total_loss * self.loss_weight

        return {
            'loss_focal_tversky': focal_tversky,
            'loss_dice': dice_loss,
            'loss': total_loss
        }