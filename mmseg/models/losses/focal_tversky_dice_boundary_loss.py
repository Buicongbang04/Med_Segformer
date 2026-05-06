import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .dice_loss import DiceLoss

@LOSSES.register_module()
class FocalTverskyDiceBoundaryLoss(nn.Module):
    def __init__(self,
                 alpha=0.7,
                 beta=0.3,
                 gamma=0.75,
                 lambda_dice=2.0,
                 lambda_boundary=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

        self.dice_loss = DiceLoss(loss_weight=1.0)

    def forward(self, pred, target, ignore_index=255, **kwargs):

        num_classes = pred.shape[1]

        pred_soft = torch.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        TP = torch.sum(pred_soft * target_one_hot, dims)
        FP = torch.sum(pred_soft * (1 - target_one_hot), dims)
        FN = torch.sum((1 - pred_soft) * target_one_hot, dims)

        # Focal Tversky
        tversky = (TP + 1e-5) / (
            TP + self.alpha * FN + self.beta * FP + 1e-5
        )

        loss_ft = torch.pow((1 - tversky), self.gamma).mean()

        # Dice
        loss_dice = self.dice_loss(pred, target)

        # Boundary loss (simple version)
        boundary = torch.abs(
            F.max_pool2d(target_one_hot, kernel_size=3, stride=1, padding=1)
            - target_one_hot
        )

        loss_boundary = torch.mean(pred_soft * boundary)

        total_loss = loss_ft + self.lambda_dice * loss_dice + self.lambda_boundary * loss_boundary

        return {
            'loss_ft': loss_ft,
            'loss_dice': loss_dice,
            'loss_boundary': loss_boundary,
            'loss': total_loss
        }