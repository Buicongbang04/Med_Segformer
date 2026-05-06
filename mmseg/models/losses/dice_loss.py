import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight

    def forward(self, pred, target, ignore_index=255, **kwargs):
        pred = F.softmax(pred, dim=1)

        valid_mask = target != ignore_index

        pred_fg = pred[:, 1, :, :]
        target_fg = (target == 1).float()

        pred_fg = pred_fg * valid_mask.float()
        target_fg = target_fg * valid_mask.float()

        intersection = (pred_fg * target_fg).sum(dim=(1, 2))
        union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        return self.loss_weight * loss.mean()