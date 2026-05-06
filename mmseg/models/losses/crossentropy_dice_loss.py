import torch.nn as nn
from ..builder import LOSSES, build_loss
from .dice_loss import DiceLoss 

@LOSSES.register_module()
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self,
                 ce_cfg=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 dice_weight=1.0, **kwargs):
        super().__init__()
        
        self.ce_loss = build_loss(ce_cfg)
        self.dice_loss = DiceLoss(loss_weight=1.0)
        self.dice_weight = dice_weight

    def forward(self,
                pred,
                target,
                weight=None,
                ignore_index=255,
                **kwargs):

        # Cross Entropy
        loss_ce = self.ce_loss(
            pred,
            target,
            weight=weight,
            ignore_index=ignore_index
        )

        # Dice
        loss_dice = self.dice_loss(
            pred,
            target,
            ignore_index=ignore_index
        )

        total_loss = loss_ce + self.dice_weight * loss_dice

        return {
            'loss_ce': loss_ce,
            'loss_dice': loss_dice,
            'loss': total_loss
        }