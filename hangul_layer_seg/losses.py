from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiLabelLayerLoss(nn.Module):
    def __init__(self, dice_loss_weight: float = 0.0, no_final_t_fp_penalty: float = 0.0) -> None:
        super().__init__()
        self.dice_loss_weight = dice_loss_weight
        self.no_final_t_fp_penalty = no_final_t_fp_penalty

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        no_final_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weights = valid.expand_as(bce)
        loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
        if self.no_final_t_fp_penalty > 0 and no_final_t is not None and no_final_t.any():
            t_probs = torch.sigmoid(logits[:, 2:3])
            t_valid = valid * no_final_t.to(dtype=valid.dtype, device=valid.device)[:, None, None, None]
            loss = loss + self.no_final_t_fp_penalty * (t_probs * t_valid).sum() / t_valid.sum().clamp_min(1.0)
        if self.dice_loss_weight <= 0:
            return loss
        return loss + self.dice_loss_weight * multilabel_dice_loss(logits, target, valid)


def multilabel_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dims)
    denominator = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()
