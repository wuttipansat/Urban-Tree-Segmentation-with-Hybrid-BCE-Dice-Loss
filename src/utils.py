import torch.nn.functional as F
import torch

def smooth_bce(pred, target, eps=0.1):
    target = target * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(pred, target)


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (2. * intersection + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )

    return 1 - dice.mean()  


def loss_fn(pred, target):
    bce = smooth_bce(pred, target)
    dice = dice_loss(pred, target)

    return 0.4 * bce + 0.6 * dice 