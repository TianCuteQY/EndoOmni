import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
from util.Loss import GradientLoss


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def normalize_prediction_robust(target, mask):
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    # Only proceed with the computation if there are valid entries
    if valid.any():
        m[valid] = torch.median(
            (mask[valid] * target[valid]).view(valid.sum().item(), -1), dim=1
        ).values
        target = target - m.view(-1, 1, 1)

        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    else:
        print("Warning: No valid entries in mask. Skipping normalization.")
        return None

    return target / (s.view(-1, 1, 1))


class WeightedMAELoss(nn.Module):
    def __init__(self):
        super(WeightedMAELoss, self).__init__()

    def __call__(self, preds, targets, weights, mask):
        loss = torch.abs(preds - targets)
        weighted_loss = loss * weights * mask

        if torch.isnan(weighted_loss).any():
            # a = torch.isnan(preds).any()
            # b = torch.isnan(targets).any()
            # c = torch.isnan(weights).any()
            # d = torch.isnan(mask).any()
            raise ValueError("Loss, Stopping training")
        return torch.mean(weighted_loss)
        # return torch.sum(weighted_loss) / torch.sum(weights * mask)


class WeightedGradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()
        self.scales = scales
        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, weight, mask):
        total = 0
        for scale in range(self.scales):
            step = pow(2, scale)
            scaled_prediction = prediction[:, ::step, ::step]
            scaled_target = target[:, ::step, ::step]
            scaled_mask = mask[:, ::step, ::step]
            scaled_weight = weight[:, ::step, ::step] if weight is not None else None

            total += self.gradient_loss(scaled_prediction, scaled_target, scaled_mask, scaled_weight)

        if torch.isnan(total):
            raise ValueError("Loss, Stopping training")

        return total

    def gradient_loss(self, prediction, target, mask, weight):
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        if weight is not None:
            # Apply weights to the gradient losses
            grad_x = torch.mul(weight[:, :, 1:] * weight[:, :, :-1], grad_x)
            grad_y = torch.mul(weight[:, 1:, :] * weight[:, :-1, :], grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

        if weight is not None:
            M = torch.sum(weight * mask, (1, 2))
        else:
            M = torch.sum(mask, (1, 2))

        return self.reduction(image_loss, M)


class WeightedAffineLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(WeightedAffineLoss, self).__init__()

        self.__data_loss = WeightedMAELoss()
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask, weight=None):
        if torch.isnan(mask).any():
            raise ValueError("mask, Stopping training")
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)
        if self.__prediction_ssi is None or target_ is None:
            return torch.Tensor([0])
        
        if weight is None:
            weight = torch.ones_like(mask)

        # a = torch.isnan(prediction).any()
        # b = torch.isnan(target).any()
        # c = torch.isnan(weight).any()
        # d = torch.isnan(mask).any()

        total = self.__data_loss(self.__prediction_ssi, target_, weight, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )
        if torch.isnan(total):
            raise ValueError("Loss, Stopping training")
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
