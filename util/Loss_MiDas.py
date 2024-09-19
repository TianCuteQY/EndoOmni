'''
combine:
- https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2
- https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
'''
from math import e
import torch
import torch.nn as nn


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


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


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
                 : int(len(res) * (1.0 - trim))
                 ]

    return trimmed.sum() / (2 * M.sum())


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
    mask_x = torch.mul(mask[:, :, :, 1:], mask[:, :, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
    mask_y = torch.mul(mask[:, :, 1:, :], mask[:, :, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (2, 3)) + torch.sum(grad_y, (2, 3))
    image_loss = torch.mean(image_loss)

    return reduction(image_loss, M)


def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class TrimmedMAELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        # assert len(prediction.shape)==len(target.shape)==4
        if len(prediction.shape) == 3:
            prediction = prediction.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        elif len(mask.shape) != 4:
            raise (RuntimeError("[GradientLoss] len(mask) has to be in [3, 4]! Got %d" % len(mask.shape)))

        assert len(mask.shape) == 4
        assert mask.shape[-2:] == prediction.shape[-2:] == target.shape[-2:]

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, :, ::step, ::step], target[:, :, ::step, ::step],
                                   mask[:, :, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based', loss_method='MSELoss', if_scale_aware=False):
        super().__init__()

        self.__loss_method = loss_method
        assert self.__loss_method in ['TrimmedMAELoss', 'MSELoss', 'L1loss']

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.if_scale_aware = if_scale_aware

        # self.__prediction_ssi = None

    def forward(self, prediction, target, mask=None):
        assert len(prediction.shape) == 3
        assert len(target.shape) == 3
        assert len(mask.shape) == 3
        assert mask.dtype == torch.float32
        if mask is None:
            mask = torch.ones_like(target, dtype=target.dtype).cuda()

        # print(prediction.shape, torch.mean(prediction), torch.median(prediction), torch.max(prediction), torch.min(prediction))
        # print(target.shape, torch.mean(target), torch.median(target), torch.max(target), torch.min(target))
        # print(mask.shape, torch.mean(mask), torch.median(mask), torch.max(mask), torch.min(mask))

        if self.__loss_method == 'MSELoss':
            if self.if_scale_aware:
                scale, shift = compute_scale_and_shift(prediction, target, mask)
                # print(scale)
                # print(shift)
                __prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            else:
                __prediction_ssi = prediction
            total = self.__data_loss(__prediction_ssi, target, mask)
        elif self.__loss_method == 'TrimmedMAELoss':
            assert not self.if_scale_aware
            __prediction_ssi = normalize_prediction_robust(prediction, mask)
            target_ = normalize_prediction_robust(target, mask)
            total = self.__data_loss(__prediction_ssi, target_, mask)

        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(__prediction_ssi, target, mask)

        return total

    # def __get_prediction_ssi(self):
    #     return self.__prediction_ssi

    # prediction_ssi = property(__get_prediction_ssi)


def get_losses(config):
    loss_depth = []
    type = config.type
    if type == "full" or type == "depth":
        for l in config.loss_depth:
            if l == 'ssi':
                loss_depth.append(ScaleAndShiftInvariantLoss())
    return loss_depth
