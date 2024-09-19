import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import torch.nn.functional as F

KEY_OUTPUT = 'metric_depth'


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
    valid = (det > 0).nonzero()

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


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()
        self.name = 'MSE'

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


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

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.name = 'SSI'

    def forward(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(prediction)
        # preprocessing
        valid_mask = (target > 0) & mask

        # calcul
        scale, shift = compute_scale_and_shift(prediction.to(torch.float32), target.to(torch.float32),
                                               valid_mask.to(torch.float32))
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, valid_mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, valid_mask)
        # if torch.isnan(total):
        #     raise ValueError("Loss is NaN, Stopping training")
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""

    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x ** 2 + diff_y ** 2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""

    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
               nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""

    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth)
        depth = depth.long()
        return depth

    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, input, target, mask=None):
        """
        Calculate the masked L1 loss.

        Parameters:
        - input: Predicted tensor.
        - target: Ground truth tensor.
        - mask: Boolean tensor with the same shape as input and target,
                where True/1 indicates the positions that contribute to the loss.

        Returns:
        - The calculated masked L1 loss.
        """
        if mask is None:
            mask = np.ones_like(input)
        assert input.shape == target.shape == mask.shape, "Input, target, and mask must have the same shape"

        loss = torch.abs(input - target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()


def normalize_disparity(depth):
    # Transform depth to disparity and normalize to [0, 1]
    disp = 1.0 / torch.clamp(depth, min=1e-6)  # Avoid division by zero
    norm_disp = (disp - disp.min()) / (disp.max() - disp.min())
    return norm_disp


def compute_scale_and_shift_affine(depth, mask):
    # Flatten the tensor and only consider the masked regions
    depth_valid = depth[mask]

    # Compute the median which serves as the shift to zero-center the data
    t = torch.median(depth_valid)

    # Compute the scale factor so that the sum of absolute deviations is 1
    s = 1.0 / (torch.sum(torch.abs(depth_valid - t)) / depth_valid.size(
        0) + 1e-6)  # Add epsilon to avoid division by zero

    return t, s


class affine_invariant_loss(nn.Module):
    def __init__(self, trim_ratio=0.2):
        super().__init__()
        self.trim_ratio = trim_ratio
        self.name = 'AFFINE'

    def ssitrim_loss(self, pred, gt, mask, trim_ratio=0.0):
        """
        Compute the scale-shift invariant trimmed loss.
        """
        # Flatten the tensors and apply the mask
        pred_valid = pred[mask]
        gt_valid = gt[mask]

        # Compute the absolute residuals and sort them
        residuals = torch.abs(pred_valid - gt_valid)
        sorted_residuals, _ = torch.sort(residuals)

        # Determine the cutoff index for the trimming
        trim_index = int(len(sorted_residuals) * (1 - trim_ratio))

        # Compute the trimmed loss
        trimmed_residuals = sorted_residuals[:trim_index]
        loss = trimmed_residuals.mean()

        return loss

    def forward(self, pred, gt, mask=None):
        """
        Compute the affine-invariant mean absolute error loss.
        """
        # Normalize prediction and ground truth disparities
        # pred = normalize_disparity(pred)
        # gt = normalize_disparity(gt)

        # Compute the median and scale for both prediction and ground truth
        t_pred, s_pred = compute_scale_and_shift_affine(pred, mask)
        t_gt, s_gt = compute_scale_and_shift_affine(gt, mask)

        # Normalize the predictions and ground truth using their respective scales and shifts
        pred_normalized = s_pred * (pred - t_pred)
        gt_normalized = s_gt * (gt - t_gt)

        # Calculate the mean absolute error only over the mask
        loss = self.ssitrim_loss(pred_normalized, gt_normalized, mask, self.trim_ratio)
        # loss = torch.abs(pred_normalized - gt_normalized)[mask].mean()

        return loss


def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    with amp.autocast(enabled=False):
        M = torch.sum(mask, (1, 2))
        res = prediction - target

        res = res[mask.bool()].abs()

        # trimmed, _ = torch.sort(res.view(-1), descending=False)[
        #              : int(len(res) * (1.0 - trim))
        #              ]
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        trimmed = trimmed[: int(len(res.view(-1)) * (1.0 - trim))]

        # ret = trimmed.sum() / (2 * M.sum() + 1e-8)
        # if torch.isnan(ret):
        #     a = trimmed.sum()
        #     b = (2 * M.sum() + 1e-8)
        #     print("Affine Loss is Nan!")
        #     total = torch.Tensor([0])

        return trimmed.sum() / (2 * M.sum() + 1e-8)


class TrimmedMAELoss(nn.Module):
    def __init__(self, reduction='batch-based', trim=0.2):
        super().__init__()
        self.trim = trim

    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask, self.trim)


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def normalize_prediction_robust(target, mask):
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


class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        prediction = prediction.to(torch.float32)
        target = target.to(torch.float32)
        mask = mask.to(torch.float32)
        with amp.autocast(enabled=False):
            self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
            target_ = normalize_prediction_robust(target, mask)
            if self.__prediction_ssi is None or target_ is None:
                return torch.Tensor([0])

            total = self.__data_loss(self.__prediction_ssi, target_, mask)
            if self.__alpha > 0:
                total += self.__alpha * self.__regularization_loss(
                    self.__prediction_ssi, target_, mask
                )
        # if torch.isnan(total):
        #     print("Affine Loss is Nan!")
        #     total = torch.Tensor([0])

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class MSEGradientLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(MSEGradientLoss, self).__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        prediction = prediction.to(torch.float32)
        target = target.to(torch.float32)
        mask = mask.to(torch.float32)
        with amp.autocast(enabled=False):
            self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
            target_ = normalize_prediction_robust(target, mask)
            if self.__prediction_ssi is None or target_ is None:
                return torch.Tensor([0])

            total = self.__data_loss(self.__prediction_ssi, target_, mask)
            if self.__alpha > 0:
                total += self.__alpha * self.__regularization_loss(
                    self.__prediction_ssi, target_, mask
                )
        # if torch.isnan(total):
        #     print("Affine Loss is Nan!")
        #     total = torch.Tensor([0])

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, alpha, valid_mask=True, scales=4, reduction="batch-based", loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

    def sigloss(self, input, target, mask):
        if self.valid_mask:
            valid_mask = target > 0 & mask
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, mask):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt, mask)

        if self.__alpha > 0:
                loss_depth += self.__alpha * self.__regularization_loss(
                    depth_pred, depth_gt, mask
                )
        return loss_depth


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # if target.ndim == 3:
        #     target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            # Dg = torch.var(g, unbiased=False) + self.beta * torch.pow(torch.mean(g), 2)
            if g.numel() > 1:  # ensure there are more than one data points
                Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
            else:
                Dg = torch.tensor(0.0, device=g.device)


            loss = 10 * torch.sqrt(Dg)

        if not return_interpolated:
            return loss

        return loss, intr_input


def get_losses(config):
    loss_depth = []
    type = config.type
    if type == "full" or type == "depth":
        for l in config.loss_depth:
            if l == 'Trim':
                loss_depth.append(TrimmedProcrustesLoss(alpha=0.5))
            elif l == 'MAE':
                loss_depth.append(TrimmedProcrustesLoss(alpha=0., trim=0))
            elif l == 'MSE':
                loss_depth.append(MSEGradientLoss(alpha=0.5))
            elif l == 'SigLoss':
                loss_depth.append(SigLoss(alpha=0))
            elif l == 'SILogLoss':
                loss_depth.append(SILogLoss())
    return loss_depth
