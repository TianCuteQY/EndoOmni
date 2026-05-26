import torch
import torch.nn as nn
from util.base_trainer import BaseTrainer
from util.Loss import ScaleAndShiftInvariantLoss, MSELoss, get_losses
from util.misc import compute_metrics, AffineRescale
import torch.cuda.amp as amp
from torchvision.transforms import ColorJitter, GaussianBlur, v2, Compose
import numpy as np


def get_device(model):
    if model is not None:
        return next(model.parameters()).device
    else:
        return None


class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader=None, test_loader=None, device=None):
        super().__init__(config, model, train_loader=train_loader,
                         test_loader=test_loader, device=device)
        self.device = get_device(model)
        self.config = config
        self.loss_depth = get_losses(config)
        self.scale_factor = config.scale_factor
        self.scale, self.shift = None, None
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.rescale = AffineRescale(depth_cap=config.max_depth)

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """
        images = batch['image'].to(torch.float32).to(self.device)
        depths_gt = batch['depth'].to(torch.float32).to(self.device)
        mask = batch["mask"].to(self.device).to(torch.bool)

        with amp.autocast(enabled=self.config.use_amp):

            out = self.model(images)
            if self.config.inverse:
                pred_depths = 1.0 / (out + 1e-8) * self.config.scale_factor
            else:
                pred_depths = out * self.config.scale_factor
            pred_depths = torch.nan_to_num(pred_depths, nan=self.config.max_depth)
            pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)

            pred_depths = torch.nn.functional.interpolate(
                            pred_depths.unsqueeze(1),
                            size=depths_gt.shape[1:],
                            mode='bilinear', align_corners=True).squeeze()
            
            pred_depths = pred_depths * depths_gt[mask].median() / pred_depths[mask].median()

            losses = {}
            loss = 0
            for k in range(len(self.config.loss_depth)):
                l = self.loss_depth[k](pred_depths, depths_gt, mask)
                losses[self.config.loss_depth[k]] = l
                loss += l.to(self.device) * self.config.loss_weights[k]

        if torch.isnan(loss):
            print("input is nan", torch.isnan(images).any())
            print('gt is nan', torch.isnan(depths_gt).any())
            print('pred is nan', torch.isnan(pred_depths).any())

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        # self.optimizer.step()
        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...]},
                            depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]},
                            prefix="Train", min_depth=self.config.min_depth, max_depth=self.config.max_depth)

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses

    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):

            out = self.model(x)
            if self.config.inverse:
                pred_depths = 1.0 / (out + 1e-8) * self.config.scale_factor
            else:
                pred_depths = out * self.config.scale_factor
        return pred_depths

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(torch.float32).to(self.device)
        mask = batch["mask"].to(self.device).to(torch.bool)

        pred_depths = self.eval_infer(images)
        pred_depths = torch.nan_to_num(pred_depths, nan=self.config.max_depth)
        pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)

        pred_depths = torch.nn.functional.interpolate(
            pred_depths.unsqueeze(1),
            size=depths_gt.shape[1:],
            mode='bilinear', align_corners=True).squeeze()

        pred_depths = pred_depths * depths_gt[mask].median() / (pred_depths[mask].median() + 1e-8)

        losses = {}
        l_depth = 0.0

        with amp.autocast(enabled=self.config.use_amp):
            for k in range(len(self.config.loss_depth)):
                try:
                    l = self.loss_depth[k](pred_depths, depths_gt, mask).to(self.device)
                    losses[self.config.loss_depth[k]] = l
                    l_depth += l * self.config.loss_weights[k]
                except:
                    pass

        metrics = compute_metrics(depths_gt, pred_depths, config=self.config)

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]},
                            depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]},
                            prefix="Test", min_depth=self.config.min_depth, max_depth=self.config.max_depth)

        return metrics, losses
