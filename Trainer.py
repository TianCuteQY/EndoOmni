import torch
import torch.nn as nn
from util.base_trainer import BaseTrainer
from util.Loss import ScaleAndShiftInvariantLoss, MSELoss, get_losses
from util.misc import compute_metrics, AffineRescale
import torch.cuda.amp as amp
from torchvision.transforms import ColorJitter, GaussianBlur, v2, Compose
import numpy as np
import cv2, os


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
        images = batch['image'].to(self.device)

        with amp.autocast(enabled=self.config.use_amp):

            depths_gt = 1.0 / (batch['depth'] + 1e-8).to(torch.float32).to(self.device)
            
            mask = batch["mask"].to(self.device).to(torch.bool)
            nan_inds = torch.isnan(images)
            images[nan_inds] = images[nan_inds]

            out = self.model(images)
            if self.config.inverse:
                pred_depths = 1.0 / (out + 1e-8) * self.config.scale_factor
            else:
                pred_depths = out * self.config.scale_factor
            pred_depths = torch.nan_to_num(pred_depths, nan=self.config.max_depth)

            pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)

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
                            prefix="Train")

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
            pred_depths = pred_depths.squeeze()
        return pred_depths

    def validate_on_batch(self, batch, val_step, epoch):
        images = batch['image'].to(self.device)
        depth_map_gt = batch['depth'].to(self.device)
        depths_gt = 1.0 / (depth_map_gt + 1e-8)
        mask = batch["mask"].to(self.device).to(torch.bool)

        depths_gt_norm = 1.0 / (depth_map_gt + 1e-8)
        depths_gt_norm[mask] = depths_gt_norm[mask] / (depths_gt_norm[mask].max() + 1e-8)
        
        pred_depths = self.eval_infer(images)
        pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)
        if pred_depths.dim() == 4:
            pred_depths = pred_depths.squeeze(1)
        elif pred_depths.dim() == 2:
            pred_depths = pred_depths.unsqueeze(0)

        pred_depths = torch.nn.functional.interpolate(
            pred_depths.unsqueeze(1),
            size=depths_gt.shape[1:],
            mode="bicubic",
            align_corners=True).squeeze()

        # Recover scale and shift if only SSI or Silog Losses are used.

        losses = {}
        l_depth = 0.0

        with amp.autocast(enabled=self.config.use_amp):
            for k in range(len(self.config.loss_depth)):
                try:
                    l = self.loss_depth[k](pred_depths, depths_gt_norm, mask).to(self.device)
                    losses[self.config.loss_depth[k]] = l
                    l_depth += l * self.config.loss_weights[k]
                except:
                    pass

        # if "mae" not in self.config.loss_depth and "mse" not in self.config.loss_depth:
        pred_depths = self.rescale(pred_depths, depths_gt, mask)

        depths_gt[torch.logical_not(mask)] = -99
        metrics = compute_metrics(1.0 / (depths_gt + 1e-8), pred_depths, config=self.config)

        if val_step == 1 and self.should_log:
            self.log_images(rgb={"Input": images[0]},
                            depth={"GT": depths_gt[0], "PredictedMono": 1.0 / (pred_depths[0] + 1e-8)},
                            prefix="Test")
            
        # i = 0
        # for img, pred, gt in zip(images, pred_depths, depth_map_gt):
        #     i += 1
        #     img = img * 0.5 + 0.5  # Unnormalize
        #     img = img.cpu().numpy().transpose(1, 2, 0)  # Move channels to last dimension
        #     img = (img * 255).astype(np.uint8)  # Convert to uint8 for saving

        #     # Save the RGB image
        #     vis_file_name = os.path.join('./out', "epoch%03d"%epoch + "_%03d" % i + "_im.png")
        #     print("Saving ", vis_file_name)
        #     cv2.imwrite(vis_file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        #     # Save the depth image
        #     pred = pred.cpu().numpy()
        #     pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 65535
        #     depth_map_uint16 = np.clip(pred, 0, 65535).astype(np.uint16)
        #     vis_file_name = os.path.join('./out', "epoch%03d"%epoch + "_%03d" % i + "_depth.png")
        #     cv2.imwrite(vis_file_name, depth_map_uint16)
            
        #     # Save the depth image
        #     gt = gt.cpu().numpy()
        #     gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt)) * 65535
        #     depth_map_uint16 = np.clip(gt, 0, 65535).astype(np.uint16)
        #     vis_file_name = os.path.join('./out', "epoch%03d"%epoch + "_%03d" % i + "_gt.png")
        #     cv2.imwrite(vis_file_name, depth_map_uint16)

        return metrics, losses

