import torch
import torch.nn as nn
from util.base_trainer import BaseTrainer
from util.Loss import ScaleAndShiftInvariantLoss, MSELoss, get_losses
from util.misc import compute_metrics, AffineRescale
import torch.cuda.amp as amp
from torchvision.transforms import ColorJitter, GaussianBlur, v2, Compose
import numpy as np
from aug.get_aug import Aug


def get_device(model):
    if model is not None:
        return next(model.parameters()).device
    else:
        return None


class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader=None, test_loader=None, device=None,
                 teacher_model=None, dino_model=None):
        super().__init__(config, model, train_loader=train_loader,
                         test_loader=test_loader, device=device)
        self.device = get_device(model)
        self.teacher_device, self.dino_device = get_device(teacher_model), get_device(dino_model)
        self.config = config
        self.loss_depth = get_losses(config)
        self.scale_factor = config.scale_factor
        self.scale, self.shift = None, None
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.student_transform = Aug(config)
        self.with_dino = dino_model is not None
        self.dinov2 = dino_model
        self.embedding_loss = nn.CosineEmbeddingLoss(margin=0.15, reduction='mean')
        self.rescale = AffineRescale(depth_cap=config.max_depth)

    @torch.no_grad()
    def get_dino_features(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            return self.dinov2.get_intermediate_layers(x.to(self.dino_device), 1, return_class_token=True)[0]

    @torch.no_grad()
    def get_pseudo_label(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            out = self.teacher_model(x.to(self.teacher_device))
            return out

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """
        images = batch['image'].to(self.device)
        # images_aug = self.student_transform((batch['image'] + 1)/2)
        # images_aug = (images_aug * 2 - 1).to(self.device)

        with amp.autocast(enabled=self.config.use_amp):

            no_label_inds = [i for i in range(batch['depth'].shape[0]) if batch['dataset'][i] == "unlabeled"]

            depths_gt = 1.0 / (batch['depth'] + 1e-8).to(torch.float32).to(self.device)
            depths_gt[no_label_inds] = self.get_pseudo_label(batch['image'][no_label_inds]).to(torch.float32).to(self.device)

            mask = batch["mask"].to(self.device).to(torch.bool)
            if torch.any(mask):
                C = depths_gt.shape[0]
                channel_maxs = (depths_gt * mask).view(C, -1).max(dim=1)[0].view(C, 1, 1).expand_as(
                    depths_gt)
                depths_gt[mask] = depths_gt[mask] / (channel_maxs[mask] + 1e-8)

            images_aug = self.student_transform(images, depths_gt, batch['dataset']).to(self.device)
            nan_inds = torch.isnan(images_aug)
            images_aug[nan_inds] = images[nan_inds]

            r = torch.rand(1)
            if self.student_transform.with_cutmix and r < self.student_transform.cutmix_prob:
                # generate mixed sample

                lam = torch.distributions.Beta(1.0, 1.0).sample()
                rand_index = torch.randperm(images.size(0)).to(self.device)

                target_a = depths_gt
                target_b = depths_gt[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(images_aug.size(), lam)
                images_aug[:, :, bbx1:bbx2, bby1:bby2] = images_aug[rand_index, :, bbx1:bbx2, bby1:bby2]
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images_aug.size()[-1] * images_aug.size()[-2]))

                # Compute Mask based on cutmix and original dataset
                mask_a = torch.zeros_like(mask).to(self.device)
                mask_a[:, bbx1:bbx2, bby1:bby2] = 1
                mask_a = mask_a.to(torch.bool) & mask
                mask_b = torch.ones_like(mask).to(self.device)
                mask_b[:, bbx1:bbx2, bby1:bby2] = 0
                mask_b = mask_b.to(torch.bool) & mask[rand_index]

                # compute output
                out = self.model(images_aug)
                if self.config.inverse:
                    pred_depths = 1.0 / (out + 1e-8) * self.config.scale_factor
                else:
                    pred_depths = out * self.config.scale_factor
                pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)

                losses = {}
                loss = 0
                for k in range(len(self.config.loss_depth)):
                    l = self.loss_depth[k](pred_depths, target_a, mask_a) * lam +\
                        self.loss_depth[k](pred_depths, target_b, mask_b) * (1. - lam)
                    losses[self.config.loss_depth[k]] = l
                    loss += l.to(self.device) * self.config.loss_weights[k]

            else:
                out = self.model(images_aug)
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

            if self.with_dino:
                fs = self.model.intermediate_features[0].to(self.dino_device)
                ft = self.get_dino_features(images)[0]
                target = torch.ones(fs.shape[0], fs.shape[1]).to(self.dino_device)
                loss_embedding = self.embedding_loss(fs.reshape(-1, fs.shape[-1]),
                                                     ft.reshape(-1, ft.shape[-1]),
                                                     target.view(-1))
                losses["Embedding"] = loss_embedding
                loss += loss_embedding.to(self.device) * self.config.embedding_weight

        if torch.isnan(loss):
            print("input is nan", torch.isnan(images).any())
            print("input aug is nan", torch.isnan(images_aug).any()) 
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

            self.log_images(rgb={"Input": images[0, ...], "AugInput": images_aug[0, ...]},
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

    def validate_on_batch(self, batch, val_step):
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
            align_corners=False).squeeze()

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

        pred_depths = self.rescale(pred_depths, depths_gt, mask)

        metrics = compute_metrics(depth_map_gt, pred_depths, config=self.config)

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]},
                            depth={"GT": depths_gt[0], "PredictedMono": 1.0 / (pred_depths[0] + 1e-8)},
                            prefix="Test")

        return metrics, losses


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
