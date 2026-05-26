import torch
import torch.nn as nn
from util.base_trainer import BaseTrainer
from util.WeightedAffineLoss import WeightedAffineLoss
from util.misc import compute_metrics, AffineRescale
import torch.cuda.amp as amp
import numpy as np
from aug.get_aug import Aug
from torchvision.transforms import functional as TF
import torch.nn.functional as F


def get_device(model):
    if model is not None:
        return next(model.parameters()).device
    else:
        return None


@torch.no_grad()
def normalize(value, mask):
    value = value.to(torch.float32)

    value[~mask] = float(1e6)
    min_vals = value.amin(dim=(1, 2), keepdim=True)
    value[~mask] = float(1e-8)
    max_vals = value.amax(dim=(1, 2), keepdim=True)

    valid = (max_vals - min_vals) > 0
    min_vals[~valid] = 0
    max_vals[~valid] = 1

    normalized = (value - min_vals) / (max_vals - min_vals)

    normalized[~mask] = 0

    return normalized


class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader=None, test_loader=None, device=None,
                 teacher_model=None, dino_model=None):
        super().__init__(config, model, train_loader=train_loader,
                         test_loader=test_loader, device=device)
        self.device = get_device(model)
        self.teacher_device, self.dino_device = get_device(teacher_model), get_device(dino_model)
        self.config = config
        self.loss_depth = WeightedAffineLoss()
        self.scale_factor = config.scale_factor
        self.scale, self.shift = None, None
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_transform = Aug(config, device=self.device, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, kernel_size=0)
        self.student_transform = Aug(config, device=self.device)
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

    @staticmethod
    def apply_augmentations(image):
        """Apply random augmentations and return the transformations applied."""
        transformations = []
        # if torch.rand(1) < 0.5:
        image = TF.hflip(image)
        transformations.append(TF.hflip)  # Store the flip operation to reverse later

        # Define an array of angles that are multiples of 90
        angles = [0, 90, 180, 270]
        # Randomly select an angle
        angle = angles[torch.randint(0, len(angles), (1,)).item()]
        image = TF.rotate(image, angle)
        # Define the reverse rotation, which is just rotating by the negative of the original angle
        transformations.append(lambda x: TF.rotate(x, -angle))

        return image, transformations

    @staticmethod
    def reverse_augmentations(image, transformations):
        """Reverse the augmentations applied to the image."""
        for transform in reversed(transformations):  # Reverse the transformations
            image = transform(image)
        return image

    @torch.no_grad()
    def calculate_consistency_weights(self, original_images, mask, aug_images=None):
        # aug_images, reverse_transform = self.apply_augmentations(original_images)  

        # combined_images = torch.cat([original_images, aug_images], dim=0)
        # combined_preds = self.teacher_model(combined_images.to(self.teacher_device))

        # nan_inds = torch.isnan(combined_preds)
        # combined_preds[nan_inds] = 0

        # original_preds = combined_preds[:len(original_images)]
        # aug_preds = combined_preds[len(original_images):]
        # aug_preds = self.reverse_augmentations(aug_preds, reverse_transform)

        # consistency = torch.square_(normalize(original_preds, mask) - normalize(aug_preds, mask))
        # weights = torch.exp(-consistency / 0.5)

        # return weights.to(self.device), 0.5 * (original_preds + aug_preds).to(torch.float32)
        if aug_images is None:
            aug_images = self.teacher_transform(original_images)
    
        combined_images = torch.cat([original_images, aug_images], dim=0)
        combined_preds = self.teacher_model(combined_images.to(self.teacher_device))

        nan_inds = torch.isnan(combined_preds)
        combined_preds[nan_inds] = 0

        original_preds = combined_preds[:len(original_images)]
        aug_preds = combined_preds[len(original_images):]

        consistency = torch.square_(normalize(original_preds, mask) - normalize(aug_preds, mask))
        weights = torch.exp(-consistency / 0.5)

        return weights.to(self.device), original_preds.to(torch.float32)
        

    def train_on_batch(self, batch, train_step):
        images = batch['image'].to(self.device)
        depths_gt = 1.0 / (batch['depth'] + 1e-8).to(torch.float32).to(self.device)
        mask = batch["mask"].to(self.device).to(torch.bool)
        weights = torch.ones_like(batch["mask"], dtype=torch.float32).to(self.device)

        labeled_inds = [i for i in range(batch['depth'].shape[0]) if batch['dataset'][i] != "unlabeled"]
        unlabeled_inds = [i for i in range(batch['depth'].shape[0]) if batch['dataset'][i] == "unlabeled"]

        if torch.isnan(mask).any():
            raise 'mask nan'
        if torch.isnan(depths_gt).any():
            raise 'depths_gt nan'

        with amp.autocast(enabled=self.config.use_amp):
            images_aug = self.student_transform(images, depths_gt, batch['dataset']).to(self.device)
            if unlabeled_inds:
                # Generate pseudo-labels and calculate consistency weights for unlabeled data
                weights_unlabeled, pseudo_labels = self.calculate_consistency_weights(images[unlabeled_inds],
                                                                                      mask[unlabeled_inds].to(
                                                                                          self.teacher_device))
                weights[unlabeled_inds] = weights_unlabeled.to(torch.float32)
                depths_gt[unlabeled_inds] = pseudo_labels.to(self.device)

            if labeled_inds:
                # Calculate consistency weights for labeled data using teacher model
                teacher_preds = self.teacher_model(images[labeled_inds].to(self.teacher_device)).to(self.device).to(torch.float32)
                nan_inds = torch.isnan(teacher_preds)
                teacher_preds[nan_inds] = depths_gt[labeled_inds][nan_inds]
                consistency_labeled = torch.square_(normalize(teacher_preds.clone(), mask[labeled_inds])
                                                    - normalize(depths_gt[labeled_inds].clone(), mask[labeled_inds]))
                weights_labeled = torch.exp(-consistency_labeled / 0.1)
                # weights_labeled = weights_labeled / (weights_labeled.max() + 1e-8)  # Normalize weights
                weights[labeled_inds] = weights_labeled.to(torch.float32)
                if torch.isnan(weights_labeled).any():
                    a = torch.isnan(consistency_labeled).any()
                    b = torch.isnan(teacher_preds).any()
                    raise

            nan_inds = torch.isnan(depths_gt)
            depths_gt[nan_inds] = 0
            mask[nan_inds] = False

            # # Normalize depths_gt using mask
            # if torch.any(mask):
            #     C = depths_gt.shape[0]
            #     channel_maxs = (depths_gt * mask).view(C, -1).max(dim=1)[0].view(C, 1, 1).expand_as(depths_gt)
            #     depths_gt[mask] = depths_gt[mask] / (channel_maxs[mask] + 1e-8)

            # Process images for model input
            nan_inds = torch.isnan(images_aug)
            images_aug[nan_inds] = images[nan_inds]
            out = self.model(images_aug)
            if self.config.inverse:
                pred_depths = 1.0 / (out + 1e-8) * self.config.scale_factor
            else:
                pred_depths = out * self.config.scale_factor
            pred_depths = torch.nan_to_num(pred_depths, nan=self.config.max_depth)
            pred_depths = torch.clip(pred_depths, self.config.min_depth, self.config.max_depth)

            # Calculate loss
            losses = {}
            loss = 0
            if torch.isnan(depths_gt).any():
                raise "wrong label"
            if torch.isnan(mask).any():
                raise "wrong mask"
            l = self.loss_depth(pred_depths, depths_gt, mask, weights)

            losses['WeightedAffineLoss'] = l.mean()
            loss += l.mean()

            # DINO loss if applicable
            if self.with_dino:
                fs = self.model.intermediate_features[0].to(self.dino_device)
                ft = self.get_dino_features(images)[0]

                fs_norm = torch.norm(fs, p=2, dim=-1, keepdim=True)
                ft_norm = torch.norm(ft, p=2, dim=-1, keepdim=True)
                fs = torch.where(fs_norm > 0, fs, torch.full_like(fs, 1e-6))
                ft = torch.where(ft_norm > 0, ft, torch.full_like(ft, 1e-6))

                fs = fs.reshape(-1, fs.shape[-1])
                ft = ft.reshape(-1, ft.shape[-1])

                if not torch.isfinite(fs).all() or not torch.isfinite(ft).all():
                    print("Warning: Non-finite values detected before calculating embedding loss.")
                    fs = torch.nan_to_num(fs)
                    ft = torch.nan_to_num(ft)

                target = torch.ones(fs.shape[0], device=self.dino_device)
                loss_embedding = self.embedding_loss(fs.reshape(-1, fs.shape[-1]),
                                                     ft.reshape(-1, ft.shape[-1]),
                                                     target.view(-1))
                losses["Embedding"] = loss_embedding
                loss += loss_embedding.to(self.device) * self.config.embedding_weight

            # Gradient backpropagation and optimization
            if torch.isnan(loss):
                print("Input is NaN", torch.isnan(images).any())
                print("Input aug is NaN", torch.isnan(images_aug).any())
                print('GT is NaN', torch.isnan(depths_gt).any())
                print('Pred is NaN', torch.isnan(pred_depths).any())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Log images for visualization
            # if self.should_log and (train_step % int(1)) == 0:
            if self.should_log and (train_step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
                # -99 is treated as invalid depth in the log_images function and is colored grey.
                depths_gt[torch.logical_not(mask)] = -99
                self.log_images(rgb={"Input": images[0], "AugInput": images_aug[0]},
                                depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]},
                                weight={"Weight": weights[0]},
                                prefix="Train")

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
                    l = self.loss_depth(pred_depths, depths_gt_norm, mask).to(self.device)
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
