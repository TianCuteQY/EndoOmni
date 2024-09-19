# MIT License
import copy
# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os, argparse
import uuid
import warnings
from datetime import datetime as dt
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image 
import wandb
from wandb import AlertLevel
from tqdm import tqdm
from PIL import Image

from util.misc import RunningAverageDict, colorize, colors


def is_rank_zero(args):
    return args.rank == 0


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch, warmup_epochs=0):
        assert ((n_epochs - decay_start_epoch) >= 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1, self.warmup_epochs))
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch + 1e-8)


class BaseTrainer:
    def __init__(self, config, model, train_loader, test_loader=None, device=None, **kwargs):
        """ Base Trainer class for training a model."""

        self.config = config
        self.metric_criterion = "abs_rel"
        if device is None:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    def resize_to_target(self, prediction, target):
        if prediction.shape[2:] != target.shape[-2:]:
            prediction = nn.functional.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=True
            )
        return prediction

    def init_optimizer(self):
        m = self.model

        if self.config.same_lr:
            print("Using same LR")
            if hasattr(m, 'core'):
                m.core.unfreeze()
            params = self.model.parameters()
        else:
            print("Using diff LR")
            if not hasattr(m, 'get_lr_params'):
                raise NotImplementedError(
                    f"Model {m.__class__.__name__} does not implement get_lr_params. Please implement it or use the same LR for all parameters.")

            params = m.get_lr_params(self.config.lr)

        return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

    def init_scheduler(self):
        # lrs = [l['lr'] for l in self.optimizer.param_groups]
        # return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        # return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.config.epochs, self.config.epoch,
        #                                                                       self.config.decay_epoch,
        #                                                                       warmup_epochs=self.config.warmup_epoch).step)
        # sched_kwargs = {"div_factor": 1, "final_div_factor": 10000, "pct_start": 0.7, "three_phase":False, "cycle_momentum": True}

        # return optim.lr_scheduler.OneCycleLR(self.optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader),
        #                                      cycle_momentum=sched_kwargs['cycle_momentum'],
        #                                      base_momentum=0.85, max_momentum=0.95, div_factor=sched_kwargs['div_factor'],
        #                                      final_div_factor=sched_kwargs['final_div_factor'], 
        #                                      pct_start=sched_kwargs['pct_start'],
        #                                      three_phase=sched_kwargs['three_phase'])
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=7000, gamma=0.8)

    def train_on_batch(self, batch, train_step):
        raise NotImplementedError

    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError

    def raise_if_nan(self, losses):
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value):
                if self.should_log:
                    wandb.alert(
                        title="Loss is Nan",
                        text="Loss is Nan",
                        level=AlertLevel.ERROR
                    )
                raise ValueError(f"{key} is NaN, Stopping training")

    @property
    def iters_per_epoch(self):
        return len(self.train_loader)

    @property
    def total_iters(self):
        return self.config.epochs * self.iters_per_epoch

    def should_early_stop(self):
        if self.config.__dict__.get('early_stop', False) and self.step > self.config.early_stop:
            return True

    def train(self):
        print(f"Training {self.config.name}")
        if self.config.uid is None:
            self.config.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{self.config.uid}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = self.config.enable
        self.should_log = self.should_write  # and logging
        if self.should_log:
            wandb.init(project=self.config.name, entity=self.config.username, dir=self.config.root,
                       settings=wandb.Settings(start_method="fork"))
            wandb.config = {
                "learning_rate": self.config.lr,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
            }

        self.model.train()
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.config.validate_every * self.iters_per_epoch)

        if self.config.prefetch:

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Prefetching...",
                                 total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(
                self.train_loader):
                pass

        losses = {}

        def stringify_losses(L):
            return "; ".join(map(
                lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(), 3):.4e}", L.items()))

        for epoch in range(self.config.epochs):
            if self.should_early_stop():
                break

            self.epoch = epoch
            ################################# Train loop ##########################################################
            if self.should_log:
                wandb.log({"Epoch": epoch}, step=self.step)
            pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader)
            for i, batch in pbar:
                if self.should_early_stop():
                    print("Early stopping")
                    break
                # print(f"Batch {self.step+1} on rank {self.config.rank}")
                self.model.train()
                losses = self.train_on_batch(batch, i)
                # print(f"trained batch {self.step+1} on rank {self.config.rank}")

                self.raise_if_nan(losses)
                if is_rank_zero(self.config) and self.config.print_losses:
                    pbar.set_description(
                        f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                self.scheduler.step()

                if self.should_log and self.step % 50 == 0:
                    # Log the losses
                    log_data = {f"Train/{name}": loss.item() for name, loss in losses.items()}
                    
                    # Log the learning rate(s)
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        log_data[f"Train/LR_group_{i}"] = param_group['lr']

                    # Log everything to wandb
                    wandb.log(log_data, step=self.step)

                self.step += 1

                ########################################################################################################

                if self.test_loader:
                    if (self.step % validate_every) == 0:
                        self.model.eval()
                        if self.should_write:
                            self.save_checkpoint(
                                f"{self.config.experiment_id}_latest.pt")

                        ################################# Validation loop ##################################################
                        # validate on the entire validation set in every process but save only from rank 0, I know, inefficient, but avoids divergence of processes
                        metrics, test_losses = self.validate()
                        # print("Validated: {}".format(metrics))
                        if self.should_log:
                            wandb.log(
                                {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)

                            wandb.log({f"Metrics/{k}": v for k,
                                                             v in metrics.items()}, step=self.step)

                            if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                                self.save_checkpoint(
                                    f"{self.config.experiment_id}_best.pt")
                                best_loss = metrics[self.metric_criterion]

                        # self.model.train()

                        if self.config.distributed and self.config.multigpu:
                            dist.barrier()
                        # print(f"Validated: {metrics} on device {self.config.rank}")

                # print(f"Finished step {self.step} on device {self.config.rank}")
                #################################################################################################
            # self.scheduler.step()

        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        self.save_checkpoint(f"{self.config.experiment_id}_latest.pt")
        if self.test_loader:

            ################################# Validation loop ##################################################
            metrics, test_losses = self.validate()
            # print("Validated: {}".format(metrics))
            if self.should_log:
                wandb.log({f"Test/{name}": tloss for name,
                                                     tloss in test_losses.items()}, step=self.step)
                wandb.log({f"Metrics/{k}": v for k,
                                                 v in metrics.items()}, step=self.step)

                if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                    self.save_checkpoint(
                        f"{self.config.experiment_id}_best.pt")
                    best_loss = metrics[self.metric_criterion]

        self.model.train()

    def validate(self):
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            for i, batch in tqdm(enumerate(self.test_loader),
                                 desc=f"Epoch: {self.epoch + 1}/{self.config.epochs}. Loop: Validation",
                                 total=len(self.test_loader), disable=not is_rank_zero(self.config)):
                metrics, losses = self.validate_on_batch(batch, val_step=i)

                if losses:
                    losses_avg.update(losses)
                if metrics:
                    metrics_avg.update(metrics)

            return metrics_avg.get_value(), losses_avg.get_value()

    def test(self, scale_mode="affine"):
        self.should_log = False
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                metrics, losses = self.test_on_batch(batch, val_step=i, scale_mode=scale_mode)

                if losses:
                    losses_avg.update(losses)
                if metrics:
                    metrics_avg.update(metrics)

                # print(metrics_avg.get_value())

            return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        # if not self.should_write:
        #     return
        root = self.config.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.config.multigpu else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": None,
                # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                "epoch": self.epoch
            }, fpath)

    def log_images(self, rgb: Dict[str, list] = {}, depth: Dict[str, list] = {}, scalar_field: Dict[str, list] = {},
                   weight: Dict[str, list] = {},
                   prefix="", scalar_cmap="jet", min_depth=None, max_depth=None):
        if not self.should_log:
            return

        depth = {k: colorize(v, vmin=min_depth, vmax=max_depth)
                 for k, v in depth.items()}
        scalar_field = {k: colorize(
            v, vmin=None, vmax=None, cmap=scalar_cmap) for k, v in scalar_field.items()}
        images = {**rgb, **depth, **weight, **scalar_field}
        wimages = {
            prefix + "Predictions": [wandb.Image(v, caption=k) for k, v in images.items()]}
        wandb.log(wimages, step=self.step)

    def log_line_plot(self, data):
        if not self.should_log:
            return

        plt.plot(data)
        plt.ylabel("Scale factors")
        wandb.log({"Scale factors": wandb.Image(plt)}, step=self.step)
        plt.close()

    def log_bar_plot(self, title, labels, values):
        if not self.should_log:
            return

        data = [[label, val] for (label, val) in zip(labels, values)]
        table = wandb.Table(data=data, columns=["label", "value"])
        wandb.log({title: wandb.plot.bar(table, "label",
                                         "value", title=title)}, step=self.step)
