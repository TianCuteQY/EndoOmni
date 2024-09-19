# MIT License

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

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)
import os, sys, time
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.Mix_Dataloader import DepthDataLoader, DATASETS_CONFIG
import numpy as np


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        # print("image shape:", image.shape)
        # print("depth shape:", depth.shape)
        image, depth = image.to(config.device), depth.to(config.device)
        # depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).to(config.device))  # This magic number (focal) is only used for evaluating BTS model
        t1 = time.time()
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)
        print(time.time()-t1)

        pred = pred * config.scale_factor + config.shift_factor
        if config.inverse:
            pred = 1.0 / (pred + 1e-8)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), 0, 150)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 150)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))

        # print(depth.shape, pred.shape)
        m = compute_metrics(depth, pred, config=config)
        nan_flag = False
        if not m:
            continue
        for v in m.values():
            if np.isnan(v):
                nan_flag = True
        if not nan_flag:
            metrics.update(m)

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, mode="test", shuffle=False).data
    model = model.to(config.device)
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    config.device = "cuda:7"
    config.mix_datasets = [dataset]
    config.im_h, config.im_w = 378, 378
    config.batch_size=1
    # config.save_images = "/mnt/data/tqy/out/imgs/"
    config.path = DATASETS_CONFIG[dataset]['root']
    config.scale_factor, config.shift_factor = 1, 0
    config.inverse = False
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='Hamlyn', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
