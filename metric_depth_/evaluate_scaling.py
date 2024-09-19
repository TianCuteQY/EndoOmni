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
import os, sys, time, cv2
import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.Mix_Dataloader import DepthDataLoader, DATASETS_CONFIG
import scipy.stats as st


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


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
    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []
    
    errors = []
    ratios = []

    time_start = time.time()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            image, depth = sample['image'], sample['depth']
            # print("image shape:", image.shape)
            # print("depth shape:", depth.shape)
            image, depth = image.to(config.device), depth.squeeze().numpy()

            pred = infer(model, image, dataset=sample['dataset'][0]).squeeze().cpu().numpy()
            inference_time = time.time() - time_start
            inference_times.append(inference_time)

            pred = pred * config.scale_factor + config.shift_factor
            if config.inverse:
                pred = 1.0 / (pred + 1e-8)

            gt_height, gt_width = depth.shape[-2], depth.shape[-1]
            # print(np.shape(pred))
            # print((gt_width, gt_height))
            pred = cv2.resize(pred, (gt_width, gt_height))
            mask = np.logical_and(depth > 1, depth < 100)
            
            pred = pred[mask]
            depth = depth[mask]
            
            ratio = np.median(depth) / np.median(pred)
            if not np.isnan(ratio).all():
                ratios.append(ratio)
            pred *= ratio
            pred[pred < 1] = 1
            pred[pred > 100] = 100
            error = compute_errors(depth, pred)
            if not np.isnan(error).all():
                errors.append(error)

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    cls = []
    # for i in range(len(mean_errors)):
    #     cl = st.t.interval(alpha=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
    #     cls.append(cl[0])
    #     cls.append(cl[1])
    # cls = np.array(cls)
    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    # print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
    print("\n-> Done!")

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, mode="test", shuffle=False).data
    model = model.to(config.device)
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    # metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    config.device = "cuda:4"
    config.mix_datasets = [dataset]
    config.im_h, config.im_w = 378, 378
    config.batch_size=1
    config.path = DATASETS_CONFIG[config.dataset]['root']
    # config.scale_factor, config.shift_factor = 0.011610080458343256, 0.01681966941795187
    # config.inverse = True
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
                        default='SCARED', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
