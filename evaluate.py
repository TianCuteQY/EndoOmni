import json
import argparse
import torch
from tqdm import tqdm
import os
import sys
import time
import cv2
import numpy as np
from util.misc import (RunningAverageDict, colors, compute_metrics, AffineRescale,
                       count_parameters)
from util.Mix_Dataloader import get_mixed_loader, DepthDataLoader, DATASETS_CONFIG
from metric_depth_.zoedepth.data.preprocess import get_black_border
from PIL import Image
from torchvision import transforms
from models.depth_anything.dpt import DepthAnything

def get_model(model_path, model_type="dpt_hybrid", device="cuda:0", load_backbone=False, **kwargs):
    if model_type == "DPT_DINOv2":
        model = DepthAnything(**kwargs)
        if model_path is not None:
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model'])
            except:
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        if load_backbone:
            model.initialize_head_weights()
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    return model

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
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
def infer(model, images):
    """Inference with flip augmentation"""
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            return pred
        elif isinstance(pred, (list, tuple)):
            return pred[-1]
        elif isinstance(pred, dict):
            return pred.get('metric_depth', pred.get('out'))
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")

    pred1 = model(images)
    pred1 = get_depth_from_prediction(pred1)
    return pred1

@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    ratios = []
    errors = []
    rescale = AffineRescale(depth_cap=config.max_depth)

    for sample in tqdm(test_loader, total=len(test_loader)):
        image, depth = sample['image'], sample['depth'].squeeze()
        image, depth = image.to(config.device), depth.numpy()

        pred_disp = infer(model, image).squeeze().cpu().numpy()
        gt_height, gt_width = depth.shape[-2], depth.shape[-1]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        mask = np.logical_and(depth > 1, depth < 100)

        if config.scaling == "affine":
            pred = rescale(pred_disp, 1 / (depth + 1e-8), mask).squeeze().numpy()
        elif config.scaling == "median":
            pred = 1 / (pred_disp + 1e-8)
            ratio = np.median(depth[mask]) / np.median(pred[mask])
            if not np.isnan(ratio).all():
                ratios.append(ratio)
            pred *= ratio
        else:
            pred = 1 / (pred_disp + 1e-8)

        pred = pred[mask]
        depth = depth[mask]

        pred[pred < 1] = 1
        pred[pred > 100] = 100
        error = compute_errors(depth, pred)
        if not np.isnan(error).all():
            errors.append(error)

    ratios = np.array(ratios)
    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("\n-> Done!")

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
        config = argparse.Namespace(**config)
    config.model_weights = args.model_weights
    config.dataset = args.dataset
    config.path = DATASETS_CONFIG[config.dataset]['root']
    config.batch_size = 1
    config.scaling = args.scaling
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config.model_weights, config.model_type, config.gpu)
    model.to(config.device)
    model.eval()

    test_loader = DepthDataLoader(config, mode="test", shuffle=False).data
    evaluate(model, test_loader, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate depth estimation model.')
    parser.add_argument('--config', type=str, default="./config.json", required=True, help='Path to the config JSON file')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for evaluation')
    parser.add_argument('--scaling', type=str, default="affine", choices=["affine", "median"], help='Scaling method to use for depth prediction')
    args = parser.parse_args()

    main(args)
