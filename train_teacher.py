import json, argparse
import numpy as np

from Trainer import Trainer

import torch, os
import torch.multiprocessing as mp

from dpt.models import DPTDepthModel
from util.Mix_Dataloader import get_mixed_loader

from DepthAnything.depth_anything.dpt import DepthAnything
from util.misc import parallelize, count_parameters


default_models = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "DPT_DINOv2":  None,
}


def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): random seed
    """
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_path, model_type="dpt_hybrid", load_backbone=False, **kwargs):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
            load_backbone=load_backbone,
            **kwargs
        )
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            load_backbone=load_backbone,
            **kwargs
        )

    elif model_type == "DPT_DINOv2":
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

def main_worker(gpu, ngpus_per_node, config):

    fix_random_seed(config.seed)
    config.gpu = gpu

    if config.model_weights is None:
        config.model_weights = default_models[config.model_type]

    ## train set
    train_dataloader = get_mixed_loader(config, mode="train", shuffle=True)

    ## validation set
    val_dataloader = get_mixed_loader(config, mode="val", shuffle=False)

    model = get_model(config.model_weights, config.model_type,
                    load_backbone=config.only_load_backbone,
                    pretrained_lr_factor=config.pretrained_lr_factor,
                    scratch_lr_factor=config.scratch_lr_factor)
    model = parallelize(config, model)

    total_params = f"{round(count_parameters(model)/1e6,2)}M"
    config.total_params = total_params
    print(f"Total parameters : {total_params}")

    trainer = Trainer(config, model, train_dataloader, val_dataloader, device=config.gpu)
    trainer.train()

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = '/mnt/data/tqy/cache/'
    os.environ['HF_HOME'] = "/mnt/data/tqy/cache/"

    with open('config teacher.json', 'r') as f:
        config = json.load(f)
        config = argparse.Namespace(**config)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 1
        main_worker(config.gpu, ngpus_per_node, config)