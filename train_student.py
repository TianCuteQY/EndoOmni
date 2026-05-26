import json, argparse
import numpy as np

import torch, os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import FileStore, init_process_group

from dpt.models import DPTDepthModel
from util.Mix_Unlabeled_Dataloader import get_mixed_unlabeled_loader
from util.Mix_Dataloader import get_mixed_loader

from DepthAnything.depth_anything.dpt import DepthAnything

from util.misc import parallelize, count_parameters
import os

os.environ['MASTER_ADDR'] = '192.168.31.198'  # IP address of the master node
os.environ['MASTER_PORT'] = '12345'  # Use a free port
os.environ['RANK'] = '0'  # Rank of the process
os.environ['WORLD_SIZE'] = '2'  # Total number of processes

default_models = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "DPT_DINOv2": None,
}


def init_distributed_mode(world_size, rank):
    file_path = "./my_torch_filestore"  # Ensure this is not a directory
    store = FileStore(file_path, world_size)
    init_process_group(
        backend='nccl',
        store=store,
        rank=rank,
        world_size=world_size
    )


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


def get_model(model_path, model_type="DPT_DINOv2", device="cuda:0", load_backbone=False, **kwargs):
    # select device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: %s" % device)

    # load network
    if model_type == "DPT_DINOv2":
        model = DepthAnything(**kwargs)
        if model_path is not None:
            a = torch.load(model_path)
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


def get_teacher_model(device):
    weights = {"teacher_dptlarge": "/data0/tqy/checkpoints/DPTNone_04-Aug_10-45-c98c96efc8e0_best.pt"}
    # weights = {"teacher_dptlarge": "/data0/tqy/output/Teacher/TeacherNone_16-Aug_03-46-8224a535edb1_best.pt"}
    # weights = {"teacher_dptlarge": "/data0/tqy/checkpoints/Student ScratchHead noCutMix Gradientlater latest.pt"}

    model_weights = weights['teacher_dptlarge']
    model_type = "DPT_DINOv2"

    model = get_model(model_weights, model_type, device).to(device)
    model.eval()
    return model


def get_da_model(device):
    # weights = {"teacher_dptlarge": "/data0/tqy/cache/depth_anything_vitl14.pth"}
    weights = {"teacher_dptlarge": "/data0/tqy/checkpoints/DPTNone_04-Aug_10-45-c98c96efc8e0_best.pt"}

    model_weights = weights['teacher_dptlarge']
    model_type = "DPT_DINOv2"

    model = get_model(model_weights, model_type, device).to(device)
    model.eval()
    return model


def main_worker(gpu, ngpus_per_node, config):
    fix_random_seed(config.seed)
    config.gpu = gpu

    if config.model_weights is None:
        config.model_weights = default_models[config.model_type]

    ## train set
    train_dataloader = get_mixed_unlabeled_loader(config, mode="train", shuffle=True)

    ## validation set
    val_dataloader = get_mixed_loader(config, mode="val", shuffle=False)

    model = get_model(config.model_weights, config.model_type, device=config.gpu[0],
                      load_backbone=config.only_load_backbone,
                      pretrained_lr_factor=config.pretrained_lr_factor,
                      scratch_lr_factor=config.scratch_lr_factor,
                      encoder=config.student_encoder)
    model.to(config.gpu[0])

    total_params = f"{round(count_parameters(model) / 1e6, 2)}M"
    config.total_params = total_params
    print(f"Total parameters : {total_params}")

    localhub = True
    if localhub:
        dinov2 = torch.hub.load('/data0/tqy/cache/facebookresearch_dinov2_main',
                                'dinov2_{:}14'.format(config.student_encoder), source='local', pretrained=True)
    else:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(config.student_encoder),
                                pretrained=True)
    dinov2.to(config.gpu[1])
    dinov2.eval()

    if "weighted" in config.loss_depth:
        from Trainer_student_weighted import Trainer
    else:
        from Trainer_student import Trainer

    trainer = Trainer(config, model, train_dataloader, val_dataloader,
                      device=config.gpu[0], teacher_model=get_teacher_model(config.gpu[1]), dino_model=dinov2)
    trainer.train()


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = '/data0/tqy/cache/'
    os.environ['HF_HOME'] = "/data0/tqy/cache/"

    with open('config.json', 'r') as f:
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

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node

    rank = 0
    init_distributed_mode(ngpus_per_node, rank)

    config.gpu = ['cuda:2', 'cuda:3']

    main_worker(config.gpu, ngpus_per_node, config)
