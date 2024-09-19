import itertools
import torch
import random
from torch.utils.data import DataLoader, Dataset, Sampler
from util.Dataset_tqy import get_tqy_loader
from util.Dataset_Hamlyn import get_Hamlyn_loader
from util.Dataset_EndoAbs import get_EndoAbs_loader
from util.Dataset_SERVCT import get_SERV_CT_loader
from util.Dataset_SCARED import get_SCARED_loader
from util.Dataset_C3VD import get_C3VD_loader
from util.Dataset_SimColon import get_SimColon_loader
from util.Dataset_EndoMapper import get_EndoMapper_loader
from util.Dataset_ExpNoise import get_ExpNoise_loader
from util.Dataset_Unlabeled import *

DATASETS_CONFIG = {
    "tqy_endo": {
        "dataset": "tqy_endo",
        "root": "/mnt/data/tqy/BDL-V/",
    },
    "C3VD": {
        "dataset": "C3VD",
        "root": "/mnt/data/tqy/C3VD/",
    },
    "EndoAbs": {
        "dataset": "EndoAbs",
        "root": "/mnt/data/tqy/EndoAbS_dataset/",
    },
    "Hamlyn": {
        "dataset": "Hamlyn",
        "root": "/mnt/data/tqy/hamlyn_data/",
    },
    "SCARED": {
        "dataset": "SCARED",
        "root": "/mnt/data/tqy/SCARED/out/",
    },
    "SERV-CT": {
        "dataset": "SERV-CT",
        "root": "/mnt/data/tqy/SERV-CT/"
    },
    "SimColon": {
        "dataset": "SimColon",
        "root": "/mnt/data/tqy/SimColonDepth/"
    },
    "EndoMapper-Sim": {
        "dataset": "EndoMapper-Sim",
        "root": "/mnt/data/tqy/EndoMapper/Simulated Sequences/"
    },
    "EndoSLAM-Sim": {
        "dataset": "EndoSLAM-Sim",
        "root": "/mnt/data/tqy/EndoSLAM/UnityCam"
    },
    "CholecT50": {
        "dataset": "CholecT50",
        "root": "/mnt/data/tqy/CholecT50/"
    },
    "EndoMapper": {
        "dataset": "EndoMapper",
        "root": "/mnt/data/tqy/EndoMapper/"
    },
    "CVC-ClinicDB": {
        "dataset": "CVC-ClinicDB",
        "root": "/mnt/data/tqy/CVC-ClinicDB/"
    },
    "EAD2020": {
        "dataset": "EAD2020",
        "root": "/mnt/data/tqy/EAD Dataset/EAD2020_train/allDetection_training/bbox_images/",
    },
    "ROBUST-MIS": {
        "dataset": "ROBUST-MIS",
        "root": "/mnt/data/tqy/ROBUST-MIS/Raw Data/"
    },
    "EndoSLAM": {
        "dataset": "EndoSLAM",
        "root": "/mnt/data/tqy/EndoSLAM/Cameras/"
    },
    "Surgical-Vis": {
        "dataset": "Surgical-Vis",
        "root": "/mnt/data/tqy/SurgVisdom/surgvisdom_dataset/train/Porcine/frames/"
    },
    "ExpNoise": {
        "dataset": "ExpNoise",
        "root": "/mnt/data/tqy/Exp_NoisyLabels/"
    },
    "EndoFM":{
        "dataset": "EndoFM",
        "root": "/mnt/data/tqy/EndoFM/"
    }
}


class DepthDataLoader(object):
    def __init__(self, config, mode, batch_size=1, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config: Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'tqy_endo':
            self.data = get_tqy_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'Hamlyn':
            self.data = get_Hamlyn_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'EndoAbs':
            self.data = get_EndoAbs_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'SCARED':
            self.data = get_SCARED_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'SERV-CT':
            self.data = get_SERV_CT_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'C3VD':
            self.data = get_C3VD_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'SimColon':
            self.data = get_SimColon_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "EndoMapper-Sim":
            self.data = get_EndoMapper_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'CholecT50':
            self.data = get_CholecT50_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'EndoMapper':
            self.data = get_EndoMapperu_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'CVC-ClinicDB':
            self.data = get_CVC_ClinicDB_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "EAD2020":
            self.data = get_EAD2020_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == 'ROBUST-MIS':
            self.data = get_ROBUST_MIS_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "EndoSLAM":
            self.data = get_EndoSLAM_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "Surgical-Vis":
            self.data = get_Surgical_Vis_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "ExpNoise":
            self.data = get_ExpNoise_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "EndoFM":
            self.data = get_EndoFM_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return


class ProportionalSampler(Sampler):
    def __init__(self, labeled_datasets, unlabeled_datasets, batch_size, labeled_ratio=1, unlabeled_ratio=2,
                 shuffle=True):
        super().__init__(labeled_datasets)
        self.labeled_datasets = labeled_datasets
        self.unlabeled_datasets = unlabeled_datasets
        self.labeled_ratio = labeled_ratio
        self.unlabeled_ratio = unlabeled_ratio
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = self._generate_indices()

        self.total_unlabeled = sum(len(ds) for ds in self.unlabeled_datasets)
        # Prevent division by zero when unlabeled_ratio is 0
        if self.unlabeled_ratio > 0:
            self.unlabeled_per_batch = int(
                self.batch_size * self.unlabeled_ratio / (self.labeled_ratio + self.unlabeled_ratio))
        else:
            self.unlabeled_per_batch = 0

    def _generate_indices(self):
        labeled_indices = [(1, ind, idx) for ind, ds in enumerate(self.labeled_datasets) for idx in range(len(ds))]
        unlabeled_indices = [(0, ind, idx) for ind, ds in enumerate(self.unlabeled_datasets) for idx in range(len(ds))]

        if self.shuffle:
            random.shuffle(labeled_indices)
            random.shuffle(unlabeled_indices)

        balanced_indices = []
        labeled_iter = itertools.cycle(labeled_indices)
        unlabeled_iter = itertools.cycle(unlabeled_indices) if self.unlabeled_ratio > 0 else iter([])

        num_labeled_per_batch = self.batch_size if self.unlabeled_ratio == 0 else int(
            self.batch_size * self.labeled_ratio / (self.labeled_ratio + self.unlabeled_ratio))
        num_unlabeled_per_batch = self.batch_size - num_labeled_per_batch

        total_batches = len(labeled_indices) // num_labeled_per_batch
        if self.unlabeled_ratio > 0:
            total_batches = min(total_batches, len(unlabeled_indices) // num_unlabeled_per_batch)

        for _ in range(total_batches):
            batch_indices = list(itertools.islice(labeled_iter, num_labeled_per_batch)) + \
                            list(itertools.islice(unlabeled_iter, num_unlabeled_per_batch))
            if self.shuffle:
                random.shuffle(batch_indices)
            balanced_indices.extend(batch_indices)

        return balanced_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class MixedDataset(Dataset):
    def __init__(self, config, data_config=None, mode="train"):
        if data_config is None:
            data_config = DATASETS_CONFIG
        self.labeled_datasets = []
        self.unlabeled_datasets = []

        # Loading data from labeled datasets
        for dataset_name in config.mix_datasets:
            dataset_config = data_config[dataset_name]
            config.dataset = dataset_config['dataset']
            config.path = dataset_config['root']
            dataset_loader = DepthDataLoader(config, mode).data
            self.labeled_datasets.append(dataset_loader.dataset)
            print("Loaded ", len(dataset_loader.dataset), " file from ", dataset_name, "to", mode)

        # Loading data from unlabeled datasets
        if mode == "train":
            for dataset_name in config.unlabeled_datasets:
                dataset_config = data_config[dataset_name]
                config.dataset = dataset_config['dataset']
                config.path = dataset_config['root']
                dataset_loader = DepthDataLoader(config, mode).data
                self.unlabeled_datasets.append(dataset_loader.dataset)
                print("Loaded ", len(dataset_loader.dataset), " file from ", dataset_name, "to", mode)
        self.mode = mode

    def __getitem__(self, index):
        islabeled, dataset_index, sample_index = index
        if islabeled:
            sample = self.labeled_datasets[dataset_index][sample_index]
        else:
            sample = self.unlabeled_datasets[dataset_index][sample_index]

        # if self.mode == "train":
        #     flip_prob = random.random()
        #     if flip_prob > 0.5:
        #         sample["image"] = torch.flip(sample["image"], dims=[-1]) 
                
        #         if "depth" in sample:
        #             sample["depth"] = torch.flip(sample["depth"], dims=[-1]) 

        #         if "mask" in sample:
        #             sample["mask"] = torch.flip(sample["mask"], dims=[-1]) 

        return sample

    def __len__(self):
        return sum(len(dataset) for dataset in self.labeled_datasets) \
               + sum(len(dataset) for dataset in self.unlabeled_datasets)


def get_mixed_unlabeled_loader(config, data_config=None, mode="train", shuffle=True, **kwargs):
    mixed_dataset = MixedDataset(config, data_config=data_config, mode=mode)
    if mode == "train":
        labeled_ratio, unlabeled_ratio = 1, 2
    else:
        labeled_ratio, unlabeled_ratio = 1, 0
    sampler = ProportionalSampler(
        mixed_dataset.labeled_datasets,
        mixed_dataset.unlabeled_datasets,
        config.batch_size,
        labeled_ratio=labeled_ratio,
        unlabeled_ratio=unlabeled_ratio,
        shuffle=shuffle
    )
    return DataLoader(mixed_dataset, config.batch_size, sampler=sampler, num_workers=config.workers)


if __name__ == "__main__":
    import json, argparse

    with open('../config.json', 'r') as f:
        config = json.load(f)
    config = argparse.Namespace(**config)

    loader = get_mixed_unlabeled_loader(config, mode="train", shuffle=True)

    for i, batch in enumerate(loader):
        # print(batch["image"].shape)  # Adjust based on your actual batch structure
        # print(batch["depth"].shape)
        print(batch["dataset"])
        # print(batch['image'].min(), batch['image'].max())
        # if i > 5:
        #     break
