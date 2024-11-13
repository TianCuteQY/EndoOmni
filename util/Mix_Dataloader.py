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
from util.Dataset_tqy_v import get_tqyv_loader
from util.Dataset_bronchotest import get_bronchotest_loader
from util.Dataset_phantomtest import get_phantomtest_loader
from util.Dataset_ExpNoise import get_ExpNoise_loader


DATASETS_CONFIG = {
    "tqy_endo": {
        "dataset": "tqy_endo",
        "root": "/data0/tqy/BDL-V/",
    },
    "tqy_v": {
        "dataset": "tqy_v",
        "root": "/data0/tqy/BDL-V/",
    },
    "C3VD": {
        "dataset": "C3VD",
        "root": "/data0/tqy/C3VD/",
    },
    "EndoAbs": {
        "dataset": "EndoAbs",
        "root": "/data0/tqy/EndoAbS_dataset/",
    },
    "Hamlyn": {
        "dataset": "Hamlyn",
        "root": "/data0/tqy/hamlyn_data/",
    },
    "SCARED": {
        "dataset": "SCARED",
        "root": "/data0/tqy/SCARED/out/",
    },
    "SERV-CT": {
        "dataset": "SERV-CT",
        "root": "/data0/tqy/SERV-CT/"
    },
    "SimColon": {
        "dataset": "SimColon",
        "root": "/data0/tqy/SimColonDepth/"
    },
    "EndoMapper-Sim": {
        "dataset": "EndoMapper-Sim",
        "root": "/data0/tqy/EndoMapper/Simulated Sequences/"
    },
    "EndoSLAM-Sim": {
        "dataset": "EndoSLAM-Sim",
        "root": "/data0/tqy/EndoSLAM/UnityCam"
    },
    "CholecT50": {
        "dataset": "CholecT50",
        "root": "/data0/tqy/CholecT50/"
    },
    "EndoMapper": {
        "dataset": "EndoMapper",
        "root": "/data0/tqy/EndoMapper/"
    },
    "CVC-ClinicDB": {
        "dataset": "CVC-ClinicDB",
        "root": "/data0/tqy/CVC-ClinicDB/"
    },
    "EAD2020": {
        "dataset": "EAD2020",
        "root": "/data0/tqy/EAD Dataset/EAD2020_train/allDetection_training/bbox_images/",
    },
    "ROBUST-MIS": {
        "dataset": "ROBUST-MIS",
        "root": "/data0/tqy/ROBUST-MIS/Raw Data/"
    },
    "EndoSLAM": {
        "dataset": "EndoSLAM",
        "root": "/data0/tqy/EndoSLAM/Cameras/"
    },
    "Surgical-Vis": {
        "dataset": "Surgical-Vis",
        "root": "/data0/tqy/SurgVisdom/surgvisdom_dataset/train/Porcine/frames/"
    },
    "BronchoTest":
    {   
        "dataset": "BronchoTest",
        "root": "/data0/tqy/BronchoTest/"
    },
    "PhantomTest":
    {   
        "dataset": "PhantomTest",
        "root": "/data0/tqy/PhantomTest/"
    },
    "ExpNoise": {
        "dataset": "ExpNoise",
        "root": "/data0/tqy/Exp_NoisyLabels/"
    },
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
        if config.dataset == 'tqy_v':
            self.data = get_tqyv_loader(config.path, resize_shape=(
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
        if config.dataset == "EndoMapper-Sim":
            self.data = get_EndoMapper_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "BronchoTest":
            self.data = get_bronchotest_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "PhantomTest":
            self.data = get_phantomtest_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return
        if config.dataset == "ExpNoise":
            self.data = get_ExpNoise_loader(config.path, resize_shape=(
                config.im_h, config.im_w), mode=mode, batch_size=batch_size, num_workers=config.workers)
            return


class ProportionalSampler(Sampler):
    def __init__(self, datasets, proportions, shuffle=True):
        super().__init__()
        self.datasets = datasets
        self.proportions = proportions
        self.shuffle = shuffle
        self.indices = self._generate_indices()

    def _generate_indices(self):
        total_size = sum([len(ds) * prop for ds, prop in zip(self.datasets, self.proportions)])
        indices = []
        for i, (dataset, proportion) in enumerate(zip(self.datasets, self.proportions)):
            dataset_size = len(dataset)
            if dataset_size == 0:
                continue  # Skip datasets with zero samples
            sample_size = int(total_size * proportion)
            sampled_indices = torch.multinomial(torch.ones(dataset_size), sample_size, replacement=True).tolist()
            indices.extend([(i, idx) for idx in sampled_indices])

        if self.shuffle:
            random.shuffle(indices)

        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class MixedDataset(Dataset):
    def __init__(self, config, data_config=None, mode="train"):
        if data_config is None:
            data_config = DATASETS_CONFIG
        self.datasets = []
        self.dataset_sizes = []
        for dataset_name in config.mix_datasets:
            # print("Loading ", dataset_name)
            dataset_config = data_config[dataset_name]
            config.dataset = dataset_config['dataset']
            config.path = dataset_config['root']
            dataset_loader = DepthDataLoader(config, mode).data
            self.datasets.append(dataset_loader.dataset)  # Access the underlying dataset
            self.dataset_sizes.append(len(dataset_loader.dataset))
            print("Loaded ", len(dataset_loader.dataset), " files from ", dataset_name, "to", mode)

        self.total_size = sum(self.dataset_sizes)
        self.proportions = [size / self.total_size for size in self.dataset_sizes]
        self.mode = mode

    def __getitem__(self, index):
        dataset_index, sample_index = index
        sample = self.datasets[dataset_index][sample_index]

        if self.mode == "train":
            rotation_angle = random.choice([0, 90, 180, 270]) 

            if rotation_angle > 0: 
                k = rotation_angle // 90  
                sample["image"] = torch.rot90(sample["image"], k, dims=[-2, -1])  

                if "depth" in sample:
                    sample["depth"] = torch.rot90(sample["depth"], k, dims=[-2, -1])  

                if "mask" in sample:
                    sample["mask"] = torch.rot90(sample["mask"], k, dims=[-2, -1])  

        return sample

    def __len__(self):
        return self.total_size


def get_mixed_loader(config, data_config=None, mode="train", **kwargs):
    mixed_dataset = MixedDataset(config, data_config=data_config, mode=mode)
    sampler = ProportionalSampler(mixed_dataset.datasets, mixed_dataset.proportions, **kwargs)
    return DataLoader(mixed_dataset, config.batch_size, sampler=sampler, num_workers=config.workers)


if __name__ == "__main__":
    import json, argparse

    with open('../config.json', 'r') as f:
        config = json.load(f)
    config = argparse.Namespace(**config)

    loader = get_mixed_loader(config, mode="train", shuffle=True)

    for i, batch in enumerate(loader):
        print(batch["image"].shape)  # Adjust based on your actual batch structure
        print(batch["depth"].shape)
        print(batch["dataset"])
        print(batch['image'].min(), batch['image'].max())
        if i > 5:
            break
