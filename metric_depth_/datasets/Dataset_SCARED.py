from pathlib import Path

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from path import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import re


class ToTensor(object):
    def __init__(self, resize_shape, mode='train'):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.shape = resize_shape
        self.resize = transforms.Resize(resize_shape, antialias=True)
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape, antialias=True),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.GaussianBlur(kernel_size=7),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mode = mode

    def crop(self, image):
        # Generate random crop parameters
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.shape)
        return i, j, h, w

    def apply_same_crop(self, images):
        cropped_images = []
        i, j, h, w = self.crop(images[0])
        for img in images:
            cropped_img = TF.crop(img, i, j, h, w)
            cropped_images.append(cropped_img)
        return cropped_images

    def __call__(self, sample):
        image, depth, mask = sample['image'], sample['depth'], sample['mask']
        
        depth = self.to_tensor(depth)
        mask = self.to_tensor(mask)

        if self.mode == 'train':
            image = self.transform(image)
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = self.to_tensor(image)
            image = self.normalize(image)
            image = self.resize(image)

        return {'image': image, 'depth': depth, 'mask': mask, 'dataset': "SCARED"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def read_txt(path):
    with open(path, "r") as file:
        read_list = [line.strip() for line in file.readlines()]
    return read_list


def save_txt(path, list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for item in list:
            file.write("%s\n" % item)


class SCARED(Dataset):

    def __init__(self, path, resize_shape, mode="train"):
        self.transform = ToTensor(resize_shape, mode)
        self.path = path
        self.cache_path = './cache/' + str(self.__class__.__name__) + '/' + mode + '/'
        self.scene_list_paths = [s for s in sorted(Path(self.path).files('*.txt')) if mode in str(s)]
        scene_list_path = self.path + '/' + mode + '.txt'
        # self.datasets = [self.path + '/' + folder[:-1] for folder in open(scene_list_path)]
        self.samples = {}
        self.image_files, self.vimage_files, self.depth_files, self.mask_files = [], [], [], []
        if os.path.exists(self.cache_path):
            self.image_files = read_txt(self.cache_path + 'image.txt')
            self.depth_files = read_txt(self.cache_path + 'depth.txt')
        else:
            self.crawl_folders()
            save_txt(self.cache_path + 'image.txt', self.image_files)
            save_txt(self.cache_path + 'depth.txt', self.depth_files)

    def crawl_folders(self):
        for sl in self.scene_list_paths:
            lines = readlines(sl)
            self.image_files.extend([
                os.path.join(self.path, re.sub(r'(\d+)', r'_\1', line.split()[0]), "data/left",
                             f"{int(line.split()[1]):06d}.png")
                for line in lines
            ])
            self.depth_files.extend([
                os.path.join(self.path, re.sub(r'(\d+)', r'_\1', line.split()[0]), "data/depthmap",
                             f"{int(line.split()[1]):06d}.png")
                for line in lines
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path).convert('RGB')
        depth = np.array(Image.open(depth_path)) / 255.0
        depth = depth[..., None]

        mask = (1 < depth) & (depth < 300)

        # try:
        #     image = np.asarray(image, dtype=np.float32) / 255.0
        # except:
        #     print("Failed opening", image_path)
        #     idx += 1
        #     return self.__getitem__(idx)

        sample = dict(image=image, depth=depth, mask=mask)

        # return sample
        sample = self.transform(sample)

        # if idx == 0:
        #     print(sample["image"].shape)
        sample['depth'] = torch.squeeze(sample['depth'])
        sample['mask'] = torch.squeeze(sample['mask']).to(torch.int)

        return sample


def get_SCARED_loader(data_dir_root, resize_shape, mode="train", batch_size=1, shuffle=True, **kwargs):
    dataset = SCARED(data_dir_root, resize_shape, mode)
    return DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


if __name__ == "__main__":
    loader = get_SCARED_loader(
        data_dir_root="/mnt/data/tqy/SCARED/out", resize_shape=[384, 384], mode="val")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            # from util.misc import colorize
            # sample['depth'][torch.logical_not(sample['mask'])] = -99
            # a = sample['depth'].numpy()
            # depth = colorize(sample['depth'].numpy(), vmin=None, vmax=None)
            # image = sample["image"].numpy()
            # d = Image.fromarray(depth)
            # d.show()
            break
