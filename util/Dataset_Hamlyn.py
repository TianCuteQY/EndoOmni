import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from path import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metric_depth_.zoedepth.data.preprocess import get_black_border
import torch.nn as nn


class ToTensor(object):
    def __init__(self, resize_shape, mode='train'):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.shape = resize_shape
        self.resize = transforms.Resize(resize_shape, antialias=True)
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

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)
        mask = self.to_tensor(mask)

        if self.mode != 'test':
            [image, depth, mask] = self.apply_same_crop([image, depth, mask])
        else:
            image = self.resize(image)

        return {'image': image, 'depth': depth, 'mask': mask, 'dataset': "Hamlyn"}

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


def read_txt(path):
    with open(path, "r") as file:
        read_list = [line.strip() for line in file.readlines()]
    return read_list


def save_txt(path, list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for item in list:
            file.write("%s\n" % item)


class Hamlyn(Dataset):
    def __init__(self, path, resize_shape, mode="train"):
        self.transform = ToTensor(resize_shape, mode)
        self.path = path
        self.cache_path = './cache/' + str(self.__class__.__name__) + '/' + mode + '/'
        self.mode = mode
        # if mode == 'val':
        #     self.datasets = path + '/' + "train" + '/'
        # else:
        self.datasets = path + '/' + mode + '/'
        scene_list_path = self.path + '/' + mode + '.txt'
        self.scenes = [self.path + '/' + folder[:-1] for folder in open(scene_list_path)]

        self.scenes = [self.path + f for f in os.listdir(self.path)]

        self.image_files, self.depth_files, self.mask_files = [], [], []
        # if os.path.exists(self.cache_path):
        #     self.image_files = read_txt(self.cache_path + 'image.txt')
        #     self.depth_files = read_txt(self.cache_path + 'depth.txt')
        # else:
        self.crawl_folders()
            # save_txt(self.cache_path + 'image.txt', self.image_files)
            # save_txt(self.cache_path + 'depth.txt', self.depth_files)

    def crawl_folders(self):
        for scene in self.scenes:
            if os.path.isfile(scene):
                continue
            elif "rectified" not in scene:
                continue
            if "rectified09" in scene:
                continue
            img_path = scene + '/image01/'
            depth_path = scene + '/depth01/'

            imgs = sorted(Path(img_path).files('*.*g'))
            depths = sorted(Path(depth_path).files('*.*g'))

            self.image_files.extend(imgs)
            self.depth_files.extend(depths)

            # for i in range(len(imgs)):
            #     self.image_files.append(imgs[i])
            #     self.depth_files.append(depths[i])

            # img_path = scene + '/image02/'
            # depth_path = scene + '/depth02/'
            #
            # imgs = sorted(Path(img_path).files('*.*g'))
            # depths = sorted(Path(depth_path).files('*.*g'))
            #
            # for i in range(len(imgs)):
            #     self.image_files.append(imgs[i])
            #     self.depth_files.append(depths[i])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        im = Image.open(image_path).convert('RGB')
        depth = np.array(Image.open(depth_path)).astype(np.float32)
        depth = depth[..., None]

        mask = (1 < depth) & (depth < 300)

        image = np.asarray(im, dtype=np.float32) / 255.0

        x_np = np.array(im, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right

        image = image[top:bottom, left:right, :]
        depth = depth[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        sample = dict(image=image, depth=depth, mask=mask)

        # return sample
        sample = self.transform(sample)

        # if idx == 0:
        #     print(sample["image"].shape)
        sample['depth'] = torch.squeeze(sample['depth'])
        sample['mask'] = torch.squeeze(sample['mask']).to(torch.int)

        return sample


def get_Hamlyn_loader(data_dir_root, resize_shape, mode="test", batch_size=1, **kwargs):
    dataset = Hamlyn(data_dir_root, resize_shape, mode)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_Hamlyn_loader(
        data_dir_root="/mnt/data/tqy/hamlyn_data/", resize_shape=[384, 384], mode="test")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        # if i > 5:
        #     from util.misc import colorize
        #     from torchvision.utils import save_image
        #     sample['depth'][torch.logical_not(sample['mask'])] = -99
        #     depth = colorize(sample['depth'].numpy(), vmin=None, vmax=None)
        #     d = Image.fromarray(depth)
        #     d.save('depth.png')

        #     save_image(sample['image'][0], "im.png")
        #     break