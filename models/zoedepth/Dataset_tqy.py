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

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from path import Path
from torchvision import transforms


class ToTensor(object):
    def __init__(self, resize_shape):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Resize(resize_shape)

    def __call__(self, sample):
        image, depth, mask, vimage = sample['image'], sample['depth'], sample['mask'], sample['vimage']

        image = self.to_tensor(image)
        image = self.normalize(image)
        vimage = self.to_tensor(vimage)
        vimage = self.normalize(vimage)
        depth = self.to_tensor(depth)
        mask = self.to_tensor(mask)

        image = self.resize(image)
        vimage = self.resize(vimage)
        depth = self.resize(depth)
        mask = self.resize(mask)

        return {'image': image, 'depth': depth, 'mask': mask, "vimage": vimage, 'dataset': "tqy_endo"}

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


class TQY_Endo(Dataset):
    """A data loader where the files are arranged in this way:
            root/scene_001/00000.png
            root/scene_001/00001.png
            ..
            root/scene_001/trans_rot.csv
            root/scene_002/00000.png
            .
        """

    def __init__(self, path, resize_shape, mode="train"):
        # random.seed(seed)
        self.transform = ToTensor(resize_shape)
        self.path = path
        scene_list_path = self.path + '/train.txt' if mode == 'train' else self.path + '/val.txt'
        self.scenes = [self.path + '/' + folder[:-1] for folder in open(scene_list_path)]
        self.samples = {}
        self.image_files, self.vimage_files, self.depth_files, self.mask_files = [], [], [], []
        self.crawl_folders()

    def crawl_folders(self):
        for scene in self.scenes:
            img_path = scene + '/img/'
            vimg_path = scene + '/rgb/'
            depth_path = scene + '/depth_pfm/'
            mask_path = scene + '/mask/'

            imgs = sorted(Path(img_path).files('*.*g'))
            vimgs = sorted(Path(vimg_path).files('*.*g'))
            depths = sorted(Path(depth_path).files('*.pfm'))
            if os.path.exists(mask_path):
                masks = sorted(Path(mask_path).files('*.*g'))

            for i in range(len(imgs)):
                self.image_files.append(imgs[i])
                self.vimage_files.append(vimgs[i])
                self.depth_files.append(depths[i])
                if os.path.exists(mask_path):
                    self.mask_files.append(masks[i])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        vimage_path = self.vimage_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        vimage = Image.open(vimage_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[..., None]

        try:
            mask_path = self.mask_files[idx]
            mask = np.array(Image.open(mask_path))
            mask = mask[..., None]
        except:
            mask = np.ones_like(depth)

        image = np.asarray(image, dtype=np.float32) / 255.0
        vimage = np.asarray(vimage, dtype=np.float32) / 255.0

        sample = dict(image=image, depth=depth, mask=mask, vimage=vimage)

        # return sample
        sample = self.transform(sample)

        # if idx == 0:
        #     print(sample["image"].shape)
        sample['depth'] = torch.squeeze(sample['depth'])
        sample['mask'] = torch.squeeze(sample['mask'])

        return sample


def get_tqy_loader(data_dir_root, resize_shape, mode="train", batch_size=1, **kwargs):
    dataset = TQY_Endo(data_dir_root, resize_shape, mode)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_tqy_loader(
        data_dir_root="/home/t/data/BDL-V/")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break
