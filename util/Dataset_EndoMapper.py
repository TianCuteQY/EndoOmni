from pathlib import Path

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from path import Path
from torchvision import transforms
import OpenEXR, Imath


class ToTensor(object):
    def __init__(self, resize_shape):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Resize(resize_shape, antialias=True)

    def __call__(self, sample):
        image, depth, mask = sample['image'], sample['depth'], sample['mask']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)
        mask = self.to_tensor(mask)

        image = self.resize(image)
        depth = self.resize(depth)
        mask = self.resize(mask)

        return {'image': image, 'depth': depth, 'mask': mask, 'dataset': "EndoMapper-Sim"}

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


def read_exr(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get the header
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in ('R', 'G', 'B')]

    # Combine the channels into one array
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    img[:, :, 0] = R.reshape(size[1], size[0])
    img[:, :, 1] = G.reshape(size[1], size[0])
    img[:, :, 2] = B.reshape(size[1], size[0])

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


class EndoMapper(Dataset):
    def __init__(self, path, resize_shape, mode="train"):
        self.transform = ToTensor(resize_shape)
        self.path = path
        self.cache_path = './cache/' + str(self.__class__.__name__) + '/' + mode + '/'
        self.mode = mode
        if mode == 'val':
            self.datasets = path + '/' + "train" + '/'
        else:
            self.datasets = path + '/' + mode + '/'
        scene_list_path = self.path + '/' + mode + '.txt'
        self.scenes = [self.path + '/' + folder[:-1] for folder in open(scene_list_path)]
        self.image_files, self.depth_files, self.mask_files = [], [], []
        if os.path.exists(self.cache_path):
            self.image_files = read_txt(self.cache_path + 'image.txt')
            self.depth_files = read_txt(self.cache_path + 'depth.txt')
        else:
            self.crawl_folders()
            save_txt(self.cache_path + 'image.txt', self.image_files)
            save_txt(self.cache_path + 'depth.txt', self.depth_files)

    def crawl_folders(self):
        # for scene in self.scenes:
        #     img_path = scene + '/rgb/'
        #     depth_path = scene + '/depth/'
        #
        #     imgs = sorted(Path(img_path).files('*.*g'))
        #     # depths = sorted(Path(depth_path).files('*.exr'))
        #
        #     for i in range(len(imgs)):
        #         self.image_files.append(imgs[i])
        #         index = str(imgs[i]).split('/')[-1].split('.')[0]
        #         self.depth_files.append(depth_path + 'aov_' + index + '.exr')

        for scene in self.scenes:
            img_path = os.path.join(scene, 'rgb')
            depth_path = os.path.join(scene, 'depth')

            # Using glob from Path to get all image files sorted
            imgs = sorted(Path(img_path).glob('*.*g'))

            # Extend the list of image paths
            self.image_files.extend(imgs)

            # Simultaneously build the depth files list
            self.depth_files.extend([os.path.join(depth_path, f'aov_{img.stem}.exr') for img in imgs])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path).convert('RGB')
        depth = 1.0 / ((read_exr(depth_path) + 1e-8) * 10)  # dm --> mm
        depth = depth[..., 0][..., None]

        mask = np.ones_like(depth, dtype=np.int32)

        image = np.asarray(image, dtype=np.float32) / 255.0

        sample = dict(image=image, depth=depth, mask=mask)

        # return sample
        sample = self.transform(sample)

        # if idx == 0:
        #     print(sample["image"].shape)
        sample['depth'] = torch.squeeze(sample['depth']).to(torch.float32)
        sample['mask'] = torch.squeeze(sample['mask'])

        return sample


def get_EndoMapper_loader(data_dir_root, resize_shape, mode="train", batch_size=1, **kwargs):
    dataset = EndoMapper(data_dir_root, resize_shape, mode)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_EndoMapper_loader(
        data_dir_root="/mnt/samba_share/EndoMapper/Simulated Sequences", resize_shape=[384, 384], mode="train")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())

        if i > 5:
            from util.misc import colorize
            depth = colorize(sample['mask'].numpy(), vmin=None, vmax=None)
            d = Image.fromarray(depth)
            # d.show()
            break
