from pathlib import Path

import os

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

        return {'image': image, 'depth': depth, 'mask': mask, 'dataset': "SimColon"}

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


class SimColon(Dataset):
    def __init__(self, path, resize_shape, mode="train"):
        self.transform = ToTensor(resize_shape)
        self.path = path
        self.cache_path = './cache/' + str(self.__class__.__name__) + '/' + mode + '/'
        self.mode = mode
        if mode == 'val':
            self.datasets = path + '/' + "train" + '/'
        else:
            self.datasets = path + '/' + mode + '/'
        texture_list_path = self.path + '/' + mode + '.txt'
        self.textures = [self.path + '/' + folder[:-1] for folder in open(texture_list_path)]
        self.image_files, self.depth_files, self.mask_files = [], [], []
        if os.path.exists(self.cache_path):
            self.image_files = read_txt(self.cache_path + 'image.txt')
            self.depth_files = read_txt(self.cache_path + 'depth.txt')
        else:
            self.crawl_folders()
            save_txt(self.cache_path + 'image.txt', self.image_files)
            save_txt(self.cache_path + 'depth.txt', self.depth_files)

    def crawl_folders(self):
        for texture in self.textures:
            texture_lights = [texture + '/' + l for l in list(os.listdir(texture))]
            for tl in texture_lights:
                imgs = sorted(Path(tl).files('FrameBuffer_*.*g'))
                depths = sorted(Path(tl).files('Depth_*.*g'))

                self.image_files.extend(imgs)
                self.depth_files.extend(depths)

                # for i in range(len(imgs)):
                #     self.image_files.append(imgs[i])
                #     self.depth_files.append(depths[i])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_path = self.image_files[idx]
            depth_path = self.depth_files[idx]

            image = Image.open(image_path).convert('RGB')
            # The depth is scaled to [0,1] which corresponds to [0,20] cm.
            depth = np.array(Image.open(depth_path)).astype(np.float32) / 255 * 200  # scaling
            depth = depth[..., None]

            mask = (1 < depth) & (depth < 300)

            image = np.asarray(image, dtype=np.float32) / 255.0

            sample = dict(image=image, depth=depth, mask=mask)

            # return sample
            sample = self.transform(sample)

            # if idx == 0:
            #     print(sample["image"].shape)
            sample['depth'] = torch.squeeze(sample['depth'])
            sample['mask'] = torch.squeeze(sample['mask']).to(torch.int)

            return sample
        except:
            print(image_path)
            self.__getitem__(idx+1)


def get_SimColon_loader(data_dir_root, resize_shape, mode="train", batch_size=1, **kwargs):
    dataset = SimColon(data_dir_root, resize_shape, mode)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_SimColon_loader(
        data_dir_root="/media/t/Elements1/SimColonDepth", resize_shape=[384, 384])
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            from util.misc import colorize
            depth = colorize(sample['depth'].numpy(), vmin=None, vmax=None)
            d = Image.fromarray(depth)
            d.show()
            break
