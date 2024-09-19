from pathlib import Path

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from path import Path
from torchvision import transforms
from util.mask_black import mask_black
import torchvision.transforms.functional as TF
import imageio

class ToTensor(object):
    def __init__(self, resize_shape, mode='crop'):
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.shape = resize_shape
        self.resize = transforms.Resize(resize_shape, antialias=True)
        self.mode = mode

    def resize_shorter_side(self, image, depth, mask):
        # Resize shorter side to the target size
        _, h, w = image.size()
        if w < h:
            new_w = self.shape[0]
            new_h = int(max(h * new_w / w, 1))
        else:
            new_h = self.shape[1]
            new_w = int(max(w * new_h / h, 1))

        resize_transform = transforms.Resize((new_h, new_w), antialias=True)

        image = resize_transform(image)
        depth = resize_transform(depth)
        mask = resize_transform(mask)
        return image, depth, mask

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

        if self.mode == 'crop':
            image, depth, mask = self.resize_shorter_side(image, depth, mask)
            [image, depth, mask] = self.apply_same_crop([image, depth, mask])
            # except:
            #     image = self.resize(image)
            #     depth = self.resize(depth)
            #     mask = self.resize(mask)
            #     self.mode = "resize"
        else:
            image = self.resize(image)
            depth = self.resize(depth)
            mask = self.resize(mask)

        return {'image': image, 'depth': depth, 'mask': mask, 'dataset': "unlabeled"}

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


class Unlabeled(Dataset):
    def __init__(self, scene_list, resize_shape, endwith='*.*g', transform="crop", dataset_name="Unlabeled", mode="train"):
        self.cache_path = './cache/' + dataset_name + '/' + mode + '/'
        self.transform = ToTensor(resize_shape, transform)
        self.scene_list = scene_list
        self.image_files, self.depth_files = [], []
        if os.path.exists(self.cache_path):
            self.image_files = read_txt(self.cache_path + 'image.txt')
            self.depth_files = read_txt(self.cache_path + 'depth.txt')
        else:
            self.crawl_folders(endwith)
            save_txt(self.cache_path + 'image.txt', self.image_files)
            save_txt(self.cache_path + 'depth.txt', self.depth_files)

    def crawl_folders(self, endwith):
        for scene in self.scene_list:
            imgs = Path(scene).files(endwith)
            self.image_files.extend(imgs)

            # for i in range(len(imgs)):
            #     self.image_files.append(imgs[i])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        try:
            image = imageio.imread(image_path)
        except:
            print(image_path)
            image_path = self.image_files[idx+1]
            image = imageio.imread(image_path)
        # The depth is scaled to [0,1] which corresponds to [0,20] cm.

        image = np.array(image)

        # i = Image.fromarray(image)
        # i.show()

        mask = mask_black(image).astype(int)[..., None]
        depth = -1 * np.ones([np.shape(image)[0], np.shape(image)[1], 1], dtype=np.float32)

        sample = dict(image=np.asarray(image, dtype=np.float32) / 255.0, depth=depth, mask=mask)

        # return sample
        sample = self.transform(sample)

        # if idx == 0:
        #     print(sample["image"].shape)
        sample['depth'] = torch.squeeze(sample['depth'])
        sample['mask'] = torch.squeeze(sample['mask']).to(torch.int)

        return sample


def get_CholecT50_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == "train":
        scenes = data_dir_root + '/videos/'
        scene_list = [os.path.join(scenes, file) for file in sorted(os.listdir(scenes))]
    else:
        scene_list = []

    dataset = Unlabeled(scene_list, resize_shape, dataset_name="CholecT50", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_EndoMapperu_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode =='train':
        scenes = data_dir_root + '/Sequences/'
        scene_list = [os.path.join(scenes, file, "frames") for file in sorted(os.listdir(scenes))]
    else:
        scene_list = []

    dataset = Unlabeled(scene_list, resize_shape, dataset_name="EndoMapperu", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_CVC_ClinicDB_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == 'train':
        scene_list = [data_dir_root + 'Original/']
    else:
        scene_list = []

    dataset = Unlabeled(scene_list, resize_shape, endwith="*.tif", transform="crop", dataset_name="CVC_ClinicDB", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_EAD2020_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == "train":
        scene_list = [data_dir_root]
    else:
        scene_list = []

    dataset = Unlabeled(scene_list, resize_shape, dataset_name="EAD2020", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_ROBUST_MIS_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == "train":
        scenes = [os.path.join(data_dir_root, f) for f in os.listdir(data_dir_root)]
        scene_list = []
        for s in scenes:
            scene_list.extend([os.path.join(s, file, 'frames') for file in sorted(os.listdir(s))])
    else:
        scene_list = []

    dataset = Unlabeled(scene_list, resize_shape, dataset_name="ROBUST_MIS", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_EndoSLAM_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == "train":
        cams = [os.path.join(data_dir_root, f) for f in os.listdir(data_dir_root)]
        scene_list = []
        for c in cams:
            scene_list.extend([os.path.join(c, file)
                               for file in sorted(os.listdir(c)) if "Calibration" not in str(file)])
        traj_list = []
        for s in scene_list:
            traj_list.extend([os.path.join(s, file, 'Frames') for file in sorted(os.listdir(s))])
    else:
        traj_list = []
    dataset = Unlabeled(traj_list, resize_shape, dataset_name="EndoSLAM", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_Surgical_Vis_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode == "train":
        scenes = [os.path.join(data_dir_root, f) for f in os.listdir(data_dir_root)]
        # scene_list = []
        # for s in scenes:
        #     scene_list.extend([os.path.join(s, file, 'frames') for file in sorted(os.listdir(s))])
    else:
        scene_list = []
    dataset = Unlabeled(scenes, resize_shape, dataset_name="Surgical-Vis", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)


def get_EndoFM_loader(data_dir_root, resize_shape, batch_size=1, mode='train', **kwargs):
    if mode =='train':
        scene_list = [os.path.join(data_dir_root, file, "frames") for file in sorted(os.listdir(data_dir_root))]
    else:
        scene_list = []
    dataset = Unlabeled(scene_list, resize_shape, dataset_name="EndoFM", transform="crop", mode=mode)
    return DataLoader(dataset, batch_size, **kwargs)

if __name__ == "__main__":
    loader = get_EndoFM_loader(
        data_dir_root="/mnt/data/tqy/EndoFM/", resize_shape=[384, 384], mode="train")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            from misc import colorize
            depth = colorize(sample['mask'].numpy(), vmin=None, vmax=None)
            d = Image.fromarray(depth)
            d.show()
            break
