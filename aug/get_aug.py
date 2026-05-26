import torch
from torchvision.transforms import ColorJitter, GaussianBlur, v2, Compose
import numpy as np
import torchvision.transforms as transforms
from typing import List

class Aug():
    def __init__(self, config, device="cuda", brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2, kernel_size=7):
        self.config = config
        self.with_cutmix = False
        self.cutmix_prob = 0
        self.transform = self.get_perturbation(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2, kernel_size=7)

    def get_perturbation(self, brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2, kernel_size=7):
        transforms = []
        if "CJ" in self.config.student_perturbation:
            transforms.append(ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        if "Blur" in self.config.student_perturbation and kernel_size > 0:
            transforms.append(GaussianBlur(kernel_size=kernel_size))
        if "CutMix" in self.config.student_perturbation:
            self.with_cutmix = True
            self.cutmix_prob = 0.5
        return Compose(transforms)

    def __call__(self, img, depth=None, source=None, *args, **kwargs):
        img = (img + 1) / 2
        imgaug = self.transform(img)
        return imgaug * 2 - 1
