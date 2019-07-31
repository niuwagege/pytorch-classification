import torch.utils.data as data
from PIL import Image
import os
import numpy as np

"""
This is for multi classes.

label txt and image folder should be provided
label txt format:
filename1,label1
filename2,label2
filename3,label3
filename4,label4
......
label start from 0,like 0,1,2,3...
"""

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(label_txt_path, image_folder):
    fr = open(label_txt_path, "r")
    lines = fr.readlines()
    lines = [line.strip() for line in lines]
    images = []
    for line in lines:
        image_file, label = line.split(',')
        image_path = os.path.join(image_folder, image_file)
        images.append((image_path, int(label)))
    return images




class TxtImageLabel(data.Dataset):
    def __init__(self, image_folder, label_txt_path, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(label_txt_path, image_folder)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
