import torch.utils.data as data
from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(dir, label_dict):
    images = []
    detail_labels = list(label_dict.keys())
    cate_labels = list(set(label_dict.values()))
    import json
    json.dump(detail_labels,open("detail.json","w"))
    json.dump(cate_labels,open("cate.json","w"))
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        d_index = detail_labels.index(target)
        c_index = cate_labels.index(label_dict[target])
        d_onehot = np.zeros(len(detail_labels))
        d_onehot[d_index] = 1
        c_onehot = np.zeros(len(cate_labels))
        c_onehot[c_index] = 1
        onehot = np.concatenate((d_onehot,c_onehot),0)
        for filename in os.listdir(d):
            if is_image_file(filename):
                # path = '{0}/{1}'.format(target, filename)
                path = os.path.join(dir,target,filename)

                item = (path, onehot)
                images.append(item)
    return images

def read_label_dict(file_path):
    label_dict = dict()
    with open(file_path,"r") as fr:
        for line in fr.readlines():
            key_d, key_c = line.strip().split(',')
            label_dict[key_d] = key_c
    return label_dict

class TrashImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        label_dict = read_label_dict(os.path.join(root,"trash_label.txt"))
        imgs = make_dataset(root,label_dict)
        self.root = root
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