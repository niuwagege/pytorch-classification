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

root = '/root/tdata/train'
count = 0
for folder in os.listdir(root):
    d = os.path.join(root,folder)
    if not os.path.isdir(d):
        continue
    print(d)
    for f in os.listdir(d):
        path = os.path.join(d,f)
        if not is_image_file(path):
            continue
        try:
            Image.open(path).convert('RGB')
        except OSError as e:
            os.remove(path)
            count += 1
            print(count)
        except Exception as e:
            print("other exception")