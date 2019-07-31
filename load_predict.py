import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
image_path = "/root/tdata/val/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])
fr = open("labels_dicts.txt","r")
fw = open("labels_dicts2.txt","w")
lines = fr.readlines()
pre_dict = {}
for line in lines:
    line = line.strip()
    key,value = line.split(':')
    pre_dict[value] = key
count = 0
for folder in os.listdir(image_path):
    if folder in pre_dict:
        continue
    if folder[0] is '.':
        continue
    print(count,folder)
    count += 1
    
    res_dict = {}
    folder_path = os.path.join(image_path,folder)
    if not os.path.isdir(folder_path):
        continue
    labels = []
    if len(os.listdir(folder_path)) == 0:
        print(folder)
        continue
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path,file)
        if not is_image_file(img_path):
            continue
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = transform(img)
        model = torch.load('./trash_cpu.pth',map_location='cpu')
        model.eval()

        result = model(img.unsqueeze(0))
        _,indexes = torch.topk(result[:,0:103],k=5,dim=1)
        labels.append(indexes.numpy()[0][0])
#         print(torch.topk(result[:,0:103],k=5,dim=1))
#         print(torch.topk(result[:,103:],k=1,dim=1))
#         print("---------------")
    counts = np.bincount(labels)
    label = np.argmax(counts)
    fw.writelines(str(label)+":"+folder+"\n")
    fw.flush()
    res_dict[label] = folder
print("done!")
json.dump(res_dict,open("./label.json","w"))
print("saved!")