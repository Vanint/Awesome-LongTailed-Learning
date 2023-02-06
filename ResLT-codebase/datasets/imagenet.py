import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random

class LT_Dataset(Dataset):
    num_classes=1000
    def __init__(self, root, txt, transform=None, class_balance=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data=[[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y=self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]
        sorted_classes=np.argsort(self.cls_num_list)
        self.class_map=[0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           label = self.class_map[self.labels[index]]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class LT_Dataset_Test(Dataset):
    def __init__(self, root, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.class_map[self.labels[index]]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class ImageNet(object):
    def __init__(self, batch_size=60, root="/research/dept6/taohu/share/ImageNet/", class_balance=False, num_works=40):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        train_txt="datasets/txts/ImageNet_LT/ImageNet_LT_train.txt"
        val_txt="datasets/txts/ImageNet_LT/ImageNet_LT_val.txt"
        test_txt="datasets/txts/ImageNet_LT/ImageNet_LT_test.txt"

        trainset = LT_Dataset(root, train_txt, transform=transform_train, class_balance=class_balance)
        testset = LT_Dataset_Test(root, test_txt, transform=transform_test, class_map=trainset.class_map)

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = num_works, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = num_works, pin_memory = True)
