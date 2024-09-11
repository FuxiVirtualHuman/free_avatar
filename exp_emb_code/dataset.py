import numpy as np
import pandas as pd
import torch
import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import torchvision.transforms.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from timm.data import create_transform
import PIL
import pickle


class FecData(data.dataset.Dataset):
    def __init__(self, csv_file, img_path, transform=None):
        self.transform = transform
        
        self.csv_file = csv_file
        self.img_path = img_path

        self.data_anc = []
        self.data_pos = []
        self.data_neg = []
        self.type = []

        self.pd_data = pd.read_csv(self.csv_file)
        self.data = self.pd_data.to_dict("list")
        anc, pos, neg, tys = self.data["anchor"],self.data["positive"],self.data["negative"], self.data["type"]
        self.data_anc = [os.path.join(self.img_path, k) for k in anc]
        self.data_pos = [os.path.join(self.img_path, k) for k in pos]
        self.data_neg = [os.path.join(self.img_path, k) for k in neg]
        self.type = tys


    def __len__(self):
        return 100
        #return len(self.data_anc)

    def __getitem__(self, index):
        type = self.type[index]
        anc_list = self.data_anc[index]
        pos_list = self.data_pos[index]
        neg_list = self.data_neg[index]

        anc_img = Image.open(anc_list).convert('RGB')
        pos_img = Image.open(pos_list).convert('RGB')
        neg_img = Image.open(neg_list).convert('RGB')

        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        dict = {
            "name" : anc_list,
            "anc":anc_img,
            "pos":pos_img,
            "neg":neg_img,
            "type":type
        }
        
        return dict


def build_transform(is_train):
    mean = [0.49895147219604985,0.4104390648367995,0.3656147590417074]
    std = [0.2970847084907291,0.2699003075660314,0.2652599579468044]
    input_size = 224
    if is_train:
        transform = create_transform(
            input_size=224,
            is_training=True,
            scale=(0.08,1.0),
            ratio=(7/8,8/7),
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

def build_dataset(config,mode):
    train_transform = build_transform(True)
    val_transform = build_transform(False)

    dataset = None
    if mode == "train":
        dataset = FecData(config["train_csv"],config["train_img_path"],train_transform)
    elif mode == "val":
        dataset = FecData(config["val_csv"],config["val_img_path"],train_transform)
    return dataset