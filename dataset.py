import os
import cv2
import numpy as np
import pandas as pd
import torch
import glob as gl
import random
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class AnimeDataSet(Dataset):
    def __init__(self, root='', mode='',trans=None,trans_gray=None):
        super().__init__()
        self.transform = transforms.Compose(trans)
        self.transform_gray=transforms.Compose(trans_gray)
        self.source_path = os.path.join(root, "source/*")
        self.style_path = os.path.join(root, f"{mode}/style/*")
        self.smooth_path = os.path.join(root, f"{mode}/smooth/*")

        self.list_style = gl.glob(self.style_path)
        self.list_smooth = gl.glob(self.smooth_path)
        self.list_source = gl.glob(self.source_path)

    def __getitem__(self, index):
        data = {}
        style_path = random.choice(self.list_style)
        smooth_path = random.choice(self.list_smooth)
        img_path = random.choice(self.list_source)
        style = Image.open(style_path).convert('RGB')
        style_gray=Image.open(style_path).convert('L')
        smooth_gray = Image.open(smooth_path).convert('L')
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_A = self.transform(style)
        img_B = self.transform_gray(style_gray)
        img_C = self.transform_gray(smooth_gray)
        img_B=img_B.squeeze(0)
        img_C = img_C.squeeze(0)
        img_B = np.stack([img_B, img_B, img_B], axis=0)
        img_C = np.stack([img_C, img_C, img_C], axis=0)
        data.update({'source': img, 'style': img_A, 'style_gray':img_B,'smooth_gray':img_C})
        return data

    def __len__(self):
        return max(len(self.list_style), len(self.list_smooth),len(self.list_source))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    trans = [transforms.Resize(286, InterpolationMode.BICUBIC),
             transforms.CenterCrop(256),
             transforms.RandomHorizontalFlip(0.5),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    trans_gray = [transforms.Resize(286, InterpolationMode.BICUBIC),
                  transforms.CenterCrop(256),
                  transforms.RandomHorizontalFlip(0.5),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5))]
    data_loader = DataLoader(
        AnimeDataSet(root='datasets',mode='Hayao', trans=trans, trans_gray=trans_gray),
        batch_size=6, shuffle=True,
        pin_memory=True, drop_last=True)
    for i ,data in enumerate(data_loader):
        source=data['style_gray']
        print(source.shape)
        break


