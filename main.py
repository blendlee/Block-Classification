from model import BaseModel
from utils import split_data,augmentation
from train import train
from dataset import CustomDataset

import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings(action='ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_labels(df):
    return df.iloc[:,2:-1].values


CFG = {
    'IMG_SIZE':224,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41
}


if __name__ =='__main__':
    seed_everything(CFG['SEED']) # Seed 고정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    df = pd.read_csv('data/train_rembg.csv')

    train_df,val_df = split_data(df)

    if os.path.isfile('data/train_rembg_aug.csv'):
        train_df = pd.read_csv('data/train_rembg_aug.csv')
    else:
        train_df = augmentation(train_df)

    #train_df = pd.read_csv('data/train_aug.csv')
    #val_df = pd.read_csv('data/valid.csv')
    
    train_labels = get_labels(train_df)
    val_labels = get_labels(val_df)
    #val_labels = val_df.iloc[:,3:].values

    # train_transform = A.Compose([
    #                             A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
    #                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    #                             ToTensorV2()
    #                             ])

    # test_transform = A.Compose([
    #                             A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
    #                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    #                             ToTensorV2()
    #                             ])

    train_dataset = CustomDataset(train_df['img_path'].values, train_labels, transforms=None)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_df['img_path'].values, val_labels, transforms=None)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


    model = BaseModel()
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

    infer_model = train(model, CFG,optimizer, train_loader, val_loader, scheduler, device)