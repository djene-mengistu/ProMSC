import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from PIL import Image
import cv2
import albumentations as A


class DEFDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.X[idx] + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, self.X[idx] + '.png'), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        else:
            img = Image.fromarray(img)

        img = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

def load_DEF_dataloaders(image_path_train, mask_path_train, 
                         image_path_test, mask_path_test,
                         batch_size=16, unlabeled_ratio=0.8, num_workers=16):
    
    # Create dataframes
    def create_df(img_path):
        return pd.DataFrame({
            'id': [f.split('.')[0] for f in os.listdir(img_path) if f.endswith('.jpg')]
        })

    df = create_df(image_path_train)
    df_test = create_df(image_path_test)
    print(f"Total Train Images: {len(df)} | Total Test Images: {len(df_test)}")

    # Split datasets
    X_test = df_test['id'].values 
    XX_train = df['id'].values
    X_train, X_untrain = train_test_split(XX_train, test_size=unlabeled_ratio, random_state=45)

    print(f"Train Size: {len(X_train)} | Unlabeled_Train Size: {len(X_untrain)}")
    print(f"Test Size: {len(X_test)} | Test Size: {len(X_test)}")

    # Mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # mean = [0.625, 0.600, 0.576]
    # std = [0.120, 0.119, 0.117]

    # Transforms
    t_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        # A.Blur(p=0.3),
        # A.ColorJitter(p=0.3),
        # A.RandomRotate90(p=0.3),
        # A.GaussNoise(p=0.3)
    ])

    t_val_test = A.Compose([
        A.Resize(224, 224)
    ])

    # Dataset dictionary
    datasets = {
        'train': DEFDataset(image_path_train, mask_path_train, X_train, mean, std, t_train),
        'train_u': DEFDataset(image_path_train, mask_path_train, X_untrain, mean, std, t_train),
        # 'val': DEFDataset(image_path_train, mask_path_train, X_val, mean, std, t_val_test),
        'test': DEFDataset(image_path_test, mask_path_test, X_test, mean, std, t_val_test),
    }

    # DataLoader settings
    loaders = {}
    for k, dset in datasets.items():
        shuffle = k in ['train', 'train_u']
        # drop_last = k in ['train', 'train_u']
        loaders[k] = DataLoader(
            dset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, pin_memory=True, drop_last=False
        )

    return loaders
