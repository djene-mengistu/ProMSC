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
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, is_unlabeled=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std 
        self.is_unlabeled = is_unlabeled
        
        if self.is_unlabeled:
            # Base spatial transforms (applied to both weak and strong)
            self.spatial_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.4),
                A.RandomRotate90(p=0.2),
            ])
            # Weak non-spatial transforms
            self.weak_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(var_limit=5.0, p=0.2),
                # A.Blur(p=0.2),
            ])
            # Strong non-spatial transforms
            self.strong_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                A.Blur(blur_limit=3, p=0.3), # Minimal blur
                A.GaussNoise(var_limit=(5.0, 10.0), p=0.3), # Light noise
                # A.ColorJitter(p=0.3),
            ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.X[idx] + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, self.X[idx] + '.png'), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Failed to load image or mask for {self.X[idx]}") 

        if self.is_unlabeled:
            # Apply same spatial transforms to both weak and strong versions
            spatial_aug = self.spatial_transform(image=img, mask=mask)
            img_spatial = spatial_aug['image']
            mask = spatial_aug['mask'] 
            weak_aug = self.weak_transform(image=img_spatial) 
            img_weak = weak_aug['image']

            # Apply strong non-spatial transforms
            strong_aug = self.strong_transform(image=img_spatial) 
            img_strong = strong_aug['image']

            # Normalize both versions
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize(self.mean, self.std)
            ])
            img_weak = normalize(img_weak)
            img_strong = normalize(img_strong)
            mask = torch.from_numpy(mask).long() 
            return img_weak, img_strong, mask#, boundary#, class_label
        else:
            # For labeled data, apply the regular transform
            if self.transform is not None:
                aug = self.transform(image=img, mask=mask) 
                img = aug['image']
                mask = aug['mask']
            else:
                # img = Image.fromarray(img)
                img = img 
            img = T.Compose([
                T.ToTensor(),
                T.Normalize(self.mean, self.std)])(img)
            mask = torch.from_numpy(mask).long() 
            
            return img, mask#, boundary#, class_label

def load_DEF_dataloaders(image_path_train, mask_path_train,
                         image_path_test, mask_path_test,
                         batch_size=16, unlabeled_ratio=0.8, num_workers=4):
    
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

    # Transforms for labeled data
    t_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(5.0, 10.0), p=0.2),         
        A.RandomRotate90(p=0.2),
        # A.ColorJitter(p=0.3),
        # A.GaussNoise(p=0.3)
    ])

    t_val_test = A.Compose([
        A.Resize(224, 224)
    ])

    # Dataset dictionary
    datasets = {
        'train': DEFDataset(image_path_train, mask_path_train, X_train, mean, std, t_train),
        'train_u': DEFDataset(image_path_train, mask_path_train, X_untrain, mean, std,
                            transform=None, is_unlabeled=True),
        # 'val': DEFDataset(image_path_train, mask_path_train, X_val, mean, std, t_val_test),
        'test': DEFDataset(image_path_test, mask_path_test, X_test, mean, std, t_val_test),
    }

    # DataLoader settings
    loaders = {}
    for k, dset in datasets.items():
        shuffle = k in ['train', 'train_u']
        drop_last = k in ['train_u']  # Set drop_last=True only for 'train_u' #CHECK
        loaders[k] = DataLoader(
            dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last
        )

    return loaders