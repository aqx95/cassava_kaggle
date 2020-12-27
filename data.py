import cv2
import os
import pandas as pd
from commons import *
from torch.utils.data import Dataset,DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

class CassavaDataset(Dataset):
    def __init__(self,
                df: pd.DataFrame,
                data_root,
                config,
                transforms=None,
                output_label=True):

        self.df = df
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

        if output_label == True:
            self.labels = self.df['label'].values


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.output_label:
            target = self.labels[index]

        img_path = os.path.join(self.data_root, self.df.loc[index]['image_id'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        #do label smoothing (add)
        if self.output_label == True:
            return img, target
        else:
            return img


def get_train_transforms(config):
    return Compose([
            RandomResizedCrop(config['img_size'], config['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(config):
    return Compose([
            CenterCrop(config['img_size'], config['img_size'], p=1.),
            Resize(config['img_size'], config['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def prepare_dataloader(df, config, trn_idx, val_idx, data_root):
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    train_ds = CassavaDataset(train_, data_root, config, transforms=get_train_transforms(config),
                output_label=True)
    valid_ds = CassavaDataset(valid_, data_root, config, transforms=get_valid_transforms(config),
                output_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['data']['train_batch'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=3)

    val_loader = DataLoader(
        valid_ds,
        batch_size=config['data']['valid_batch'],
        num_workers=4,
        shuffle=False,
        pin_memory=False)

    return train_loader, val_loader
