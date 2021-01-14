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
                config,
                transforms=None,
                output_label=True):

        self.df = df
        self.transforms = transforms
        self.config = config
        self.output_label = output_label

        if output_label == True:
            self.labels = self.df['label'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.output_label:
            target = self.labels[index]

        img_path = os.path.join(self.config.paths['train_path'], self.df.loc[index]['image_id'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # return (H x W x C)

        if self.transforms:
            img = self.transforms(image=img)['image'] # return (C x H x W)

            if self.config.cmix:
                img, target = cutmix(img, self.df, index, self.config, transforms=self.transforms)

        #do label smoothing (add)
        if self.output_label == True:
            return img, target
        else:
            return img  #model accepts (C x H x W)


def get_train_transforms(config):
    return Compose([
            RandomResizedCrop(config.resize, config.resize),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Transpose(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(config):
    return Compose([
            RandomResizedCrop(config.resize, config.resize),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_test_transforms(config):
    return Compose([
            RandomResizedCrop(config.resize, config.resize),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Transpose(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def prepare_dataloader(train_df, valid_df, config):
    train_ds = CassavaDataset(train_df, config, transforms=get_train_transforms(config),
                output_label=True)
    valid_ds = CassavaDataset(valid_df, config, transforms=get_valid_transforms(config),
                output_label=True)
    valid_tta_ds = CassavaDataset(valid_df, config, transforms=get_test_transforms(config),
                output_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4)

    val_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=False)

    tta_loader = DataLoader(
        valid_tta_ds,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=False)

    return train_loader, val_loader, tta_loader
