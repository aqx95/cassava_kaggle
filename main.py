
import cv2
import os
import torch
import time
import random
import timm
import sklearn
import warnings
import pydicom
import joblib
import pandas as pd
import numpy as np
import yaml

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from glob import glob
from skimage import io
from scipy.ndimage.interpolation import zoom
from catalyst.data.sampler import BalanceClassSampler


from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss

from sklearn import metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from data import prepare_dataloader
from trainer import Fitter
from commons import seed_everything
from model import CassavaImgClassifier


# CFG = {
#     'fold_num': 5,
#     'seed': 719,
#     'model_arch': 'tf_efficientnet_b4_ns',
#     'img_size': 512,
#     'epochs': 10,
#     'train_bs': 16,
#     'valid_bs': 32,
#     'T_0': 10,
#     'lr': 1e-4,
#     'min_lr': 1e-6,
#     'weight_decay':1e-6,
#     'num_workers': 4,
#     'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
#     'verbose_step': 1,
#     'device': 'cuda:0'
# }
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config







# ### TRAINING
#
# def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device,
#                     scheduler=None, schd_batch_update=False):
#     model.train()
#
#     t = time.time()
#     running_loss = None
#
#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     for step, (imgs, image_labels) in pbar:
#         imgs = imgs.to(device).float()
#         image_labels = image_labels.to(device).long()
#
#         with autocast(): #only wrap forward pass (with loss calc) with autocast
#             image_preds = model(imgs)
#             loss = loss_fn(image_preds, image_labels)
#
#         scaler.scale(loss).backward()
#
#         if running_loss is None:
#             running_loss = loss.item()
#         else:
#             running_loss = running_loss*.99 + loss.item()*0.01
#
#         if ((step+1) % CFG['accum_iter'] == 0 or (step+1) == len(train_loader)):
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#
#             if scheduler is not None and schd_batch_update:
#                 scheduler.step()
#
#         if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
#             description = f'epoch {epoch} loss: {running_loss:.4f}'
#             pbar.set_description(description)
#
#     if scheduler is not None and not schd_batch_update:
#         scheduler.step()
#
#
# def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
#     model.eval()
#
#     t = time.time()
#     loss_sum = 0
#     sample_num = 0
#     image_preds_all = []
#     image_targets_all = []
#
#     pbar = tqdm(enumerate(val_loader), total=len(val_loader))
#     for step, (imgs, image_labels) in pbar:
#         imgs = imgs.to(device).float()
#         image_labels = image_labels.to(device).long()
#
#         image_preds = model(imgs)   #output = model(input)
#         #print(image_preds.shape, exam_pred.shape)
#         image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
#         image_targets_all += [image_labels.detach().cpu().numpy()]
#
#         loss = loss_fn(image_preds, image_labels)
#
#         loss_sum += loss.item()*image_labels.shape[0]
#         sample_num += image_labels.shape[0]
#
#         if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
#             description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
#             pbar.set_description(description)
#
#     image_preds_all = np.concatenate(image_preds_all)
#     image_targets_all = np.concatenate(image_targets_all)
#     print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
#
#     if scheduler is not None:
#         if schd_loss_update:
#             scheduler.step(loss_sum/sample_num)
#         else:
#             scheduler.step()


###MAIN LOOP
if __name__ == '__main__':

    train = pd.read_csv('cassava-leaf-disease-classification/train.csv')
    config = load_config('config.yaml')
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=config['train']['fold'], shuffle=True,
                            random_state=config['train']['seed']).split(np.arange(train.shape[0]),
                            train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break;

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx,
                                    data_root='../train_images/')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(device))

        model = CassavaImgClassifier(config['model']['model_arch'], train.label.nunique(), pretrained=True).to(device)
        fitter = Fitter(model, device, config)
        fitter.fit(train_loader, val_loader)
        # scaler = GradScaler()
        # optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1,
        #                                                                 eta_min=CFG['min_lr'], last_epoch=-1)

        # loss_tr = nn.CrossEntropyLoss().to(device)
        # loss_fn = nn.CrossEntropyLoss().to(device)

        # for epoch in range(CFG['epochs']):
        #     train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device,
        #                     scheduler=scheduler, schd_batch_update=False)
        #
        #     with torch.no_grad():
        #         valid_one_epoch(epoch, model, loss_fn, val_loader, device,
        #                         scheduler=None, schd_loss_update=False)
        #
        # del model, optimizer, train_loader, val_loader, scaler, scheduler
        # torch.cuda.empty_cache()
