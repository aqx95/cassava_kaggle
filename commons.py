import random
import os
import numpy as np
import torch
import cv2


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    return im_rgb


def cutmix(img, df, idx, config, transforms=None):
    target = df.iloc[idx]['label']
    #get random image for mix
    cmix_idx = np.random.randint(idx)
    cmix_img = get_img(os.path.join(config.paths['train_path'], df.iloc[cmix_idx]['image_id']))
    if transforms:
        cmix_img = transforms(image=cmix_img)['image']
    #generate bounding box for cutmix
    lam = np.clip(np.random.beta(config.cmix_params['alpha'], config.cmix_params['alpha']), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox((config.image_size, config.image_size), lam)
    #cutmix
    img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

    rate = 1 - ((bbx2-bbx1) * (bby2-bby1) / (config.image_size * config.image_size))
    target = rate*target + (1.-rate)* df.iloc[cmix_idx]['label']

    return img, target
