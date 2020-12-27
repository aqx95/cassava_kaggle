
import torch
import pandas as pd
import numpy as np
import yaml

import torch.nn.functional as F
from glob import glob
from skimage import io
from scipy.ndimage.interpolation import zoom
from catalyst.data.sampler import BalanceClassSampler

from torch.nn.modules.loss import _WeightedLoss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

from data import prepare_dataloader
from trainer import Fitter
from commons import seed_everything
from model import CassavaImgClassifier


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config



###MAIN LOOP
if __name__ == '__main__':

    train = pd.read_csv('../train.csv')
    config = load_config('config.yaml')
    seed_everything(config['train']['seed'])

    folds = StratifiedKFold(n_splits=config['train']['fold'], shuffle=True,
                            random_state=config['train']['seed']).split(np.arange(train.shape[0]),
                            train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break;

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, config, trn_idx, val_idx,
                                    data_root='../train_images/')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(device))

        model = CassavaImgClassifier(config['model']['model_arch'], train.label.nunique(), pretrained=True).to(device)
        fitter = Fitter(model, device, config)
        fitter.fit(train_loader, val_loader)
