
import torch
import os
import pandas as pd
import numpy as np
import yaml
import random

import sklearn
import torch.nn.functional as F
from glob import glob
from skimage import io

from sklearn.model_selection import GroupKFold, StratifiedKFold

from data import prepare_dataloader
from trainer import Fitter
from model import CassavaImgClassifier
from config import GlobalConfig


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_folds(train_csv, config):
    df_folds = train_csv.copy()
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed).split(
                        X=df_folds[config.image_col_name], y=df_folds[config.class_col_name])

    for fold, (train_idx, val_idx) in enumerate(skf):
        df_folds.loc[val_idx, 'fold'] = int(fold+1)

    return df_folds


def train_on_fold(df_folds, config, device, fold):
    model = CassavaImgClassifier(config.model_name, config.num_classes, pretrained=True).to(device)
    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    train_loader, valid_loader = prepare_dataloader(train_df, val_df, config)

    fitter = Fitter(model, device, config)
    fold_checkpoint = fitter.fit(train_loader, valid_loader, fold)

    val_df[[str(c) for c in range(config.num_classes)]] = fold_checkpoint["oof_preds"]
    val_df["preds"] = fold_checkpoint["oof_preds"].argmax(1)

    return val_df


def get_acc_score(config, result_df):
    """Get the accuracy of model predictions."""
    preds = result_df["preds"].values
    labels = result_df[config.class_col_name].values
    score = sklearn.metrics.accuracy_score(y_true=labels, y_pred=preds)
    return score


def train_loop(df_folds: pd.DataFrame, config, device, fold_num: int = None, train_one_fold=False):
    """Perform the training loop on all folds."""
    # here The CV score is the average of the validation fold metric.
    cv_score_list = []
    oof_df = pd.DataFrame()
    if train_one_fold:
        _oof_df = train_on_fold(df_folds=df_folds, config=config, device=device, fold=fold_num)
        oof_df = pd.concat([oof_df, _oof_df])
        curr_fold_best_score = get_acc_score(config, _oof_df)
        print("Fold {} OOF Score is {}".format(fold_num,
                                               curr_fold_best_score))
    else:
        # the below for loop guarantees it starts from 1 for fold.
        # https://stackoverflow.com/questions/33282444/pythonic-way-to-iterate-through-a-range-starting-at-1
        for fold in (number+1 for number in range(config.num_folds)):
            _oof_df = train_on_fold(df_folds=df_folds, config=config, device=device, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_best_score = get_acc_score(config, _oof_df)
            cv_score_list.append(curr_fold_best_score)
            print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(fold, curr_fold_best_score))

        print("CV score", np.mean(cv_score_list))
        print("Variance", np.var(cv_score_list))
        print("Five Folds OOF", get_acc_score(config, oof_df))

    oof_df.to_csv("oof.csv")



###MAIN LOOP
if __name__ == '__main__':
    config = GlobalConfig
    seed_everything(config.seed)

    train_csv = pd.read_csv(config.paths['csv_path'])

    df_folds = make_folds(train_csv, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_five_folds = train_loop(df_folds=df_folds, config=config, device=device, fold_num=1, train_one_fold=True)
