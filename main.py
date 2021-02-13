
import torch
import os
import pandas as pd
import numpy as np
import yaml
import seaborn as sns
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
import torch.nn.functional as F
from glob import glob
from skimage import io

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold

from data import prepare_dataloader
from trainer import Fitter
from model import create_model
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
    model = create_model(config).to(device)
    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    train_loader, valid_loader, tta_loader = prepare_dataloader(train_df, val_df, config)
    fitter = Fitter(model, device, config)
    fold_checkpoint = fitter.fit(train_loader, valid_loader, fold)

    print('Getting {}xTTA predictions'.format(config.tta))
    oof_tta = []
    for _ in range(config.tta):
        oof_tta += [test_inference(model, tta_loader, device, config)]

    oof_tta = np.mean(oof_tta, axis=0)
    oof_tta = oof_tta.argmax(1)
    val_df[[str(c) for c in range(config.num_classes)]] = fold_checkpoint["oof_preds"]
    val_df["preds"] = fold_checkpoint["oof_preds"].argmax(1)
    val_df['tta_preds'] = oof_tta

    return val_df


def get_acc_score(config, result_df):
    """Get the accuracy of model predictions."""
    preds = result_df["preds"].values
    labels = result_df[config.class_col_name].values
    score = sklearn.metrics.accuracy_score(y_true=labels, y_pred=preds)
    return score

def get_tta_acc_score(config, result_df):
    preds = result_df["tta_preds"].values
    labels = result_df[config.class_col_name].values
    score = sklearn.metrics.accuracy_score(y_true=labels, y_pred=preds)
    return score

def get_confusion_matrix(config, result_df, normalize=True):
    preds = result_df["tta_preds"].values
    labels = result_df[config.class_col_name].values
    conf_matrix = confusion_matrix(labels, preds)
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(conf_matrix, columns=np.unique(labels), index=np.unique(labels))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    confusion_plot = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
    confusion_plot.figure.savefig(os.path.join(config.paths["save_path"],config.model+'.png'))


def train_loop(df_folds: pd.DataFrame, config, device, fold_num: int = None, train_one_fold=False):
    """Perform the training loop on all folds."""
    # here The CV score is the average of the validation fold metric.
    cv_score_list = []
    oof_df = pd.DataFrame()
    oof_tta = pd.DataFrame()
    print('Image size: {} x {}'.format(config.image_size, config.image_size))

    if train_one_fold:
        _oof_df = train_on_fold(df_folds=df_folds, config=config, device=device, fold=fold_num)
        oof_df = pd.concat([oof_df, _oof_df])
        curr_fold_best_score = get_acc_score(config, _oof_df)
        print("Fold {} OOF Score is {}".format(fold_num,
                                               curr_fold_best_score))
        print("Fold {} OOF TTA Score is {}".format(fold_num, get_tta_acc_score(config, oof_df)))
    else:
        for fold in (number+1 for number in range(config.num_folds)):
            _oof_df = train_on_fold(df_folds=df_folds, config=config, device=device, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_best_score = get_acc_score(config, _oof_df)
            curr_fold_tta_score = get_tta_acc_score(config, _oof_df)
            cv_score_list.append(curr_fold_best_score)
            print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(fold, curr_fold_best_score))
            print("\n\n\nTTA Score for Fold {}: {}\n\n\n".format(fold, curr_fold_tta_score))

        print("CV score", np.mean(cv_score_list))
        print("Variance", np.var(cv_score_list))
        print("Five Folds OOF", get_acc_score(config, oof_df))
        print("Five Folds OOF TTA", get_tta_acc_score(config, oof_df))

    get_confusion_matrix(config, oof_df)
    oof_df.to_csv(f"oof_{config.model_name}.csv")
    return oof_df

#Inference with TTA
def test_inference(model, dataloader, device, config):
    model.eval()
    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    full_pred = []
    batch_pred = []
    for i, (images, labels) in tbar:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        batch_pred+=[y_preds.softmax(1).to('cpu').numpy()]
    full_pred = np.concatenate(batch_pred)
    return full_pred



###MAIN LOOP
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cassava')
    parser.add_argument('--epochs', type=int, default=10,
            help='number of epochs')
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingWarmRestarts",
            help="scheduler")
    parser.add_argument("--optimizer", type=str, default="Adam",
            help="optimizer")
    parser.add_argument("--criterion", type=str, default="crossentropy",
            help="loss function")
    parser.add_argument("--image-size", type=int, default=512,
            help="image size")
    parser.add_argument("--train-one-fold", type=bool, default=False,
            help="Train one fold")
    parser.add_argument("--model-type", type=str, required=True,
            help="model type")
    parser.add_argument("--model-name", type=str, required=True,
            help="batch size when evaluating")
    parser.add_argument("--train-swa", type=bool, default=False,
            help="use SWA model")
    args = parser.parse_args()

    #overwrite
    config = GlobalConfig
    config.num_epochs = args.epochs
    config.scheduler = args.scheduler
    config.optimizer = args.optimizer
    config.criterion = args.criterion
    config.image_size = args.image_size
    config.model = args.model_type
    config.model_name = args.model_name
    config.swa = args.train_swa

    print(args)
    seed_everything(config.seed)
    train_csv = pd.read_csv(config.paths['csv_path'])

    df_folds = make_folds(train_csv, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_five_folds = train_loop(df_folds=df_folds, config=config, device=device,
                        fold_num=1, train_one_fold=args.train_one_fold)
