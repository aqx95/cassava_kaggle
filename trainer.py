import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import time
import pytz

from loss import loss_fn
from torch.cuda.amp import autocast, GradScaler
from commons import cutmix
from torchcontrib.optim import SWA
from torch.optim.swa_utils import AveragedModel, SWALR
from meter import AverageLossMeter, AccuracyMeter

class Fitter():
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        self.best_acc = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.monitored_metrics = None
        self.val_predictions = None

        if not os.path.exists(self.config.paths['save_path']):
            os.makedirs(self.config.paths['save_path'])
        if not os.path.exists(self.config.paths['log_path']):
            os.makedirs(self.config.paths['log_path'])

        self.loss = loss_fn(config.criterion, config).to(self.device)
        self.scaler = GradScaler()
        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                **config.optimizer_params[config.optimizer])

        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer=self.optimizer,
                                **config.scheduler_params[config.scheduler])

        #SWA
        self.swa_model = AveragedModel(self.model)
        self.swa_start = 0.75*self.config.num_epochs
        self.swa_scheduler = SWALR(self.optimizer, anneal_strategy='cos', anneal_epochs=5, swa_lr=1e-4)

        self.log("Fitter Class prepared. Training with {} \n".format(self.device))

    def fit(self, train_loader, valid_loader, fold):
        self.log('Training on Fold {} with {} \n'.format(fold, self.config.model_name))

        for epoch in range(self.config.num_epochs):
            #get lr
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime("%Y-%m-%d %H-%M-%S")
            self.log('{}\nLR: {}\n'.format(timestamp,lr))

            ##Training
            start_time = time.time()
            avg_train_loss = self.train_epoch(epoch, train_loader)
            end_time = time.time()

            train_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.log("[RESULT]: Train. Epoch {} | Avg Train Summary Loss: {:.6f} | "
                    "Time Elapsed: {}".format(epoch + 1, avg_train_loss, train_elapsed_time))

            ##Validation
            start_time = time.time()
            avg_val_loss, avg_val_acc, val_pred = self.valid_epoch(epoch, valid_loader)
            end_time = time.time()

            val_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.log("[RESULT]: Validation. Epoch: {} | " "Avg Validation Summary Loss: {:.6f} | "
                     "Validation Accuracy: {:.6f} | Time Elapsed: {}".format(
                     epoch + 1, avg_val_loss, avg_val_acc, val_elapsed_time))

            self.val_predictions = val_pred
            self.monitored_metrics = avg_val_acc

            if self.best_loss > avg_val_loss:
                self.best_loss = avg_val_loss

            if self.best_acc < avg_val_acc:
                self.best_acc = avg_val_acc
#                 for path in glob(f'{self.config.paths["save_path"]}/{self.config.model_name}_fold{fold}_epoch*.pt'):
#                     os.remove(path)
                self.save(os.path.join(self.config.paths['save_path'], '{}_fold{}.pt').format(
                        self.config.model_name, fold))

            if self.config.val_step_scheduler and epoch < self.swa_start:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

            else:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            self.epoch += 1

        #computes activation statistics for each bn layer in model
        torch.optim.swa_utils.update_bn(train_loader, self.swa_model)

        fold_best_checkpoint = self.load(os.path.join(self.config.paths['save_path'], '{}_fold{}.pt').format(
                        self.config.model_name, fold))

        return fold_best_checkpoint


    def train_epoch(self, epoch, train_loader):
        self.model.train()
        summary_loss = AverageLossMeter()

        start_time = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs, image_labels =  imgs.to(self.device).float(), image_labels.to(self.device)
            batch_size = image_labels.shape[0]
            mix_decision = np.random.rand()
            if mix_decision < 0.5 and self.config.cutmix and self.epoch>1:
                imgs, image_labels =cutmix(imgs, image_labels, self.config.cmix_params['alpha'])
            with autocast():
                image_preds = self.model(imgs)
                if mix_decision < 0.5 and self.config.cutmix and self.epoch>1:
                    loss = self.loss(image_preds, image_labels[0])*image_labels[2] \
                            + self.loss(image_preds, image_labels[1])*(1-image_labels[2])
                else:
                    loss = self.loss(image_preds, image_labels)

            summary_loss.update(loss.item(), batch_size)
            self.scaler.scale(loss).backward()

            if ((step+1) % self.config.accum_iter == 0 or (step+1) == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.config.train_step_scheduler:
                    self.scheduler.step(epoch+step/len(train_loader))

            end_time = time.time()
            if self.config.verbose:
                if (step % self.config.verbose_step) == 0:
                    description = f"Train Steps {step}/{len(train_loader)} summary_loss: {summary_loss.avg:.3f}, time: {(end_time - start_time):.3f}"
                    pbar.set_description(description)

        return summary_loss.avg


    def valid_epoch(self, epoch, valid_loader):
        self.model.eval()
        summary_loss = AverageLossMeter()
        accuracy_scores = AccuracyMeter()

        start_time = time.time()
        val_gt_label_list, val_preds_softmax_list, val_preds_argmax_list = [], [], []

        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        with torch.no_grad():
            for step, (imgs, image_labels) in pbar:
                imgs = imgs.to(self.device).float()
                image_labels = image_labels.to(self.device).long()
                batch_size = image_labels.shape[0]

                image_preds = self.swa_model(imgs)
                loss = self.loss(image_preds, image_labels)
                summary_loss.update(loss.item(), batch_size)

                y_true = image_labels.cpu().numpy()
                softmax_preds = torch.nn.Softmax(dim=1)(input=image_preds).to("cpu").numpy()
                y_preds = np.argmax(a=softmax_preds, axis=1)
                accuracy_scores.update(y_true, y_preds, batch_size=batch_size)
                val_gt_label_list.append(y_true)
                val_preds_softmax_list.append(softmax_preds)
                val_preds_argmax_list.append(y_preds)
                end_time = time.time()

                if self.config.verbose:
                    if (step % self.config.verbose_step) == 0:
                        description = f"Validation Steps {step}/{len(valid_loader)}, \
                                    summary_loss: {summary_loss.avg:.3f},\
                                    val_acc: {accuracy_scores.avg:.6f} time: {(end_time - start_time):.3f}"
                        pbar.set_description(description)

            val_gt_label_array  = np.concatenate(val_gt_label_list, axis=0)
            val_preds_softmax_array = np.concatenate(val_preds_softmax_list, axis=0)
            val_preds_argmax_array = np.concatenate(val_preds_argmax_list,axis=0)


        return summary_loss.avg, accuracy_scores.avg, val_preds_softmax_array


    def save(self, path):
        """Save the weight for the best evaluation loss."""
        self.swa_model.eval()
        torch.save(
            {
                "model_state_dict": self.swa_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_acc": self.best_acc,
                "best_loss": self.best_loss,
                "epoch": self.epoch,
                "oof_preds": self.val_predictions,
            },
            path,
        )


    def load(self, path):
        """Load a model checkpoint from the given path."""
        checkpoint = torch.load(path)
        return checkpoint


    def log(self, message):
        """Log a message."""
        if self.config.verbose:
            print(message)
        with open(os.path.join(self.config.paths['log_path'],'log.txt'), "a+") as logger:
            logger.write(f"{message}\n")
