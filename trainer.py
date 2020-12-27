import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from torch.cuda.amp import autocast, GradScaler

class Fitter():
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        self.best_acc = 0
        self.best_loss = np.inf
        self.monitored_metrics = None

        self.loss = getattr(torch.nn, config.criterion)(**config.criterion_params[config.criterion])
        self.scaler = GradScaler()
        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                **config.optimizer_params[config.optimizer])

        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer=self.optimizer,
                                **config.scheduler_params[config.scheduler])

        self.log("Fitter Class prepared. Training with {} \n".format(self.device))

    def fit(self, train_loader, valid_loader, fold):
        self.log('Training on Fold {} with {} \n'.format(fold, config.model_name))

        for epoch in range(config.num_epochs):
            #get lr
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime("%Y-%m-%d %H-%M-%S")
            self.log('{}\nLR: {}\n'.format(timestamp,lr))

            ##Training
            start_time = time.time()
            avg_train_loss, avg_train_acc = self.train_epoch(epoch, train_loader)
            end_time = time.time()

            train_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(train_end_time - train_start_time))
            self.log("[RESULT]: Train. Epoch {} | Avg Train Summary Loss: {:.6f} | "
                    "Time Elapsed: {}".format(epoch + 1, avg_train_loss, train_elapsed_time))

            ##Validation
            start_time = time.time()
            avg_val_loss, avg_val_acc, val_pred = self.valid_epoch(epoch, valid_loader)
            end_time = time.time()

            val_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(val_end_time - val_start_time))
            self.log("[RESULT]: Validation. Epoch: {} | " "Avg Validation Summary Loss: {:.6f} | "
                     "Validation Accuracy: {:.6f} | Time Elapsed: {}".format(
                     epoch + 1, avg_val_loss, avg_val_acc_score, val_elapsed_time))

            self.monitored_metrics = avg_val_acc_score

            if self.best_loss > avg_val_loss:
                self.best_loss = avg_best_loss

            if self.best_acc < avg_val_acc:
                self.best_acc = avg_val_acc
                self.save(os.path.join(self.config.paths['save_path'], '{}_fold{}.pt').format(
                        self.config.model_name, fold))

            if self.config.val_step_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

        fold_best_checkpoint = self.load(os.path.join(self.config.paths["save_path"],
                                '{}_fold{}.pt').format(self.config.model_name, fold))

        return fold_bext_checkpoint


    def train_epoch(self, epoch, train_loader):
        self.model.train

        summary_loss = AvergaeLossMeter()

        start_time = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs, image_labels =  imgs.to(self.device).float(), image_labels.to(self.device)
            batch_size = image_labels.shape[0]
            with autocast():
                image_preds = self.model(imgs)
                loss = self.loss(image_preds, image_labels)

            summary_loss.update(loss.item(), batch_size)
            self.scaler.scale(loss).backward()

            if ((step+1) % self.config.accum_iter == 0 or (step+1) == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.config.train_step_scheduler:
                    self.scheduler.step()

            end_time = time.time()
            if self.config.verbose:
                if (step % self.config.verbose_step) == 0:
                    description = f"Train Steps {step}/{len(train_loader)}, \
                                summary_loss: {summary_loss.avg:.3f}, \
                                time: {(end_time - start_time):.3f}"
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

                image_preds = self.model(imgs)
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

                if config.verbose:
                    if (step % config.verbose_step) == 0:
                        description = f"Validation Steps {step}/{len(val_loader)}, \
                                    summary_loss: {summary_loss.avg:.3f},\
                                    val_acc: {accuracy_scores.avg:.6f} time: {(end_time - start_time):.3f}"
                        pbar.set_description(description)

            val_gt_label_array  = np.concatenate(val_gt_label_list, axis=0)
            val_preds_softmax_array = np.concatenate(val_preds_softmax_list, axis=0)
            val_preds_argmax_array = np.concatenate(val_preds_argmax_list,axis=0)


        return summary_loss.avg, accuracy_scores.avg, val_preds_softmax_array


    def save(self, path):
        """Save the weight for the best evaluation loss."""
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_acc": self.best_acc,
                "best_auc": self.best_auc,
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
        with open(self.config.paths['log_path'], "a+") as logger:
            logger.write(f"{message}\n")
