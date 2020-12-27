
class Fitter():
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        self.loss = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config['optimizer']['lr'],
                                        weight_decay=self.config['optimizer']['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1,
                                                                            eta_min=self.config['optimizer']['min_lr'], last_epoch=-1)
        self.scheduler = None

    def fit(self, train_loader, valid_loader):
        for epoch in range(self.config['train']['epochs']):
            train_epoch(train_loader)
            valid_epoch(valid_loader)


    def train_epoch(self, train_loader):
        self.model.train()

        running_loss = None
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for step, (imgs, image_labels) in pbar:
            imgs, image_labels =  imgs.to(self.device).float(), image_labels.to(self.device).long()
            with autocast():
                image_preds = model(imgs)
                loss = self.loss(image_preds, image_labels)

            self.scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss*.99 + loss.item()*0.01

            if ((step+1) % self.config['train']['accum_iter'] == 0 or (step+1) == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None and self.config['scheduler']['schd_batch_update']:
                    self.scheduler.step()

            if ((step + 1) % self.config['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

        if self.scheduler is not None and not self.config['scheduler']['schd_batch_update']:
        scheduler.step()


    def valid_epoch(self, valid_loader):
        loss_sum = 0
        sample_num = 0
        image_preds_all = []
        image_targets_all = []

        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(self.device).float()
            image_labels = image_labels.to(self.device).long()

            image_preds = self.model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]

            loss = self.loss(image_preds, image_labels)

            loss_sum += loss.item()*image_labels.shape[0]
            sample_num += image_labels.shape[0]

            if ((step + 1) % self.config['verbose_step'] == 0) or ((step + 1) == len(valid_loader)):
                description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                pbar.set_description(description)

        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)
        print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))

        if self.scheduler is not None:
            if config['scheduler']['schd_loss_update']:
                self.scheduler.step(loss_sum/sample_num)
            else:
                self.scheduler.step()
