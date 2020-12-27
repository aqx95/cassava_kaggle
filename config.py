class GlobalConfig:
    seed = 2020
    num_classes = 5
    batch_size = 32
    num_epochs = 6
    accum_iter = 4

    # unpack the key dict
    scheduler = 'ReduceLROnPlateau'
    scheduler_params = {'StepLR': {'step_size':2, 'gamma':0.3, 'last_epoch':-1, 'verbose':True},

                'ReduceLROnPlateau': {'mode':'max', 'factor':0.5, 'patience':0, 'threshold':0.0001,
                                      'threshold_mode':'rel', 'cooldown':0, 'min_lr':1e-5,
                                      'eps':1e-08, 'verbose':True},

                'CosineAnnealingWarmRestarts': {'T_0':10, 'T_mult':1, 'eta_min':0, 'last_epoch':-1,
                                                'verbose':True}}

    # do scheduler.step after optimizer.step
    train_step_scheduler = False
    val_step_scheduler = True

    # optimizer
    optimizer = 'AdamW'
    optimizer_params = {'AdamW':{'lr':0.001, 'betas':(0.9,0.999), 'eps':1e-08,
                                 'weight_decay':0.001,'amsgrad':False}}

    # criterion
    criterion = 'CrossEntropyLoss'
    criterion_params = {'CrossEntropyLoss': {'weight':None,'size_average':None,
                                             'ignore_index':-100,'reduce':None,
                                             'reduction':'mean'}}

    image_size = 512
    resize = 512
    crop_size = {128:110, 256:200, 512:400}
    verbose = 1
    verbose_step = 1
    fold = 5
    image_col_name = 'image_id'
    class_col_name = 'label'
    paths = {'train_path': '../train_images/',
             'test_path': '../test_images/',
             'csv_path': '../train.csv',
             'log_path': 'log.txt',
             'save_path': 'save'
             'model_weight_path_folder': 'checkpoint'}

    model_name = 'tf_efficientnet_b4_ns'
