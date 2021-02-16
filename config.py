class GlobalConfig:
    seed = 2020
    num_classes = 5
    batch_size = 16
    num_epochs = 10
    accum_iter = 2
    tta = 5

    #Mixing augmentation
    cutmix = True
    cmix_params = {'alpha': 1}
    fmix = True
    fmix_params = {'alpha': 1, 'decay_power': 3., 'shape': (,)}

    # unpack the key dict
    scheduler = 'CosineAnnealingLR'
    scheduler_params = {'StepLR': {'step_size':2, 'gamma':0.3, 'last_epoch':-1, 'verbose':True},

                'ReduceLROnPlateau': {'mode':'max', 'factor':0.5, 'patience':0, 'threshold':0.0001,
                                      'threshold_mode':'rel', 'cooldown':0, 'min_lr':0,
                                      'eps':1e-08, 'verbose':True},

                'CosineAnnealingWarmRestarts': {'T_0':?, 'T_mult':1, 'eta_min':1e-6, 'last_epoch':-1,
                                                'verbose':True}, #train step

                'CosineAnnealingLR':{'T_max':?, 'last_epoch':-1} #validation step
                }

    # do scheduler.step after optimizer.step
    warm_up = False
    warmup_steps = 2
    total = ?
    train_step_scheduler = True
    val_step_scheduler = False

    # optimizer
    optimizer = 'Adam'
    optimizer_params = {'AdamW':{'lr':1e-4, 'betas':(0.9,0.999), 'eps':1e-08,
                                 'weight_decay':1e-6,'amsgrad':False},
                        'SGD':{'lr':0.001, 'momentum':0., 'weight_decay':0.01},
                        'Adam':{'lr':1e-4, 'weight_decay':1e-5}
                        }

    # criterion
    criterion = "crossentropy"
    criterion_params = {'crossentropy': {'weight':None,'size_average':None,
                                             'ignore_index':-100,'reduce':None,
                                             'reduction':'mean'},
                        'labelsmoothloss': {'num_class':5, 'smoothing':0.3, 'dim':-1},
                        'focalcosineloss': {'alpha':1, 'gamma':2, 'xcent':0.1},
                        'bitemperedloss': {'t1':0.8, 't2':2.0, 'smoothing':0.3},
                        'taylorcrossentropy':{'num_class':5, 'smoothing':0.3},
                        'symmetriccrossentropy': {'num_classes':5, 'alpha':0.1, 'beta':1.0, 'smoothing':0.0}
                        }

    image_size = {'vit' : 384,
                  'effnet' : 512,
                  'resnext': 512,
                  'resnet': 512,
                  'nfnet': 512}
    resize = 384
    crop_size = {128:110, 256:200, 512:400}
    verbose = 1
    verbose_step = 1
    num_folds = 5
    image_col_name = 'image_id'
    class_col_name = 'label'
    paths = {'train_path': '../train_images',
             'test_path': '../test_images',
             'csv_path': '../train.csv',
             'log_path': 'log',
             'save_path': 'save',
             'model_weight_path_folder': 'checkpoint'}

    model = 'effnet'
    model_name = 'tf_efficientnet_b4_ns' #'vit_base_patch16_384' #vit_base_patch32_384'
    drop_rate = 0.2
    pretrained = True
    swa = False
    swa_ratio = 0.7
