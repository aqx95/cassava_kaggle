import torch.nn as nn

def loss_fn(loss_name, config):
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(**config.criterion_params[config.criterion])
