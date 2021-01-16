import torch.nn as nn

def loss_fn(loss_name, config):
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(**config.criterion_params[config.criterion])
    if loss_name == 'labelsmoothloss':
        return LabelSmoothLoss(**config.criterion_params[config.criterion])

### Label Smoothing Loss
class LabelSmoothLoss(nn.Module):
    def __init__(self, class, smoothing=0.1, dim=-1):
        super().__init__()
        self.conf = 1.0 - smoothing
        self.class = class
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, input, target):
        log_probs = input.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.conf)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))
