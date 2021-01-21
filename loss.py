import torch
import torch.nn as nn

def loss_fn(loss_name, config):
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(**config.criterion_params[config.criterion])
    if loss_name == 'labelsmoothloss':
        return LabelSmoothLoss(**config.criterion_params[config.criterion])
    if loss_name == 'bitemperedloss':
        return bi_tempered_logistic_loss(**config.criterion_params[config.criterion])

### Label Smoothing Loss
class LabelSmoothLoss(nn.Module):
    def __init__(self, num_class, smoothing=0.1, dim=-1):
        super().__init__()
        self.conf = 1.0 - smoothing
        self.classes = num_class
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, input, target):
        log_probs = input.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.conf)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))


### Bi-Tempered Loss



class bi_tempered_logistic_loss(nn.Module):
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """
    def __init__(self, t1, t2, label_smoothing=0.0, num_iters=5):
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = 5

    def forward(self, activations, labels):
        if self.label_smoothing > 0.0:
            num_classes = labels.shape[-1]
            labels = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * labels + self.label_smoothing / (num_classes - 1)

        probabilities = self.tempered_softmax(activations, self.t2, self.num_iters)

        temp1 = (self.log_t(labels + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * labels
        temp2 = (1 / (2 - self.t1)) * (torch.pow(labels, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        return torch.sum(loss_values, dim=-1)

    def log_t(self, u, t):
        """Compute log_t for `u`."""
        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(self, u, t):
        """Compute exp_t for `u`."""
        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


    def compute_normalization_fixed_point(self, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """
        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -self.log_t(1.0 / logt_partition, t) + mu


    def compute_normalization(self, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None # not implemented as these values do not occur in the authors experiments...
        else:
            return self.compute_normalization_fixed_point(activations, t, num_iters)


    def tempered_softmax(activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """

        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = self.compute_normalization(activations, t, num_iters)

        return self.exp_t(activations - normalization_constants, t)
