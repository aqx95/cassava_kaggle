import torch
import torch.nn as nn

def loss_fn(loss_name, config):
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(**config.criterion_params[config.criterion])
    if loss_name == 'labelsmoothloss':
        return LabelSmoothLoss(**config.criterion_params[config.criterion])
    if loss_name == 'bitemperedloss':
        return bi_tempered_logistic_loss
    if loss_name == 'taylorcrossentropy':
        return TaylorCrossEntropyLoss(**config.criterion_params[config.criterion])

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


### Taylor Cross Entropy with Softmax
class TaylorSoftmax(nn.Module):
    def __init__(self, dim=-1, n=2):
        super().__init__()
        assert n%2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            fn = fn + x.pow(i/denor)
        out = fn/fn.sum(dim=self.dim, keepdims=True)
        return out


class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, num_class, n=2, ignore_index=-1, reduction='mean', smoothing=0.05):
        super().__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothLoss(num_class, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss



### Bi-Tempered Loss
# class bi_tempered_logistic_loss(nn.Module):
#     """Bi-Tempered Logistic Loss with custom gradient.
#     Args:
#     activations: A multi-dimensional tensor with last dimension `num_classes`.
#     labels: A tensor with shape and dtype as activations.
#     t1: Temperature 1 (< 1.0 for boundedness).
#     t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
#     label_smoothing: Label smoothing parameter between [0, 1).
#     num_iters: Number of iterations to run the method.
#     Returns:
#     A loss tensor.
#     """
#     def __init__(self, t1, t2, label_smoothing=0.0, num_iters=5):
#         super().__init__()
#         self.t1 = t1
#         self.t2 = t2
#         self.label_smoothing = label_smoothing
#         self.num_iters = 5
#
#     def forward(self, activations, labels):
#         if self.label_smoothing > 0.0:
#             num_classes = labels.shape[-1]
#             labels = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * labels + self.label_smoothing / (num_classes - 1)
#
#         probabilities = self.tempered_softmax(activations, self.t2, self.num_iters)
#
#         temp1 = (self.log_t(labels + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * labels
#         temp2 = (1 / (2 - self.t1)) * (torch.pow(labels, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
#         loss_values = temp1 - temp2
#
#         return torch.sum(loss_values, dim=-1)
#
#     def log_t(self, u, t):
#         """Compute log_t for `u`."""
#         if t == 1.0:
#             return torch.log(u)
#         else:
#             return (u ** (1.0 - t) - 1.0) / (1.0 - t)
#
#     def exp_t(self, u, t):
#         """Compute exp_t for `u`."""
#         if t == 1.0:
#             return torch.exp(u)
#         else:
#             return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))
#
#
#     def compute_normalization_fixed_point(self, activations, t, num_iters=5):
#         """Returns the normalization value for each example (t > 1.0).
#         Args:
#         activations: A multi-dimensional tensor with last dimension `num_classes`.
#         t: Temperature 2 (> 1.0 for tail heaviness).
#         num_iters: Number of iterations to run the method.
#         Return: A tensor of same rank as activation with the last dimension being 1.
#         """
#         mu = torch.max(activations, dim=-1).values.view(-1, 1)
#         normalized_activations_step_0 = activations - mu
#
#         normalized_activations = normalized_activations_step_0
#         i = 0
#         while i < num_iters:
#             i += 1
#             logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
#             normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))
#
#         logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
#
#         return -self.log_t(1.0 / logt_partition, t) + mu
#
#
#     def compute_normalization(self, activations, t, num_iters=5):
#         """Returns the normalization value for each example.
#         Args:
#         activations: A multi-dimensional tensor with last dimension `num_classes`.
#         t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
#         num_iters: Number of iterations to run the method.
#         Return: A tensor of same rank as activation with the last dimension being 1.
#         """
#
#         if t < 1.0:
#             return None # not implemented as these values do not occur in the authors experiments...
#         else:
#             return self.compute_normalization_fixed_point(activations, t, num_iters)
#
#
#     def tempered_softmax(self, activations, t, num_iters=5):
#         """Tempered softmax function.
#         Args:
#         activations: A multi-dimensional tensor with last dimension `num_classes`.
#         t: Temperature tensor > 0.0.
#         num_iters: Number of iterations to run the method.
#         Returns:
#         A probabilities tensor.
#         """
#
#         if t == 1.0:
#             normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
#         else:
#             normalization_constants = self.compute_normalization(activations, t, num_iters)
#
#         return self.exp_t(activations - normalization_constants, t)
#

def log_t(u, t):
    """Compute log_t for `u'."""
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
                (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(normalized_activations - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing = 0.0,
        num_iters=5,
        reduction='mean'):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
        1.0 - labels.to(activations.dtype)],
        dim=-1)
    return bi_tempered_logistic_loss(internal_activations,
            internal_labels,
            t1,
            t2,
            label_smoothing = label_smoothing,
            num_iters = num_iters,
            reduction = reduction)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()
