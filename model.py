import timm
import torch.nn as nn

class Effnet(nn.Module):
    """
    EfficientNet model by https://arxiv.org/pdf/1905.11946.pdf
    """
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class ViT(nn.Module):
    """
    VisionTransformer model by https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnext(nn.Module):
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
          super().__init__()
          self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
          n_features = self.model.fc.in_features
          self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet(nn.Module):
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
          super().__init__()
          self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
          n_features = self.model.fc.in_features
          self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class NFNet(nn.Module):
    """
    Normalized-Free Nets https://arxiv.org/abs/2102.06171
    """
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
          super().__init__()
          self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
          n_features = self.model.head.fc.in_features
          self.model.head.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


# model factory
def create_model(config):
    if config.model == "effnet":
        model_obj = Effnet
        model = Effnet(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model

    if config.model == 'vit':
        model_obj = ViT
        model = ViT(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model

    if config.model == 'resnext':
        model_obj = Resnext
        model = Resnext(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model

    if config.model == 'resnet':
        model_obj = Resnet
        model = Resnet(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model

    if config.model == 'nfnet':
        model = NFNet(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model
