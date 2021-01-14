import torch
import torch.nn as nn

from model import create_model

if __name__ == '__main__':
    model = create_model(config).to(device)
