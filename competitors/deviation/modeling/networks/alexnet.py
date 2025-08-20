import torch
import torch.nn as nn
from torchvision import models


class feature_alexnet(nn.Module):
    def __init__(self):
        super(feature_alexnet, self).__init__()
        self.net = models.alexnet(pretrained=True)

