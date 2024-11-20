import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1) if not conv3d else nn.ReflectionPad3d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2) if not conv3d else nn.Conv3d(features, features, kernel_size=3, stride=2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)