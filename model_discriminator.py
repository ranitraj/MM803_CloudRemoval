import torch
import torch.nn as nn


# The discriminator in Pix2Pix is a PatchGAN, consisting of CNN Blocks
class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
