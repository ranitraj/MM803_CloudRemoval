import torch
import torch.nn as nn


# The discriminator in Pix2Pix is a PatchGAN, consisting of CNN Blocks
class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        Creates a CNN Block consisting of a 'convolution' defined by a Sequential Layer comprising:
            1. Convolution2D layer
            2. BatchNorm2D
            3. LeakyReLU

        :param in_channels: Number of input channels supplied
        :param out_channels: Number of output channels supplied
        :param stride: Convolution operation stride
        """
        super(CNNLayer, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )


